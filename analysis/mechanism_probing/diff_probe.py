"""FINAL probe: isolate the dense-vs-pooled representational difference (the lambda=0.1 effect).
import sys, os as _os; sys.path.insert(0, _os.path.expanduser("~/H_RDT"))
Same windows through R3 (pooled) and R4 (dense). Compute per-token (dense - pooled) difference.
Pre-committed test: is that difference vector phase-structured? (decodability + silhouette ON THE DIFF)"""
import os, numpy as np, torch, yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def extract_fixed(ckpt_dir, device, seed=0, max_batches=400, bs=8):
    """Extract action-token hiddens with a FIXED data order (same windows every call)."""
    from datasets.dataset import VLAConsumerDataset, DataCollatorForVLAConsumerDataset
    from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone
    from models.hrdt_runner import HRDTRunner
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(open(os.path.expanduser('~/H_RDT/configs/hrdt_pretrain.yaml')))
    venc = DinoSigLIPViTBackbone(vision_backbone_id='dino-siglip',
        image_resize_strategy='letterbox' if cfg['dataset']['image_aspect_ratio']=='pad' else 'resize-naive',
        default_image_size=384).to(device, torch.bfloat16).eval()
    it = venc.get_image_transform()
    ds = VLAConsumerDataset(config=cfg, image_transform=it, num_cameras=cfg['common']['num_cameras'],
        dataset_type='egodex', image_aug=False, image_corrupt_severity=None,
        upsample_rate=3, val=False, use_precomp_lang_embed=True, task_name='adjust_bottle')
    coll = DataCollatorForVLAConsumerDataset(use_precomp_lang_embed=True)
    g = torch.Generator(); g.manual_seed(seed)   # FIXED order -> same windows for both backbones
    dl = DataLoader(ds, batch_size=bs, collate_fn=coll, num_workers=0, shuffle=True, generator=g)
    runner = HRDTRunner.from_pretrained(ckpt_dir).to(device, torch.bfloat16).eval()

    H, Y, nb = [], [], 0
    torch.manual_seed(seed)   # fix noise/timestep sampling too, for fair diff
    with torch.no_grad():
        for batch in dl:
            if 'dense_lsa_embeds' not in batch: continue
            img = {k: v.to(device, torch.bfloat16) for k,v in batch['images'].items()}
            st  = batch['states'].to(device, torch.bfloat16)
            act = batch['actions'].to(device, torch.bfloat16)
            lang= batch['lang_embeds'].to(device, torch.bfloat16)
            lmask = batch['lang_attn_mask'].to(device)
            tgt = batch['dense_lsa_embeds'].float(); tmask = batch['dense_lsa_mask']
            enc_in = {k: (v.view(-1,*v.shape[-3:]) if v.dim()==5 else v) for k,v in img.items()}
            feats = venc(enc_in); image_tokens = feats.view(st.shape[0], -1, venc.embed_dim)
            img_c = runner.img_adapter(image_tokens); lang_c = runner.lang_adapter(lang)
            noise = torch.randn(act.shape, dtype=torch.bfloat16, device=device, generator=torch.Generator(device=device).manual_seed(seed+nb))
            ts = torch.full((act.shape[0],), 0.5, device=device, dtype=torch.bfloat16)  # FIXED timestep -> deterministic
            b = ts.view(-1,1,1); noisy = (act*b + noise*(1-b)).to(torch.bfloat16)
            sa = torch.cat([runner.action_encoder.encode_state(st), runner.action_encoder.encode_action(noisy)], dim=1)
            _, hidden = runner.model(sa, ts, img_c=img_c, lang_c=lang_c, lang_attn_mask=lmask, return_hidden=True)
            ah = hidden[:,1:,:].float().cpu()
            for i in range(ah.shape[0]):
                t_i = tgt[i].cpu(); uniq = torch.unique(t_i, dim=0)
                pid = torch.cdist(t_i, uniq).argmin(dim=1)
                for t in range(16):
                    if tmask[i,t]: H.append(ah[i,t].numpy()); Y.append(int(pid[t]))
            nb += 1
            if nb >= max_batches: break
    del runner, venc; torch.cuda.empty_cache()
    return np.array(H), np.array(Y)

if __name__ == '__main__':
    dev='cuda'
    print("extracting R3 (pooled), fixed windows..."); H3,Y3 = extract_fixed(os.path.expanduser('~/H_RDT/checkpoints/pretrain_human_reasoning_lsa_runE_mar23'), dev)
    print("extracting R4 (dense), SAME windows...");   H4,Y4 = extract_fixed(os.path.expanduser('~/H_RDT/checkpoints/pretrain_dense_lsa_june6'), dev)
    n = min(len(H3), len(H4))
    assert (Y3[:n]==Y4[:n]).mean() > 0.99, f"window mismatch! labels agree {100*(Y3[:n]==Y4[:n]).mean():.1f}%"
    H3,H4,Y = H3[:n],H4[:n],Y3[:n]
    np.savez('diff_cache.npz', H3=H3,H4=H4,Y=Y)
    D = H4 - H3                          # the lambda=0.1 effect, isolated
    print(f"\n{n} matched tokens. |diff|/|hidden| = {np.linalg.norm(D)/np.linalg.norm(H3):.4f} (how big the effect is)")

    # PRE-COMMITTED metric 1: is the DIFFERENCE phase-decodable?
    Ds = StandardScaler().fit_transform(D)
    Xtr,Xte,ytr,yte = train_test_split(Ds,Y,test_size=0.3,random_state=0,stratify=Y)
    acc = LogisticRegression(max_iter=2000).fit(Xtr,ytr).score(Xte,yte)
    # PRE-COMMITTED metric 2: silhouette of the difference by phase
    idx = np.random.RandomState(0).choice(len(Ds), min(4000,len(Ds)), replace=False)
    sil = silhouette_score(Ds[idx], Y[idx])
    print(f"DIFF phase-decodability: {100*acc:.1f}% (chance=25%)")
    print(f"DIFF silhouette:         {sil:.4f}")
    print("\nINTERPRET: if diff is phase-structured (acc>>25%, sil>0), the lambda=0.1 effect")
    print("is phase-specific (dense moves grip differently than rotate). If acc~25%/sil~0,")
    print("the effect is NOT phase-structured -> mechanism not in per-phase representation.")
