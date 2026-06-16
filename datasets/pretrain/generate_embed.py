import os
import h5py
import torch
import glob
import argparse
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer
import json
import numpy as np

# --- DEFAULT SETTINGS ---
# Local path to T5 model
DEFAULT_T5_PATH = "/home/ubuntu/H_RDT/models/t5-v1_1-xxl"
# ------------------------

def get_t5_embedding(text, tokenizer, model, device):
    # Tokenize and run model
    tokens = tokenizer(
        text, 
        return_tensors="pt", 
        padding="longest", 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        output = model(**tokens)
    
    # Extract last hidden state (Batch, Seq, Dim)
    embedding = output.last_hidden_state.detach().cpu()
    return embedding


# ===================== DENSE MODE (per-phase embeddings) =====================
# Union of all phase vocabularies (bottle + tableware + future). Accepts any known phase.
_DENSE_VALID = ["approach", "grip", "lift", "transport", "rotate", "insert", "place", "release", "withdraw", "stabilize", "extract"]

def _rotate_peak_frac(actions, active, search_end_frac=1.0):
    # search_end_frac: only look for the rotation peak BEFORE this frac
    # (excludes the withdraw/retract region, whose wrist motion also spins).
    wrot = actions[:, 3:9] if active == "left" else actions[:, 27:33]
    T = len(wrot)
    if T < 3: return None
    rv = np.zeros(T); rv[1:] = np.linalg.norm(np.diff(wrot, axis=0), axis=1)
    rv = np.convolve(rv, np.ones(5)/5, mode="same")
    cutoff = max(2, int(T * search_end_frac))
    return int(np.argmax(rv[:cutoff])) / T

def _active_hand(a):
    rng = a.max(0) - a.min(0)
    return "left" if rng[0:24].sum() >= rng[24:48].sum() else "right"

def _process_phase_list(phases, T, tokenizer, model, device):
    """Per-track helper: phases -> (names, frames, phase_pooled, rats). Used by bimanual path."""
    names, frames, embs, masks, rats = [], [], [], [], []
    for ph in phases:
        n = ph.get("name")
        if n not in _DENSE_VALID: continue
        s = max(0, int(round(ph["start_frac"] * T)))
        e = min(T, max(s + 1, int(round(ph["end_frac"] * T))))
        emb = get_t5_embedding(ph["rationale"], tokenizer, model, device).squeeze(0)
        tk = tokenizer(ph["rationale"], return_tensors="pt", padding=False, truncation=True, max_length=128)
        names.append(n); frames.append((s, e)); embs.append(emb)
        masks.append(tk.attention_mask.squeeze(0)); rats.append(ph["rationale"])
    phase_pooled = []
    for emb, m in zip(embs, masks):
        mm = m.unsqueeze(-1).float()
        pooled = (emb * mm).sum(0) / mm.sum().clamp(min=1)
        phase_pooled.append(pooled)
    phase_pooled = torch.stack(phase_pooled, dim=0) if phase_pooled else None
    return names, frames, phase_pooled, rats

def _dense_process_ep_bimanual(ep, ann, hdf5_path, tokenizer, model, device):
    """Bimanual: ann = {left_hand:{phases}, right_hand:{phases}} -> dual-track _dense.pt."""
    with h5py.File(hdf5_path, "r") as f:
        actions = np.array(f["actions_48d"][:])
    T = len(actions)
    lp = ann.get("left_hand", {}).get("phases", [])
    rp = ann.get("right_hand", {}).get("phases", [])
    nL, fL, ppL, ratL = _process_phase_list(lp, T, tokenizer, model, device)
    nR, fR, ppR, ratR = _process_phase_list(rp, T, tokenizer, model, device)
    flags = {}
    if len(nL) < 1 or len(nR) < 1:
        flags["low_phase_count"] = min(len(nL), len(nR))
    return {"episode": ep, "episode_len": T, "bimanual": True,
            "phase_names_left": nL, "phase_frames_left": fL, "phase_pooled_left": ppL, "phase_rationales_left": ratL,
            "phase_names_right": nR, "phase_frames_right": fR, "phase_pooled_right": ppR, "phase_rationales_right": ratR,
            "flags": flags}

def _dense_process_ep(ep, phases, hdf5_path, tokenizer, model, device):
    with h5py.File(hdf5_path, "r") as f:
        actions = np.array(f["actions_48d"][:])
    T = len(actions); active = _active_hand(actions)
    names, frames, embs, masks, rats = [], [], [], [], []
    for ph in phases:
        n = ph.get("name")
        if n not in _DENSE_VALID: continue
        s = max(0, int(round(ph["start_frac"] * T)))
        e = min(T, max(s + 1, int(round(ph["end_frac"] * T))))
        emb = get_t5_embedding(ph["rationale"], tokenizer, model, device).squeeze(0)
        tk = tokenizer(ph["rationale"], return_tensors="pt", padding=False, truncation=True, max_length=128)
        names.append(n); frames.append((s, e)); embs.append(emb)
        masks.append(tk.attention_mask.squeeze(0)); rats.append(ph["rationale"])
    # Per-phase pooled vectors for Option-D per-token alignment:
    # masked-mean each phase's (seq, 4096) T5 sequence -> (4096,) per phase.
    phase_pooled = []
    for emb, m in zip(embs, masks):
        mm = m.unsqueeze(-1).float()                      # (seq,1)
        pooled = (emb * mm).sum(0) / mm.sum().clamp(min=1)  # (4096,)
        phase_pooled.append(pooled)
    phase_pooled = torch.stack(phase_pooled, dim=0) if phase_pooled else None  # (n_phases, 4096)

    pooled_emb = get_t5_embedding(" ".join(rats), tokenizer, model, device).squeeze(0) if rats else None
    flags = {}
    if len(names) < 2: flags["low_phase_count"] = len(names)
    if "rotate" in names:
        # cap the kinematic search at the withdraw phase start (excl. retract spin); fallback 0.85
        if "withdraw" in names:
            wi = names.index("withdraw"); search_end = frames[wi][0] / T
        else:
            search_end = 0.85
        rpf = _rotate_peak_frac(actions, active, search_end_frac=search_end)
        if rpf is not None:
            ri = names.index("rotate"); rs, re_ = frames[ri]; pk = rpf * T
            flags["rotate_peak_frac"] = round(rpf, 3); flags["rotate_window"] = [round(rs/T,3), round(re_/T,3)]
            if not (rs <= pk <= re_): flags["rotate_mismatch"] = True
    return {"episode": ep, "active_hand": active, "episode_len": T,
            "phase_names": names, "phase_frames": frames, "phase_embeddings": embs,
            "phase_attn_masks": masks, "phase_rationales": rats,
            "pooled_embedding": pooled_emb, "phase_pooled": phase_pooled, "flags": flags}

def run_dense(data_root, baseline_root, tokenizer, model, device):
    # read phased JSON from data_root (processed_reasoning); read actions + write
    # {ep}_dense.pt co-located in baseline_root (processed_baseline) by matching session/ep.
    jsons = glob.glob(os.path.join(data_root, "**", "reasoning_phased.json"), recursive=True)
    print(f"DENSE: found {len(jsons)} sessions with phased annotations")
    total = flo = frot = miss = 0; summ = []
    for jp in jsons:
        sd = os.path.dirname(jp); sn = os.path.basename(sd)
        # mirror the session path under baseline_root (handles part1/ and test/ subdirs)
        rel = os.path.relpath(sd, data_root)
        base_sd = os.path.join(baseline_root, rel)
        ann = json.load(open(jp))
        for ep, d in tqdm(ann.items(), desc=sn[:22]):
            hp = os.path.join(base_sd, f"{ep}.hdf5")  # actions from BASELINE
            if not os.path.exists(hp): miss += 1; continue
            try:
                if isinstance(d, dict) and ("left_hand" in d or "right_hand" in d):
                    rec = _dense_process_ep_bimanual(ep, d, hp, tokenizer, model, device)
                else:
                    rec = _dense_process_ep(ep, d.get("phases", []), hp, tokenizer, model, device)
            except Exception as e: print(f"  FAIL {ep}: {repr(e)[:60]}"); continue
            torch.save(rec, os.path.join(base_sd, f"{ep}_dense.pt")); total += 1  # co-located
            if "low_phase_count" in rec["flags"]: flo += 1
            if rec["flags"].get("rotate_mismatch"):
                frot += 1; summ.append(f"{sn}/{ep}: rot win {rec['flags']['rotate_window']} misses peak {rec['flags']['rotate_peak_frac']}")
    print(f"\nDENSE DONE: {total} _dense.pt written into {baseline_root}")
    print(f"  low_phase(<2): {flo} | rotate_mismatch: {frot} | hdf5_missing_in_baseline: {miss}")
    for x in summ[:15]: print("  ", x)
    for x in summ[:20]: print("  ", x)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate .pt embeddings from HDF5 attributes.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the processed data folder")
    parser.add_argument("--t5_path", type=str, default=DEFAULT_T5_PATH, help="Path to local T5 model")
    parser.add_argument("--dense", action="store_true", help="Dense mode: per-phase embeddings from reasoning_phased.json")
    parser.add_argument("--baseline_root", type=str, default=None, help="processed_baseline root: where actions live + dense .pt written")
    parser.add_argument("--out", type=str, default=None, help="Output dir for dense .pt (dense mode)")
    args = parser.parse_args()

    # 1. Load Model
    print(f"🚀 Loading T5 Model from: {args.t5_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.t5_path)
        model = T5EncoderModel.from_pretrained(args.t5_path).to(device)
    except Exception as e:
        print(f"❌ Failed to load T5 model: {e}")
        return
    model.eval()

    if args.dense:
        assert args.baseline_root, "--baseline_root required (processed_baseline path)"
        run_dense(args.data_root, args.baseline_root, tokenizer, model, device)
        return

    # 2. Find Files
    search_path = os.path.join(args.data_root, "**", "*.hdf5")
    files = glob.glob(search_path, recursive=True)
    print(f"📂 Found {len(files)} HDF5 episodes in {args.data_root}")

    if len(files) == 0:
        print("⚠️ No files found! Check your path.")
        return

    # 3. Process Loop
    success_count = 0
    for file_path in tqdm(files, desc="Encoding"):
        try:
            # Read text from HDF5
            with h5py.File(file_path, 'r') as f:
                if 'language_instruction' not in f.attrs:
                    # Skip if missing (or use a fallback if you really want)
                    continue
                
                raw_text = f.attrs['language_instruction']
                
                # Handle decoding if stored as bytes
                if isinstance(raw_text, bytes):
                    text = raw_text.decode('utf-8')
                else:
                    text = str(raw_text)

            # Generate .pt file
            save_path = file_path.replace('.hdf5', '.pt')
            
            # Optimization: Skip if .pt already exists? 
            # (Uncomment next 2 lines to resume interrupted jobs)
            # if os.path.exists(save_path):
            #     continue

            embed = get_t5_embedding(text, tokenizer, model, device)
            # Also save token IDs for reasoning auxiliary loss
            tokens = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=150
            )
            token_ids = tokens.input_ids.squeeze(0)  # (seq_len,)
            attn_mask = tokens.attention_mask.squeeze(0)  # (seq_len,)
            torch.save({
                "instruction": text,
                "embeddings": embed.squeeze(0),
                "task_name": text,
                "token_ids": token_ids,
                "attention_mask": attn_mask
            }, save_path)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")

    print(f"✅ Completed! Generated {success_count}/{len(files)} embeddings.")

if __name__ == "__main__":
    main()