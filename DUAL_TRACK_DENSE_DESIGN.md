# Dual-Track Dense LSS — Design (Bimanual Extension)

## Goal
Extend dense LSS to ASYMMETRIC BIMANUAL tasks (HMD-3 groceries: one hand stabilizes,
other extracts) WITHOUT breaking the single-arm pipeline (bottle/tableware, 90/54/26).

## Core safety property
Single-arm = dual-track with the second track ALL-MASKED. The second masked-cosine term
evaluates to exactly 0, so single-arm loss == current loss, byte-for-byte. Existing
single-arm _dense.pt files DO NOT CHANGE and flow through an unchanged code path.

## Current pipeline (CONFIRMED from local code)
_dense.pt: phase_frames list[(s,e)], phase_pooled (n,4096)
_build_token_phase_targets: each of K=16 tokens -> phase_of(f0); targets[t]=phase_pooled[dom];
  mask[t]=True iff all UP=3 frames in one phase. returns (targets(K,4096), mask(K,))
collate: batch dense_lsa_embeds (B,K,4096), dense_lsa_mask (B,K)
loss: projected=lsa_head(hidden,per_token=True); lsa=masked_cos(projected,targets,mask)

## Dual-track design (5 changes)

### 1. generate_embed.py (bimanual writes both tracks)
new keys: phase_frames_left, phase_pooled_left(n_L,4096), phase_frames_right, phase_pooled_right(n_R,4096)
single-arm: UNCHANGED.

### 2. egodex_dataset._build_token_phase_targets (extract _assign, dual branch)
_assign(frames,pooled) = exact current logic.
old schema (no phase_pooled_left): return _assign(frames,pooled)  <- IDENTICAL to today
bimanual: tL,mL=_assign(left); tR,mR=_assign(right); return (tL,mL,tR,mR)

### 3. dataset.py collate (stack both)
single-arm -> LEFT track, RIGHT = zeros + mask0
batch: dense_lsa_embeds_L/mask_L, dense_lsa_embeds_R/mask_R

### 4. hrdt_runner.compute_loss (dual cosine summed)
projected = lsa_head(hidden, per_token=True)  # ONE projection
lsa_L = masked_cos(projected, targets_L, mask_L)
lsa_R = masked_cos(projected, targets_R, mask_R)  # =0 when mask_R all-zero
lsa_loss = lsa_L + lsa_R
single-arm: lsa_R = (loss*0).sum()/0.clamp(min=1) = 0 -> lsa_loss = lsa_L = TODAY

### 5. annotate_reasoning.py (bimanual prompt)
reasoning_phased.json: {left_hand:{phases}, right_hand:{phases}}; single-arm UNCHANGED.

## Verification gates (single-arm = oracle)
G1 schema:  bimanual ep has L+R; bottle ep byte-identical          (min)
G2 assign:  bottle old-schema identical; bimanual 4-tuple          (min)
G3 collate: mixed batch stacks; single-arm mask_R all-False        (min)
G4 loss:    single-arm dual-loss == current loss (allclose) <-KEY  (min)
G5 regress: bottle dual-track -> 90/54/26                          (hrs, GPU)

## Build order
1. discovery -> lock groceries vocab (n_L, n_R; hands distinct)
2. generate_embed bimanual (G1)
3. egodex_dataset _assign+dual (G2)
4. dataset collate (G3)
5. hrdt_runner dual loss (G4)
6. annotate bimanual prompt
7. G5 bottle regression (GPU)
8. groceries dense -> R4

## Locked choices
- ONE projection -> two targets (LSAHead unchanged)
- two masked-cosine SUMMED (each self-normalized)
- single-arm _dense.pt NOT regenerated (old schema -> unchanged path)
- idle/active by --mode flag (human-set)

────────────────────────────────────────────────────────────────────────
## USAGE — Bimanual Dense LSS Pipeline (end-to-end)
────────────────────────────────────────────────────────────────────────

### Step 1: Discover phases (per-hand)
    cd ~/human-policy/cet
    python3 discover_phases.py \
      --task_dir ~/human-policy/data/recordings/processed_baseline_unpack_groceries/part1 \
      --tag groceries --discover --per_block 4 --mode bimanual
    # -> discovered_phases_groceries.json (pools both hands' phase names)

### Step 2: Review + lock vocab
    python3 discover_phases.py --tag groceries --propose
    # (edit proposed_merges_groceries.txt - reject merges you disagree with)
    python3 discover_phases.py --tag groceries --lock
    # -> phase_vocab_groceries.json

### Step 3: Annotate (per-hand phased reasoning)
    python3 annotate_reasoning.py --phased --mode bimanual \
      --phase_set groceries --target_dir <session_dir>
    # -> reasoning_phased.json with {left_hand:{phases}, right_hand:{phases}}

### Step 4: Build dual-track _dense.pt
    cd ~/H_RDT
    python3 datasets/pretrain/generate_embed.py --dense \
      --data_root     ~/human-policy/data/recordings/processed_baseline_unpack_groceries \
      --baseline_root ~/human-policy/data/recordings/processed_baseline_unpack_groceries
    # auto-detects bimanual -> writes phase_pooled_left/right + phase_frames_left/right
    # (single-arm annotations still write old single-track schema)

### Step 5: Train R4 dense (dual-track loss)
    # pretrain.sh: EGODEX_DATA_ROOT=processed_baseline_unpack_groceries, --use_dense_lsa
    bash pretrain.sh
    # dual-track loss: lsa_track1 (hand-1) + lsa_track2 (hand-2)
    # single-arm data: lsa_track2 = 0 (unchanged behavior)

### Key behaviors
- AUTO-DETECTION: generate_embed --dense detects bimanual ({left_hand,right_hand})
  vs single-arm ({phases}) automatically. Same command for both.
- SINGLE-ARM UNTOUCHED: bottle/tableware -> single-track schema, unchanged path.
  Existing single-arm _dense.pt files do NOT need regeneration.
- --mode bimanual flag is human-set at discovery + annotation. Embed + training
  auto-handle bimanual from the data structure (no flag needed downstream).

### Validation status
- VERIFIED (unit): G2 token-assign byte-identical, G3 collate, G4 loss allclose.
- PENDING: G5 bottle regression (90/54/26 through dual-track, needs GPU);
           G1 full bimanual data test (needs locked groceries vocab).
