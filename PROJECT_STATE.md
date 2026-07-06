# Reasoning-VLA (genhead) — State (facts only)

## RESULT (move_can_pot transfer; batch-64; seed-42; matched eval config)
- R2 noLSS baseline: 15%, 16% (two eval runs)
- R5 genhead: 23%, 25% (two eval runs)
- Absolute gap: +8.5 percentage points (mean 24.0% vs 15.5%)
- Relative gap: ~55%
- The two eval runs per condition do not overlap.
- Bottle (in-distribution): R5 82% vs R2 79% (single eval each).

## METHOD
- Generative reasoning head reads the 16 action tokens (hidden[:,1:]) and generates the
  per-phase Gemini rationale text during Stage-2 training; head discarded at inference (zero inference cost).
- Backbone: H-RDT (DINO-SigLIP + T5-XXL + 2B Diffusion Transformer, flow-matching,
  hidden=2176, 16 action tokens, upsample_rate=3 -> 48-frame action chunk, action_dim 48 human -> 14 robot).
- Pipeline: Stage 1 (EgoDex human pretrain) -> Stage 2 (bottle + reasoning head; this repo's method)
  -> Stage 3 (RoboTwin robot finetune; action encoder/decoder reinitialized from scratch).
- Loss: L = L_flow + lambda * L_reason, with lambda = 1.0.

## CONFIG (verified from resolved DeepSpeed config, not the CLI flag)
- Effective batch 64 = micro 8 x grad_accum 4 x 2 GPU. Config file: configs/zero2_bs64.json.
- LoRA r8 / alpha16, bf16, ZeRO-2. lr 5e-4 (Stage 2) / 1e-4 (Stage 3). 10000 steps each. seed 42.
- Stage 2 init from EgoDex checkpoint-500000 (same as R2 baseline; verified by log).
- Stage 3: 50 RoboTwin episodes. Tasks: move_can_pot (transfer), adjust_bottle (in-distribution).

## MEASUREMENTS
- Phase decodability (frozen leave-one-object-out linear probe), genhead vs no-reasoning baseline:
  bottle    +0.188 -> +0.375  (delta +0.187)
  tableware +0.076 -> +0.128  (delta +0.052)
  groceries +0.022 -> +0.057  (delta +0.035)
- Per-phase decodability delta (bottle): rotate +0.295, grip +0.285, withdraw +0.143, approach +0.001.
  Absolute final recall lowest for grip (0.525).
- Stage-2 genhead final losses: diff_loss 0.273, reasoning_loss 5.44.
  R2 baseline Stage-2 diff_loss settled ~0.01-0.07.
- Stage-2 genhead diff_loss trajectory: stable ~0.25 for steps ~500-9000, then rose to ~0.5 in the final ~1000 steps ("end-spike").
- Stage-3 R5 finetune diff_loss tracked the R2 finetune curve.

## KNOWN FLAWS (measured)
- Labeling: the 48-frame action chunk is labeled by the START-FRAME phase only.
  Phase mean durations vs the 48-frame window:
    bottle:    grip 12.9f, approach 25.4f, withdraw 31.8f, rotate 47.2f  (4/4 phases < window)
    tableware: release 8.3f, lift 8.3f, grip 9.1f, approach 18.8f, insert 21f, withdraw 26f, transport 28f (7/7 < window)
    groceries: grip 11f, release 11f, approach 16f, withdraw 17f, stabilize 189f (4/5 < window)
  => most windows span 2-4 phases; the label describes only the start-frame phase.
- The R5 +8.5pt result was produced under this labeling.
- Phase rationales: ~0.74 average pairwise cosine similarity (near-identical sentences).
- Single training seed (both R5 and R2 = seed-42). The two eval runs per condition test rollout
  variance, not training-seed variance.
- Stage-2 reasoning_loss plateaued at 5.44 (did not converge lower).

## CHECKPOINTS (important constraint)
- Only the FINAL Stage-2 genhead checkpoint (step 10000) is retained. Steps 2000-8000 were deleted.
- R5 finetune used checkpoint-10000 (the post-end-spike checkpoint).
- Testing an earlier / pre-end-spike backbone therefore requires RE-RUNNING Stage-2 (not free).
- checkpoints/, bak/, models/t5-v1_1-xxl/ are SYMLINKS to ~/H_RDT/. Do not delete through them.

## ENVIRONMENT
- Repo on GitHub: smoky1126/H_RDT_genhead. Server: LXD "andrew", 2x RTX 6000 Ada 48GB, conda env hrdt, SSH only.

## VENUE
- Target: IEEE CYBER (bottle scope). HMD-2 (tableware) / HMD-3 (groceries) reserved for a future ICRA extension.
