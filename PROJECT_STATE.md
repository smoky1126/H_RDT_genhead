# Reasoning-VLA (genhead) — State Handoff

## THE RESULT (verified, positive)
move_can_pot transfer, batch-64, seed-42, matched eval config:
- R2 noLSS baseline: 15%, 16% (two runs)
- R5 genhead: 23%, 25% (two runs)
- Gap: +8.5pts (~55% relative), non-overlapping, replicated across evals
- This is the first efficacy evidence. Bottle was a tie (82% vs 79%, ceiling).

## METHOD
Generative reasoning head: reads 16 action tokens (hidden[:,1:]), generates
per-phase Gemini rationale during Stage-2 training, DISCARDED at inference (zero cost).
= LaRA-VLA/DHRD/ECoT-Lite mechanism (published, not novel).
Pipeline: S1 EgoDex → S2 (bottle + reasoning head) → S3 RoboTwin finetune (head dropped).

## KEY FIXES THAT MADE IT WORK
- Head reads action tokens NOT state token (Run D's bug)
- input LayerNorm (action tokens have norm ~59,000; diverges without it)
- per-phase rationale labels NOT episode-level
- effective batch-64 verified in resolved DeepSpeed config (8×4×2), not just CLI flag

## KNOWN FLAWS / CAVEATS (honest)
1. LABELING FLAW: 48-frame window labeled by START-FRAME phase, but windows span 2-4 phases
   (diagnostic: 4/4 bottle phases, 7/7 tableware, 4/5 groceries SHORTER than 48-frame window).
   R5's +8.5 was learned from THESE flawed labels → result is a FLOOR not ceiling.
2. Stage-2 END-SPIKE: diff_loss rose 0.25→0.5 in final 1k steps (λ=1.0 destabilizing).
   R5 finetuned from checkpoint-10000 (the spiked one). checkpoint-8000 may be cleaner.
3. 0.74-cosine rationales: phase sentences near-identical → caps how distinct structure can be.
4. SINGLE TRAINING SEED (both R5, R2 = seed-42). Eval replicates test rollout noise, NOT seed noise.
5. reasoning_loss plateaued high at 5.44 (should be lower).

## NEXT EXPERIMENTS (ranked by expected impact)
1. [FREE] Finetune move_can_pot from S2 checkpoint-8000 (pre-end-spike) vs 10000. Tests if end-spike cost points.
2. [FREE] Checkpoint sweep: finetune from S2 ckpt 6k/8k/10k, find best-transferring.
3. [REBUILD] grouped-3b: group 16 tokens by phase, emit ONE rationale per phase-group present.
   Fixes labeling. Essential for HMD-2/3. Plausibly widens +8.5. Design settled, not built.
4. [ROOT] Regenerate CONTRASTIVE rationales to break 0.74-cosine ceiling.
5. [CREDIBILITY] seed-43 replication of the +8.5 result. Not a number-lever, a defensibility-lever.

## PATHS
- Working repo (R5=23/25%): ~/H_RDT_genhead
- New repo for grouped-3b: ~/H_RDT_grouped (forked, code real, checkpoints/models/bak symlinked)
- CRITICAL: checkpoints is a SYMLINK to ~/H_RDT/checkpoints. NEVER rm through it (deletes originals).
- S2 genhead checkpoints: ~/H_RDT/checkpoints/adjusting_bottle/pretrains/S2_genhead_bottle_seed42/{checkpoint-2000..10000}
- Server: LXD "andrew", 2× RTX 6000 Ada 48GB, conda env hrdt, batch-64 config = configs/zero2_bs64.json

## TARGET: IEEE CYBER (bottle scope), HMD-2/3 → ICRA extension
