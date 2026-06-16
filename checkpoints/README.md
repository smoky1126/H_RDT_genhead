# H-RDT Checkpoint Directory — Latent Semantic Scaffolding (LSS) Project

Backbone: H-RDT (2B diffusion transformer, flow-matching, T5-XXL command encoder,
DINO+SigLIP vision, hidden_size=2176, 16 action tokens, upsample_rate=3).
3-stage pipeline: Stage 1 EgoDex pretrain → Stage 2 AVP human-data pretrain (LSS applied here)
→ Stage 3 robot finetune (RoboTwin 2.0, aloha-agilex 14-DOF, 50 episodes / 22000 steps).

## RUN LEGEND (the R-ladder, applied per human dataset)
- **R1** = H-RDT baseline: EgoDex pretrain → robot finetune. NO AVP, NO LSS.
- **R2** = +AVP human-data pretrain (Stage 2), NO LSS. (kinematics only)
- **R3** = +AVP + **Pooled LSS**: 16 action tokens pooled to ONE vector, aligned to ONE
           episode reasoning embedding.
- **R4** = +AVP + **Dense LSS** ★PROPOSED: each of 16 action tokens aligned to ITS phase's
           reasoning embedding (per-token phase alignment). Zero inference overhead
           (LSAHead discarded after Stage 2).

LSS = training-time auxiliary loss aligning action-token hidden states to frozen T5-XXL
reasoning embeddings via LSAHead projection (2176→4096). L_total = L_diffusion + λ·L_LSS (λ=0.1).

═══════════════════════════════════════════════════════════════════════
## _base/  — EgoDex foundation (shared, dataset-agnostic)
═══════════════════════════════════════════════════════════════════════
### egodex_foundation/
  EgoDex Stage-1 pretrain, checkpoint-500000. THE base backbone all runs start from.

### finetunes/  — R1 / RunA baselines (EgoDex base → robot finetune; NO AVP, NO LSS)
  These are dataset-AGNOSTIC baselines (same recipe, different robot task). R1 ≡ RunA recipe.
  - R1_baseline_shake_bottle_34pct        shake_bottle, 34%   (transfer task)
  - R1_baseline_move_can_pot_9pct         move_can_pot, 9.1%  (transfer task, hard)
  - R1_baseline_stack_bowls_two_23pct     stack_bowls_two, 23%
  - R1_baseline_put_object_cabinet_34pct  put_object_cabinet, 34%
  - RunA_baseline_adjust_bottle_70pct     adjust_bottle, 70%  (= R1 recipe, in-distribution;
                                          first row of the adjust_bottle ablation study)

═══════════════════════════════════════════════════════════════════════
## adjusting_bottle/  — HMD-1 (single-arm bottle manipulation; bottle 4-phase vocab:
##                      approach/grip/rotate/withdraw)
═══════════════════════════════════════════════════════════════════════
### pretrains/  — Stage-2 backbones (AVP bottle human data)
  - S2_noLSS       = Run B Stage-2: AVP kinematics only, no reasoning, no LSS.
  - S2_pooledLSS   = Run E Stage-2: AVP + reasoning + POOLED LSS. ★ proposed (pooled)
  - S2_denseLSS    = R4 Stage-2 (june6): AVP + DENSE LSS. ★ proposed (dense).
                     ** This is the backbone behind the 90/54/26 headline results. **

### ablation/  — adjust_bottle component ablation (Runs B-G; all eval'd on adjust_bottle)
  Progressive method components, answering "what does each piece contribute?":
  - RunB_79pct              +AVP kinematics (over RunA 70%)
  - RunC_71pct              +reasoning as NAIVE language input (no aux loss)
  - RunD_74pct              +reasoning TOKEN-PREDICTION aux loss (CoT-style)
  - RunE_85pct_pooled       +POOLED LSS ★ proposed method (best ablation result)
  - ABL_simpletarget_58pct  LSS aligned to SIMPLE instruction (WRONG target) — shows the
                            reasoning target matters; degrades vs RunE.
  - RunF1_0pct_frozen       Frozen backbone, only action enc/dec trainable → 0% (fails)
  - RunF2_0pct_frozen       Frozen + img_pos_emb + adapters trainable → 0% (fails)
  Their Stage-2 pretrains (for reproducibility):
  - S2_RunC_reasoning-naive, S2_RunD_tokenpred, S2_LSS-simpletarget

### transfer/  — R2-R4 transfer ladders (held-out robot tasks; R1 baselines in _base/finetunes/)
  shake_bottle (near-transfer, bottle-like):
    - R2_noLSS_shake_bottle_45pct
    - R3_pooledLSS_shake_bottle_39pct   (pooled HURTS here: 39 < 45)
    - R4_denseLSS_shake_bottle_54pct    ★ dense wins
  move_can_pot (far-transfer, different object/motion, hard):
    - R2_noLSS_move_can_pot_20pct
    - R3_pooledLSS_move_can_pot_19pct   (pooled ≈ neutral)
    - R4_denseLSS_move_can_pot_26pct    ★ dense wins
  adjust_bottle (in-distribution):
    - R4_denseLSS_adjust_bottle_90pct   ★ dense in-dist (vs RunE pooled 85%)

  HEADLINE RESULT (dense LSS, all 100 rollouts, seed 42):
    Task              | R1   | R2  | R3(pooled) | R4(dense)
    adjust_bottle     | 70%  | 79% | 85%        | 90%   (in-dist)
    shake_bottle      | 34%  | 45% | 39%        | 54%   (transfer)
    move_can_pot      | 9.1% | 20% | 19%        | 26%   (transfer)
  Claim: dense improves transfer on BOTH held-out tasks; pooled fails to help (hurts on
  shake_bottle, neutral on move_can_pot). Frame as granularity/mechanism, not generality.

═══════════════════════════════════════════════════════════════════════
## tidying_tableware/  — HMD-2 (single-arm pick-place dishes onto rack;
##   8-phase vocab: approach/grip/lift/transport/insert/place/release/withdraw; 2 cycles/episode)
═══════════════════════════════════════════════════════════════════════
### pretrains/
  - S2_noLSS    = R2 tableware Stage-2 (AVP no-LSS), 10000 steps. [CURRENT]
  - [S2_pooledLSS, S2_denseLSS — INCOMING]
  Data: 449 part1 + 52 test episodes. Annotations verified 0-truncated (pooled + dense).
### transfer/  [INCOMING — matched tasks: stack_bowls_two, move_can_pot]

═══════════════════════════════════════════════════════════════════════
## extracting_groceries/  — HMD-3 (BIMANUAL asymmetric: one hand stabilizes bag,
##   other extracts object) [INCOMING — needs dual-track dense LSS build]
═══════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════
## _archive/  — dead/abandoned (NOT in paper)
═══════════════════════════════════════════════════════════════════════
  - zz_dataeff_baseline_25ep_77pct / zz_dataeff_LSS_25ep_48pct
      25-episode data-efficiency runs. ABANDONED: unfair comparison (2x steps/episode).
      All valid finetunes use 50 episodes / 22000 steps.
  - zz_dup_RunD_rerun          duplicate rerun of Run D (mar20_2)
  - zz_old_human_baseline_unused   early human pretrain, not used in any final run
  - zz_abandoned_putbottles_R1 / _R2   put_bottles_dustbin task — eval had bugs, task dropped.

═══════════════════════════════════════════════════════════════════════
## NOTES
═══════════════════════════════════════════════════════════════════════
- Model weights (*.bin, checkpoint-*/) are GITIGNORED. This README is the tracked manifest.
- Scripts reference: _base/egodex_foundation (pretrain base),
  tidying_tableware/pretrains/S2_noLSS (current finetune source),
  T_R2_tableware_stack_bowls_two (in root — eval in progress, move to
  tidying_tableware/transfer/ when done).
- All evals: RoboTwin 2.0, demo_randomized, seed 42, 100 rollouts, aloha-agilex 14-DOF.
