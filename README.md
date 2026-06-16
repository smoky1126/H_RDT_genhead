# Latent Semantic Scaffolding: Training-Time Reasoning Alignment for Vision-Language-Action Models

This repository extends [H-RDT](https://github.com/HongzheBi/H_RDT) with a training-time alignment mechanism — **Latent Semantic Scaffolding (LSS)** — that aligns image-grounded action token representations with T5 reasoning text embeddings during Stage 2 pretraining.

The LSS auxiliary head is discarded at inference, so the deployed policy has zero inference overhead relative to vanilla H-RDT.

![LSS Architecture](assets/lss_architecture.png)

---

## Headline Results

**Dense LSS beats every other method on every task** — in-distribution and both held-out transfer tasks. RoboTwin 2.0, `demo_randomized`, aloha-agilex, 100 rollouts, seed 42.

| Task | R1 baseline | R2 (+AVP) | R3 (pooled LSS) | R4 (dense LSS) |
|------|-------------|-----------|-----------------|----------------|
| `adjust_bottle` (in-distribution) | 70% | 79% | 85% | **90%** |
| `shake_bottle` (transfer) | 34% | 45% | 39% | **54%** |
| `move_can_pot` (transfer, hard) | 9.1% | 20% | 19% | **26%** |

**The mechanism claim — granularity is decisive:** dense (per-phase) LSS transfers to held-out tasks; pooled (per-episode) LSS over-specializes. Pooled *hurts* `shake_bottle` (39 < 45) and is neutral on `move_can_pot` (19 ≈ 20), while dense improves both. The benefit is in *how finely* reasoning is aligned, not merely that an auxiliary loss is present.

### Run ladder (the R-ladder, applied per human dataset)

- **R1** = H-RDT baseline: EgoDex pretrain → robot finetune. No AVP, no LSS.
- **R2** = + AVP human-data pretrain (Stage 2), no LSS (kinematics only).
- **R3** = + AVP + **Pooled LSS**: 16 action tokens pooled to one vector, aligned to one episode reasoning embedding.
- **R4** = + AVP + **Dense LSS** (proposed): each action token aligned to *its phase's* reasoning embedding.

### Component ablation (adjust_bottle, isolates what each piece contributes)

| Run | AVP | Reasoning | Mechanism | Success |
|-----|-----|-----------|-----------|---------|
| A (=R1) | – | – | EgoDex pretrain + robot finetune (baseline) | 70% |
| B (=R2) | +AVP | – | AVP kinematics, no reasoning | 79% |
| C | +AVP | +reasoning | Reasoning as language input (no aux loss) | 71% |
| D | +AVP | +reasoning | Reasoning via token-prediction loss (CoT-style) | 74% |
| **E (=R3)** | **+AVP** | **+reasoning** | **Pooled LSS (embedding alignment)** | **85%** |
| Ablation | +AVP | +reasoning | LSS aligned to simple instruction (wrong target) | 58% |
| F | +AVP | +reasoning | Run E Stage 2 + frozen backbone | 0% |
| **G (=R4)** | **+AVP** | **+reasoning** | **Dense LSS (proposed)** | **90%** |

The 58% → 85% gap (wrong-target ablation → Run E) is load-bearing evidence that *what* LSS aligns to matters, not just that an auxiliary loss is present.

---

## Generality: Three Human Demonstration Datasets

The method is validated across three AVP human-demonstration datasets (HMD) of increasing complexity. Each is collected on the Apple Vision Pro rig, processed to 48-D, and annotated with pooled + phased reasoning (see `human-policy_VLA`).

| Dataset | Task | Type | Phase vocab | Episodes | Status |
|---------|------|------|-------------|----------|--------|
| **HMD-1** | `adjust_bottle` | single-arm | approach/grip/rotate/withdraw (4) | 500 | ✅ full R1–R4 ladder (headline above) |
| **HMD-2** | `tidy_tableware` | single-arm | approach/grip/lift/transport/insert/place/release/withdraw (8) | 449+52 | ⏳ R2 done; R3/R4 in progress |
| **HMD-3** | `unpack_groceries` | **bimanual asymmetric** | per-hand (discovered) | 450+50 | 🔨 dual-track dense built; data pipeline ready |

**HMD-2 (tableware)** matched-task results so far (robot task `stack_bowls_two`):

| Run | Success | Note |
|-----|---------|------|
| R1 baseline | 23% | EgoDex only |
| R2 (+AVP tableware) | 32% | +9 from human-data pretrain |
| R3 (pooled LSS) | *running* | |
| R4 (dense LSS) | *pending* | |

**HMD-3 (groceries)** is asymmetric bimanual (one hand stabilizes the bag, the other extracts) and required extending dense LSS to two phase tracks — see **Bimanual Dense LSS** below.

---

## What LSS Is

LSS adds an `LSAHead` (defined in `models/hrdt_runner.py`) that projects image-grounded action token hidden states into the T5 reasoning embedding space. A cosine-distance loss aligns the projected representations with precomputed T5 embeddings of physical reasoning text annotations:

- Total loss: `L_total = L_diffusion + lambda_lsa * L_lsa`
- LSS loss: `L_lsa = mean(1 - cos(LSAHead(h_action), T5(reasoning_text)))`

Key properties:

- **Training-only.** `LSAHead` is never saved with the model and never instantiated at inference. The eval code in the companion `Reasoning_VLA_robotwin` repo confirms this.
- **No inference overhead.** Contrasts with inference-time conditioning approaches (CoT-VLA, pi0.7).
- **Stage 2 mechanism.** LSS activates during the second pretraining stage (Apple Vision Pro human demonstrations + reasoning text), not during finetuning.

The alignment target is generated offline by a VLM (see companion repo `human-policy_VLA`) producing structured reasoning text per trajectory.

---

## Dense LSS (phase-local alignment)

**Pooled LSS** (Run E) aligns *every* sampled action-window of an episode to a *single* episode-level reasoning embedding. **Dense LSS** aligns each action-window to the embedding of the reasoning rationale for the **specific manipulation phase** that window falls in.

Motivation: test whether the *temporal structure* of reasoning matters, not just its content. A window during "grip" is aligned to grip-reasoning; a window during "rotate" to rotate-reasoning.

### Mechanism

- Each AVP episode is segmented (offline, in `human-policy_VLA`) into causal phases, each with a one-sentence rationale (`reasoning_phased.json`).
- `generate_embed.py --dense` T5-encodes each phase rationale separately and writes a per-episode `*_dense.pt` with per-phase embeddings + phase frame-ranges.
- At training time, each of the K=16 action tokens (each spanning `upsample_rate`=3 frames) is assigned to the phase its frames fall in (`_build_token_phase_targets`) and aligned to that phase's pooled T5 embedding. Tokens straddling two phases are masked out (~5%).
- The loss is unchanged — Dense reuses the same cosine alignment; only the *target* differs (phase-local vs pooled). The pooled path is untouched.

Inference is unaffected: the alignment target exists only at training time and is discarded.

`*_dense.pt` schema (single-arm): `phase_names`, `phase_frames`, `phase_embeddings`, `phase_attn_masks`, `phase_rationales`, `pooled_embedding`, `phase_pooled` (n_phases, 4096), `episode_len`, `flags`.

### Building the dense targets

```bash
python datasets/pretrain/generate_embed.py \
  --data_root      ~/human-policy/data/recordings/processed_baseline_<task> \
  --baseline_root  ~/human-policy/data/recordings/processed_baseline_<task> \
  --dense
```

Reads `reasoning_phased.json`, reads `actions_48d` for frac→frame conversion, writes `*_dense.pt` co-located in `--baseline_root`. If a dense file is missing for an episode, that episode falls back to no dense target.

### Running a Dense LSS pretrain

```bash
# in pretrain.sh: --use_dense_lsa (instead of --use_lsa)
bash pretrain.sh
```

`--use_dense_lsa` and `--use_lsa` are mutually exclusive. Dense requires `*_dense.pt` files present.

---

## Bimanual Dense LSS (dual-track) — HMD-3

Asymmetric bimanual tasks (HMD-3 groceries: one hand stabilizes, the other extracts) have **two hands doing different things concurrently** — a single phase timeline cannot represent them. Dense LSS is extended to **two phase tracks** (one per hand).

**Design (see `DUAL_TRACK_DENSE_DESIGN.md`):**
- Annotation (`--mode bimanual`) produces per-hand phases: `{left_hand:{phases}, right_hand:{phases}}`.
- `generate_embed --dense` auto-detects bimanual annotation and writes a dual-track `*_dense.pt` (`phase_pooled_left/right`, `phase_frames_left/right`).
- Each action token is assigned to its phase *in each track independently* (`_assign_track` called twice).
- The loss sums two masked cosine terms — one per track — using **one shared LSAHead projection** (no architecture change): `L_lsa = lsa_track1 + lsa_track2`.

**Single-arm safety property (the design invariant):** single-arm = dual-track with the second track all-masked. The second cosine term evaluates to exactly 0, so single-arm loss equals the original single-track loss *byte-for-byte*. Existing single-arm `*_dense.pt` files use the old schema and flow through an unchanged code path — they are **not** regenerated.

**Validation status:**
- ✅ Verified (unit): token-assignment byte-identical on single-arm data; collate; **loss `allclose` to original on single-arm batches** (the second track contributes exactly 0).
- ⬜ Pending: full bimanual data test (needs locked groceries phase vocab); bottle regression through dual-track code (must reproduce 90/54/26; needs GPU).

See `DUAL_TRACK_DENSE_DESIGN.md` for the full design, per-file changes, verification gates, and the end-to-end bimanual usage (discover → lock → annotate → embed → train).

---

## Phase Discovery (data-driven vocabularies)

Phase vocabularies are *discovered* from data rather than hand-specified, via `human-policy_VLA/cet/discover_phases.py`:

```bash
# single-arm
python3 discover_phases.py --task_dir <processed_baseline_<task>/part1> --tag <task> --discover --per_block 4
# bimanual (per-hand segmentation)
python3 discover_phases.py --task_dir <...> --tag <task> --discover --per_block 4 --mode bimanual
python3 discover_phases.py --tag <task> --propose   # cluster synonyms for human review
python3 discover_phases.py --tag <task> --lock      # write phase_vocab_<task>.json
```

Free-form Gemini segmentation across stratified block samples → propose/lock workflow with human-in-the-loop merge confirmation. Bimanual mode segments each hand separately and pools both hands' phase names into the vocabulary.

---

## Repository Layout

Files that matter for the LSS contribution:

- `models/hrdt_runner.py` — LSAHead class + compute_loss with LSS branch (pooled + dense + dual-track)
- `main.py` — CLI flags: `--use_lsa`, `--lsa_lambda`, `--use_dense_lsa`
- `train/train.py` — LSAHead instantiation; passes dense targets (incl. dual-track R-track when present)
- `datasets/pretrain/egodex_dataset.py` — AVP/EgoDex loader; `_assign_track` + `_build_token_phase_targets` (single-arm + bimanual)
- `datasets/dataset.py` — collate: stacks per-token phase targets (left + right tracks)
- `datasets/pretrain/generate_embed.py` — T5 embedding builder; `--dense` mode; auto-detects single-arm vs bimanual annotation
- `pretrain.sh` / `finetune.sh` — Stage 2 pretrain / Stage 3 finetune
- `DUAL_TRACK_DENSE_DESIGN.md` — bimanual dual-track design + verification gates + usage
- `checkpoints/README.md` — checkpoint manifest (all runs, configs, results)

---

## Transfer Probe

The transfer probe tests whether LSS-shaped representations generalize to a **task not seen in Stage 2**. Stage-3 finetunes are run on a held-out task, identical except for the Stage-2 backbone they start from (R1–R4 ladder above). All: 50 episodes, 22,000 steps, `mode=finetune` (no LSS in Stage 3).

Two held-out tasks, both complete for HMD-1 (bottle):

- **`shake_bottle`** (near-transfer, bottle-like): 34 / 45 / 39 / 54 (R1/R2/R3/R4)
- **`move_can_pot`** (far-transfer, different object + place sub-goal): 9.1 / 20 / 19 / 26

Both show the same pattern: dense (R4) transfers positively on both; pooled (R3) does not (hurts shake_bottle, neutral on move_can_pot).

> Eval note: RoboTwin success rate is `policy_successes / valid_rollouts`; seeds where the simulator's own expert-demo setup crashes are skipped (not counted as failures).
> R1 anchor: the H-RDT paper reports higher absolute numbers with full training; R1 here is a 50-episode / 22k-step finetune of the EgoDex-only backbone, so a lower floor is expected. The probe reads the R-ladder against this consistent floor (all baselines reproduced on the same hardware under identical protocol).

---

## Mechanism Probing

A supporting representational analysis (`analysis/mechanism_probing/`) tests *why* dense transfers: whether dense LSS reshapes the backbone's action-token representation **by phase** more than pooled. Using silhouette score on the LSS-induced representational change, dense yields ~2x higher phase separability than pooled (0.047 vs 0.021 vs the AVP-only baseline; 0.037 vs 0.016 vs the H-RDT baseline). Absolute values are small (≤ 0.05), bounded by the modest distinctness of per-phase reasoning targets (mean cosine 0.70). This is **supporting** evidence; the primary evidence is the behavioral transfer above.

---

## Companion Repositories

| Repo | Purpose | Edit location |
|------|---------|---------------|
| Reasoning_VLA (this repo) | Training: Stage 2 LSS pretrain (pooled, dense, dual-track) + finetune | `~/H_RDT/` |
| Reasoning_VLA_robotwin | RoboTwin inference glue: `deploy_policy.py`, eval scripts | `~/RoboTwin/policy/H_RDT/inference/robotwin2_example/H_RDT/` |
| human-policy_VLA | AVP data collection + reasoning annotation (pooled, phased, bimanual) | `~/human-policy/` |

- https://github.com/smoky1126/Reasoning_VLA
- https://github.com/smoky1126/Reasoning_VLA_robotwin
- https://github.com/smoky1126/human-policy_VLA

---

## Setup

**Environment setup follows upstream H-RDT.** Create the conda env and download pretrained weights per the original H-RDT README, then return here for training.

### Stage 2 — LSS pretrain
`bash pretrain.sh`. Key flags: `--use_lsa` (pooled), `--use_dense_lsa` (dense; mutually exclusive; auto-handles bimanual from data), `--lsa_lambda 0.1`, `--use_lora`, `--mode pretrain` (LSS only active in pretrain).

### Stage 3 — task finetune
`bash finetune.sh`. Loads the Stage-2 LSS backbone and finetunes on robot task data (no LSS in Stage 3; the prior is baked into the backbone).

### Eval
In the companion `Reasoning_VLA_robotwin` repo: `bash eval.sh`.

---

## Checkpoints

Model weights are gitignored (they live on the training server). The tracked manifest is **`checkpoints/README.md`** — it documents every run (base, per-dataset pretrains, finetunes, ablations, archive) with configs and results. All evals: RoboTwin 2.0, demo_randomized, seed 42, 100 rollouts, aloha-agilex 14-DOF.
