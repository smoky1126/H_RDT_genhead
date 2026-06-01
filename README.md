# Reasoning_VLA: H-RDT with Latent Semantic Scaffolding (LSS)

This repository extends [H-RDT](https://github.com/embodiedfoundation/H-RDT) with a training-time alignment mechanism — **Latent Semantic Scaffolding (LSS)** — that aligns image-grounded action token representations with T5 reasoning text embeddings during Stage 2 pretraining.

The LSS auxiliary head is discarded at inference, so the deployed policy has zero inference overhead relative to vanilla H-RDT.


---

## Headline Results

Single-task evaluation: RoboTwin 2.0, `adjust_bottle`, `demo_randomized`, aloha-agilex robot. Success rates averaged over 100 rollouts, seed 42.

| Run | Configuration | Success |
|-----|---------------|---------|
| A   | EgoDex baseline (Stage 1 pretrain + finetune) | 70% |
| B   | + AVP kinematics (Stage 2 human pretrain) | 79% |
| C   | + naive reasoning text injection | 71% |
| D   | + reasoning token-prediction auxiliary loss | 74% |
| **E** | **+ AVP + reasoning + LSS (proposed)** | **85%** |
| F   | frozen backbone variant — domain gap too large | 0% |
| Ablation | LSS with trivial alignment target | 58% |

The 58% to 85% gap (ablation to Run E) is the load-bearing evidence that what LSS aligns to matters, not just that an auxiliary loss is present.

---

## What LSS Is

LSS adds an `LSAHead` (defined in `models/hrdt_runner.py`) that projects image-grounded action token hidden states into the T5 reasoning embedding space. A cosine-distance loss aligns the projected representations with precomputed T5 embeddings of physical reasoning text annotations:

- Total loss: `L_total = L_diffusion + lambda_lsa * L_lsa`
- LSS loss: `L_lsa = mean(1 - cos(LSAHead(h_action), T5(reasoning_text)))`

Key properties:

- **Training-only.** `LSAHead` is never saved with the model and never instantiated at inference. The eval code in the companion `Reasoning_VLA_robotwin` repo confirms this.
- **No inference overhead.** Contrasts with inference-time conditioning approaches (CoT-VLA, pi0.7).
- **Stage 2 mechanism.** LSS activates during the second pretraining stage (Apple Vision Pro human demonstrations + reasoning text), not during finetuning.

The alignment target is generated offline by a VLM (see companion repo `human-policy_VLA`) producing structured JSON annotations of the form `{high_level_goal, subgoal, reason, objects}` per trajectory.

---

## Repository Layout

Files that matter for the LSS contribution:

- `models/hrdt_runner.py` — LSAHead class + compute_loss with LSS branch
- `main.py` — CLI flags: `--use_lsa`, `--lsa_lambda`
- `train/train.py` — LSAHead instantiation (pretrain mode only)
- `models/hrdt/model.py` — forward() supports `return_hidden=True` for LSS hook
- `pretrain.sh` — Stage 2 pretrain script with `--use_lsa`
- `finetune.sh` — Stage 3 task finetune (LSS already baked into priors)
- `datasets/pretrain/` — EgoDex + AVP human data loading
- `configs/hrdt_pretrain.yaml`, `configs/hrdt_finetune.yaml`

Other directories (`assets/`, `inference/real_example/`, etc.) are inherited from upstream H-RDT.

---

## Companion Repositories

This work spans three repos:

| Repo | Purpose | Edit location on dev machine |
|------|---------|------------------------------|
| Reasoning_VLA (this repo) | Training code: Stage 2 LSS pretrain + finetune | `~/H_RDT/` |
| Reasoning_VLA_robotwin | RoboTwin inference glue: `deploy_policy.py`, eval scripts | `~/RoboTwin/policy/H_RDT/inference/robotwin2_example/H_RDT/` |
| human-policy_VLA | AVP data collection + reasoning text annotation pipeline | `~/human-policy/` |

Links:

- https://github.com/smoky1126/Reasoning_VLA
- https://github.com/smoky1126/Reasoning_VLA_robotwin
- https://github.com/smoky1126/human-policy_VLA

---

## Setup

**Environment setup follows upstream H-RDT.** Use the instructions in the original H-RDT README to create the conda env and download pretrained weights. Then come back to this repo for training.

### Running an LSS pretrain (Stage 2)

Run `bash pretrain.sh` from the repo root.

Key flags inside `pretrain.sh`:

- `--use_lsa` enables the LSS loss
- `--lsa_lambda 0.1` weight on the alignment loss (default 0.1)
- `--use_lora` LoRA for parameter-efficient pretraining
- `--mode pretrain` LSS is only active in pretrain mode

### Running a task finetune (Stage 3)

Run `bash finetune.sh` from the repo root.

Loads the Stage 2 LSS-pretrained backbone and finetunes on robot task data. LSS priors are preserved through the frozen-block parameter selection — see `train/train.py` (`--freeze_backbone` logic).

### Running eval

Eval lives in the companion `Reasoning_VLA_robotwin` repo. From that repo's directory, run `bash eval.sh`.

---

## Checkpoint Naming Convention

Checkpoints in `checkpoints/` follow this convention:

- `pretrain_<config>_<date>/` — Stage 2 pretrain output
- `model_<letter>_<config>_<date>/` — Stage 3 finetune from the corresponding pretrain
- `model_e_*` is the proposed method (LSS + reasoning + AVP)
- `model_e_ablation_*` is the trivial-target ablation

Checkpoints are not pushed to GitHub (they live on the training server). Contact me for access.

---

## Citation

If you use this code, please cite both the upstream H-RDT paper and this repository:

```
@misc{reasoning_vla_2026,
  title  = {Latent Semantic Scaffolding for Vision-Language-Action Models},
  author = {Andrew},
  note   = {Work in progress, MPhil project, CUHK},
  year   = {2026}
}
```

For H-RDT, see https://arxiv.org/abs/2507.23523.

---

## Acknowledgments

Built on top of [H-RDT](https://github.com/embodiedfoundation/H-RDT) by the Embodied Foundation team. The H-RDT codebase, pretrained weights, and EgoDex data pipeline are the foundation this work extends. All credit for the underlying VLA architecture goes to the H-RDT authors.

AVP data collection pipeline forked from [RogerQi/human-policy](https://github.com/RogerQi/human-policy).
