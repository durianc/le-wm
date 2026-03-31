# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeWorldModel (LeWM) is a research codebase implementing a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) for learning world models from raw pixels. It trains with only two loss terms: next-embedding prediction (MSE) and a Gaussian regularizer (SIGReg), avoiding collapse without EMA, pretrained encoders, or auxiliary supervision.

## Installation

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Commands

**Training:**
```bash
python train.py data=pusht           # train on pusht dataset
python train.py data=dmc             # train on DMC dataset
python train.py data=tworoom         # train on tworoom dataset
```

**Evaluation:**
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=cube.yaml policy=cube/lewm
python eval.py --config-name=reacher.yaml policy=reacher/lewm
```

Note: The `policy` argument is the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix.

**Data setup:**
```bash
export STABLEWM_HOME=/path/to/your/storage   # defaults to ~/.stable-wm/
tar --zstd -xvf archive.tar.zst              # decompress downloaded datasets
```

## Architecture

The codebase is minimal by design — the core contribution lives in 4 files:

- **`jepa.py`** — `JEPA` nn.Module: the world model. Has `encode()` (ViT encoder + projector MLP → CLS token embedding), `predict()` (autoregressive predictor), `rollout()` (inference-time multi-step rollout), and `get_cost()` (for MPC planning via embedding-space MSE to goal).
- **`module.py`** — Neural network building blocks:
  - `ARPredictor`: causal transformer predictor (uses `ConditionalBlock` with AdaLN-zero conditioning on action embeddings)
  - `SIGReg`: Sketch Isotropic Gaussian Regularizer — enforces Gaussian-distributed latents via characteristic function matching
  - `Embedder`: action encoder (Conv1d → MLP)
  - `MLP`, `Transformer`, `Block`, `ConditionalBlock`, `Attention`: standard components
- **`train.py`** — Training entry point. Defines `lejepa_forward()` which computes the two LeWM losses: `pred_loss = MSE(predicted_emb, target_emb)` and `sigreg_loss`. Uses `stable_pretraining.Module` and `stable_pretraining.Manager` as training harness, with PyTorch Lightning + WandB logging.
- **`utils.py`** — `ModelObjectCallBack` (saves full model object via `torch.save` per epoch), plus preprocessing helpers.
- **`eval.py`** — Evaluation entry point. Loads checkpoints via `swm.policy.AutoCostModel` (for JEPA/world-model policies) or `swm.policy.AutoActionableModel` (for direct policies like GCBC/IQL). Runs `world.evaluate_from_dataset()`.

## External Dependencies

- **`stable-worldmodel`** (`swm`): handles environments (`swm.World`), HDF5 datasets, planning solvers (CEM, Adam), and policy wrappers. Configs in `config/eval/` use its API.
- **`stable-pretraining`** (`spt`): provides training infrastructure (`spt.Module`, `spt.Manager`), ViT backbones (`spt.backbone.utils.vit_hf`), and data utilities.

## Config System

Hydra-based configs under `config/`:
- `config/train/lewm.yaml` — main training config (references `data/` subconfigs)
- `config/train/data/*.yaml` — dataset-specific configs (dataset name, frameskip, action dim)
- `config/eval/*.yaml` — per-environment eval configs (world, solver, planning horizon)
- `config/eval/solver/*.yaml` — CEM or Adam solver settings

## Key Details

- **Checkpoint format**: Two files per checkpoint — `<name>_object.ckpt` (serialized Python object, used by `eval.py`) and `<name>_weights.ckpt` (state dict only).
- **Dataset path**: Dataset `.h5` files live under `$STABLEWM_HOME` (default: `~/.stable-wm/`). Dataset names in configs omit the `.h5` extension.
- **WandB**: Set `entity` and `project` in `config/train/lewm.yaml` before training.
- **`MUJOCO_GL=egl`**: Set automatically in `eval.py` for headless rendering.
- **Training objective**: `loss = pred_loss + λ * sigreg_loss` where `λ = 0.09` by default (`loss.sigreg.weight` in config). SIGReg operates on `emb` transposed to `(T, B, D)`.
