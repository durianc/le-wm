import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def load_lewm_model(policy_ref: str, cache_dir=None) -> torch.nn.Module | None:
    """Load a LeWM model from a lewm/ directory containing weights.pt.

    Infers architecture from weight shapes — no config.json required.
    Returns None if no weights.pt is found (non-lewm checkpoint).
    """
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from stable_pretraining.backbone.utils import vit_hf

    # Resolve path: absolute or relative to cache_dir
    p = Path(policy_ref)
    if p.suffix == ".pt" and p.exists():
        weights_path = p
    else:
        for candidate in (p, Path(cache_dir or swm.data.utils.get_cache_dir()) / policy_ref):
            if candidate.is_dir():
                w = candidate / "weights.pt"
                if w.exists():
                    weights_path = w
                    break
        else:
            return None  # not a lewm checkpoint

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Infer architecture from weight shapes
    act_input_dim    = ckpt["action_encoder.patch_embed.weight"].shape[1]
    hidden_dim       = ckpt["projector.net.0.weight"].shape[0]
    embed_dim        = ckpt["projector.net.0.weight"].shape[1]
    predictor_frames = ckpt["predictor.pos_embedding"].shape[1]
    predictor_depth  = sum(
        1 for k in ckpt
        if k.startswith("predictor.transformer.layers.")
        and k.endswith(".adaLN_modulation.1.weight")
    )
    proj_norm = torch.nn.BatchNorm1d if "projector.net.1.running_mean" in ckpt else torch.nn.LayerNorm

    encoder = vit_hf(size="tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    predictor = ARPredictor(
        num_frames=predictor_frames, input_dim=embed_dim, hidden_dim=embed_dim,
        output_dim=embed_dim, depth=predictor_depth, heads=16, mlp_dim=2048,
        dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=act_input_dim, emb_dim=embed_dim)
    projector  = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=hidden_dim, norm_fn=proj_norm)
    pred_proj  = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=hidden_dim, norm_fn=proj_norm)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading lewm weights: {missing}")
    if unexpected:
        print(f"  [warn] Ignored unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def patch_gcrl_compat(model):
    """Patch backward-compat attributes for older serialized GCRL modules."""
    for module in model.modules():
        # Older checkpoints may deserialize Transformer without this attribute,
        # while newer forward() expects it.
        if module.__class__.__name__ == "Transformer" and not hasattr(module, "pool_type"):
            module.pool_type = "attention"
    return model


class FastActionablePolicy(swm.policy.FeedForwardPolicy):
    """Faster feed-forward policy with batched image preprocessing."""

    def __init__(self, model, img_size, process=None, transform=None, **kwargs):
        super().__init__(model=model, process=process, transform=transform, **kwargs)
        self.img_size = int(img_size)
        stats = spt.data.dataset_stats.ImageNet
        self._mean = torch.tensor(stats["mean"]).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(stats["std"]).view(1, 1, 3, 1, 1)

    def _prepare_image_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # Expected shape: (E, T, H, W, C)
        if x.ndim != 5:
            return x

        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # -> (E, T, C, H, W)
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        h, w = x.shape[-2], x.shape[-1]
        if h != self.img_size or w != self.img_size:
            et = x.shape[:2]
            x = x.view(-1, *x.shape[-3:])
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.view(*et, *x.shape[-3:])

        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return x

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        # Shallow copy so we don't mutate env infos in place
        info_dict = dict(info_dict)

        # Lightweight numeric preprocessing (same semantics as base policy)
        for k, v in list(info_dict.items()):
            if not isinstance(v, (np.ndarray, np.generic)):
                continue
            if hasattr(self, "process") and k in self.process:
                shape = v.shape
                if len(shape) > 2:
                    v_flat = v.reshape(-1, *shape[2:])
                else:
                    v_flat = v
                v = self.process[k].transform(v_flat).reshape(shape)
            info_dict[k] = v

        # Keep compatibility with GCRL checkpoints expecting goal_pixels
        if "goal" in info_dict:
            info_dict["goal_pixels"] = info_dict["goal"]

        # Batched image preprocessing
        for key in ("pixels", "goal", "goal_pixels"):
            if key in info_dict:
                info_dict[key] = self._prepare_image_tensor(info_dict[key])

        # Convert remaining numeric arrays to tensors
        for k, v in list(info_dict.items()):
            if isinstance(v, (np.ndarray, np.generic)) and v.dtype.kind not in "USO":
                info_dict[k] = torch.from_numpy(v)

        device = next(self.model.parameters()).device
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device, non_blocking=True)

        with torch.inference_mode():
            action = self.model.get_action(info_dict)

        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    img_size = int(cfg.eval.img_size)
    world = swm.World(**cfg.world, image_shape=(img_size, img_size))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        # Try lewm weights.pt loading first
        model = load_lewm_model(cfg.policy)
        if model is not None:
            model = model.to("cuda")
            model = model.eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            config = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )
        else:
            try:
                model = swm.policy.AutoCostModel(cfg.policy)
                model = model.to("cuda")
                model = patch_gcrl_compat(model)
                model = model.eval()
                model.requires_grad_(False)
                model.interpolate_pos_encoding = True
                config = swm.PlanConfig(**cfg.plan_config)
                solver = hydra.utils.instantiate(cfg.solver, model=model)
                policy = swm.policy.WorldModelPolicy(
                    solver=solver, config=config, process=process, transform=transform
                )
            except RuntimeError as e:
                # Fallback for direct policy checkpoints (e.g., GCBC/IQL/IVL) that
                # expose get_action but not get_cost.
                if "get_cost" not in str(e):
                    raise
                model = swm.policy.AutoActionableModel(cfg.policy)
                model = model.to("cuda")
                model = patch_gcrl_compat(model)
                model = model.eval()
                model.requires_grad_(False)
                policy = FastActionablePolicy(
                    model=model, process=process, transform=transform
                    , img_size=cfg.eval.img_size
                )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
