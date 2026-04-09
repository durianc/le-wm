"""
MLP probe for agent_pos regression on frozen JEPA encoder embeddings.

Training entry point: encodes each episode's initial frame with the frozen
world model encoder, fits a 2-layer MLP regression head to predict the initial
agent position (x, y), and saves the probe weights for use in eval_cf.py.

At eval time (eval_cf.py) the probe predicts agent_pos from the initial
embedding, then oracle_rollout is called with that predicted position to
obtain a through_door classification.  This measures whether the encoder
has captured spatial structure — not whether the action sequence reaches
the door under fixed physics.

Usage
-----
python linear_probe.py  --model_ckpt <path>  --dataset tworoom
                        --probe_ckpt logs_eval/probes/probe.pt  --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import stable_worldmodel as swm
from .cf_env import register_cf_env
from .utils import add_model_suffix

register_cf_env()


# ── Model ─────────────────────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    """2-Layer MLP regression head: embedding → agent_pos (x, y)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Args:
            emb: (N, D) or (N, T, D) — if 3-D, uses the last timestep.
        Returns:
            pos_pred: (N, 2)  predicted (x, y) in pixel coordinates
        """
        if emb.ndim == 3:
            emb = emb[:, -1, :]
        return self.net(emb)

    def predict_pos(self, emb: torch.Tensor) -> torch.Tensor:
        """Return predicted agent position.

        Args:
            emb: (N, D)
        Returns:
            pos: (N, 2) float32
        """
        with torch.no_grad():
            return self(emb)


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    h5_path: str,
    img_size: int,
    device: torch.device,
    history_size: int = 1,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode each episode's initial frame and collect the initial agent_pos.

    For each episode we:
      1. Read the initial agent_pos from proprio[0].
      2. Render the initial frame in TwoRoomCF-v0.
      3. Call model.encode() to obtain the CLS-token embedding (D,).

    The probe is then trained to map embedding → agent_pos with MSE.

    Args:
        model:        JEPA model with encode().
        h5_path:      Path to the tworoom HDF5 file.
        img_size:     Target image size for the encoder.
        device:       Torch device.
        history_size: Unused; kept for API compatibility.
        batch_size:   Unused; kept for API compatibility.

    Returns:
        embeddings: (N, D) float32 tensor — initial-frame CLS embeddings.
        positions:  (N, 2) float32 tensor — ground-truth agent_pos (x, y).
    """
    import gymnasium
    from torchvision.transforms import v2 as T

    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            T.Resize(size=img_size),
        ]
    )

    env = gymnasium.make("lewm/TwoRoomCF-v0", render_mode="rgb_array")

    embs_list: list[torch.Tensor] = []
    pos_list:  list[torch.Tensor] = []

    with h5py.File(h5_path, "r") as f:
        ep_offsets  = f["ep_offset"][:]   # (num_episodes,)
        ep_lens     = f["ep_len"][:]      # (num_episodes,)
        all_proprio = f["proprio"][:]     # (total_steps, 2)

    for ep_off, ep_len in tqdm(
        zip(ep_offsets, ep_lens), total=len(ep_offsets), desc="Extracting"
    ):
        proprio   = all_proprio[ep_off : ep_off + ep_len].astype(np.float32)
        agent_pos = proprio[0, :2]  # (2,) initial position

        # Render initial frame
        env.reset(options={"agent_pos": agent_pos})
        frame = env.render()  # (H, W, 3) uint8

        # Preprocess → (1, 1, C, H, W)
        img_t = transform(frame).unsqueeze(0).unsqueeze(0).to(device)

        # Encode — only the initial frame, no rollout
        info = {"pixels": img_t}
        model.encode(info)
        emb = info["emb"]  # (1, 1, D)
        embs_list.append(emb[:, -1, :].cpu())          # (1, D)
        pos_list.append(torch.from_numpy(agent_pos))   # (2,)

    env.close()

    embeddings = torch.cat(embs_list, dim=0)   # (N, D)
    positions  = torch.stack(pos_list, dim=0)  # (N, 2)
    return embeddings, positions


# ── Training ──────────────────────────────────────────────────────────────────

def train_probe(
    probe: MLPProbe,
    embeddings: torch.Tensor,
    positions: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> float:
    """Fit the MLP probe via MSE regression on agent_pos.

    Returns final training MSE (pixels²).
    """
    device = device or torch.device("cpu")
    probe = probe.to(device)
    embeddings = embeddings.to(device)
    positions  = positions.to(device)

    # Normalise targets to zero-mean unit-variance for stable training.
    pos_mean = positions.mean(dim=0)
    pos_std  = positions.std(dim=0).clamp(min=1e-6)
    positions_norm = (positions - pos_mean) / pos_std

    loader = DataLoader(
        TensorDataset(embeddings, positions_norm),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        probe.train()
        total_loss, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            pred = probe(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        mse_norm = total_loss / n
        # Convert back to pixel² for interpretability
        mse_px = mse_norm * (pos_std ** 2).mean().item()
        print(
            f"  epoch {epoch+1}/{epochs}  "
            f"mse_norm={mse_norm:.4f}  mse_px²={mse_px:.2f}"
        )

    # Return pixel-space MSE
    probe.eval()
    with torch.no_grad():
        pred_norm = probe(embeddings)
        pred_px   = pred_norm * pos_std + pos_mean
        final_mse = ((pred_px - positions) ** 2).mean().item()
    return final_mse


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP probe to regress agent_pos from JEPA encoder embeddings."
    )
    parser.add_argument("--model_ckpt", required=True,
                        help="Path to world model checkpoint (run name or path).")
    parser.add_argument("--dataset", default="tworoom", help="Dataset name.")
    parser.add_argument("--probe_ckpt", default="/mnt/data/szeluresearch/stable-wm/tworoom/probe.pt",
                        help="Where to save probe weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--history_size", type=int, default=1)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading world model...")
    model = swm.policy.AutoCostModel(args.model_ckpt)
    model = model.to(device).eval()

    print("Extracting embeddings (this may take a while)...")
    from stable_worldmodel.data.utils import get_cache_dir
    h5_path = Path(get_cache_dir()) / f"{args.dataset}.h5"
    embeddings, positions = extract_embeddings(
        model, str(h5_path), args.img_size, device, args.history_size
    )
    print(f"  {len(embeddings)} episodes")
    print(f"  agent_pos range  x=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]"
          f"  y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")

    in_dim = embeddings.shape[-1]
    probe = MLPProbe(in_dim=in_dim, hidden_dim=256, out_dim=2)

    print("Training probe...")
    final_mse = train_probe(
        probe, embeddings, positions,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    print(f"Final training MSE: {final_mse:.2f} px²  (RMSE={final_mse**0.5:.2f} px)")

    # Save probe + normalisation stats so eval_cf can invert them
    pos_mean = positions.mean(dim=0)
    pos_std  = positions.std(dim=0).clamp(min=1e-6)
    save_path = Path(args.probe_ckpt)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "in_dim": in_dim,
            "out_dim": 2,
            "hidden_dim": 256,
            "pos_mean": pos_mean.cpu(),
            "pos_std":  pos_std.cpu(),
            "is_mlp": True, # 添加一个标志，方便在 eval 阶段识别这是 MLP
        },
        save_path,
    )
    print(f"Probe saved to {save_path}")


if __name__ == "__main__":
    main()