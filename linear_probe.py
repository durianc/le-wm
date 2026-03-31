"""
Linear probe for through_door binary classification on frozen JEPA embeddings.

Training entry point: extracts CLS-token embeddings from the world model
encoder, fits a logistic linear layer on oracle `through_door` labels, and
saves the probe weights for use in eval_cf.py.

Usage
-----
python linear_probe.py  --model_ckpt <path>  --dataset tworoom
                        --probe_ckpt probe.pt  --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import stable_worldmodel as swm
from cf_oracle import oracle_rollout
from cf_env import register_cf_env

register_cf_env()


# ── Model ─────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Single linear layer for binary through_door classification."""

    def __init__(self, in_dim: int, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Args:
            emb: (N, D) or (N, T, D) — if 3-D, uses the last timestep.
        Returns:
            logits: (N, n_classes)
        """
        if emb.ndim == 3:
            emb = emb[:, -1, :]  # last step
        return self.fc(emb)

    def predict_pass(self, emb: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (0 = no cross, 1 = crossed).

        Args:
            emb: (N, D) or (N, T, D)
        Returns:
            preds: (N,) int64
        """
        with torch.no_grad():
            logits = self(emb)
            return logits.argmax(dim=-1)


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataset,
    img_size: int,
    device: torch.device,
    history_size: int = 1,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode dataset rows and collect oracle through_door labels.

    For each dataset row we:
      1. Render the initial observation from the env.
      2. Call model.encode() to get the CLS-token embedding.
      3. Run oracle_rollout to obtain ground-truth through_door label.

    Args:
        model:        JEPA (or compatible) model with an encode() method.
        dataset:      swm.data.HDF5Dataset instance.
        img_size:     Target image size for the encoder.
        device:       Torch device.
        history_size: Number of frames per context window (usually 1).
        batch_size:   How many rows to batch for encoding.

    Returns:
        embeddings: (N, D) float32 tensor.
        labels:     (N,) int64 tensor  (0 = not crossed, 1 = crossed).
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
    labels_list: list[int] = []

    for idx in range(len(dataset)):
        row = dataset[idx]

        actions = np.asarray(row.get("action", []), dtype=np.float32)
        if actions.ndim == 1:
            continue  # single-step row; skip

        state = np.asarray(row.get("state", row.get("proprio", [])), dtype=np.float32)
        if state.ndim > 1:
            state = state[0]
        agent_pos = state[:2]

        # Render initial frame
        env.reset(options={"agent_pos": agent_pos})
        frame = env.render()  # (H, W, 3) uint8

        # Preprocess -> (1, 1, C, H, W)  [B=1, T=history_size=1]
        img_t = transform(frame).unsqueeze(0).unsqueeze(0).to(device)

        # Encode
        info = {"pixels": img_t}
        model.encode(info)
        emb = info["emb"][:, -1, :]  # (1, D)
        embs_list.append(emb.cpu())

        # Oracle label
        oracle = oracle_rollout({"agent_pos": agent_pos}, actions)
        labels_list.append(int(oracle["through_door"]))

    env.close()

    embeddings = torch.cat(embs_list, dim=0)          # (N, D)
    labels = torch.tensor(labels_list, dtype=torch.long)  # (N,)
    return embeddings, labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_probe(
    probe: LinearProbe,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> float:
    """Fit the linear probe via cross-entropy.

    Returns final training accuracy.
    """
    device = device or torch.device("cpu")
    probe = probe.to(device)
    embeddings = embeddings.to(device)
    labels = labels.to(device)

    loader = DataLoader(
        TensorDataset(embeddings, labels),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            opt.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        acc = correct / n
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  acc={acc:.4f}")

    return acc


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a linear probe on JEPA embeddings.")
    parser.add_argument("--model_ckpt", required=True, help="Path to world model checkpoint (run name or path).")
    parser.add_argument("--dataset", default="tworoom", help="HDF5Dataset name.")
    parser.add_argument("--probe_ckpt", default="probe.pt", help="Where to save probe weights.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--history_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading world model...")
    model = swm.policy.AutoActionableModel(args.model_ckpt)
    # Unwrap to JEPA if needed (AutoActionableModel returns the module with get_action)
    # We need encode(), so unwrap one level if necessary.
    if not hasattr(model, "encode"):
        for child in model.children():
            if hasattr(child, "encode"):
                model = child
                break
    model = model.to(device).eval()

    print("Loading dataset...")
    dataset = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["action", "state", "proprio"])

    print("Extracting embeddings (this may take a while)...")
    embeddings, labels = extract_embeddings(
        model, dataset, args.img_size, device, args.history_size
    )
    print(f"  {len(embeddings)} samples, class balance: {labels.float().mean():.3f}")

    in_dim = embeddings.shape[-1]
    probe = LinearProbe(in_dim=in_dim, n_classes=2)

    print("Training probe...")
    final_acc = train_probe(probe, embeddings, labels, epochs=args.epochs, lr=args.lr, device=device)
    print(f"Final training accuracy: {final_acc:.4f}")

    save_path = Path(args.probe_ckpt)
    torch.save({"state_dict": probe.state_dict(), "in_dim": in_dim}, save_path)
    print(f"Probe saved to {save_path}")


if __name__ == "__main__":
    main()
