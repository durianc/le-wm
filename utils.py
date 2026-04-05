import re
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


def get_model_name(model_ref: str | Path) -> str:
    """Derive a short artifact-friendly model name from a policy/checkpoint ref."""
    if str(model_ref) == "random":
        return "random"

    name = Path(str(model_ref).rstrip("/")).name
    if not name:
        name = Path(str(model_ref)).stem

    for suffix in (".ckpt", ".pt", "_object", "_weights"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "model"


def add_model_suffix(path: str | Path, model_ref: str | Path) -> Path:
    """Insert the model name before the final suffix, preserving parent directories."""
    resolved_path = Path(path)
    model_name = get_model_name(model_ref)
    if not model_name:
        return resolved_path

    stem = resolved_path.stem
    if stem == model_name or stem.endswith(f"_{model_name}"):
        return resolved_path

    return resolved_path.with_name(f"{stem}_{model_name}{resolved_path.suffix}")


def resolve_model_artifact_path(path: str | Path, model_ref: str | Path) -> Path:
    """Resolve a model-scoped artifact path, preferring an existing file when present."""
    resolved_path = Path(path)
    if resolved_path.exists():
        return resolved_path

    suffixed_path = add_model_suffix(resolved_path, model_ref)
    if suffixed_path.exists():
        return suffixed_path

    return suffixed_path

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")