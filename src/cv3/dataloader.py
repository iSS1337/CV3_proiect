"""DataLoader-e pentru train/val/test (+ colate pentru numar variabil de cutii per imagine)."""
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import KittiPedestrianDataset


def detection_collate(batch: list) -> dict:
    """Stivuieste imaginile, cutiile raman liste (N variabil). Produce si formatul YOLO (img_idx, cls, cx, cy, w, h)."""
    images = torch.stack([b["image"] for b in batch])
    rows = []
    for i, b in enumerate(batch):
        n = len(b["boxes"])
        if n:
            rows.append(torch.cat([torch.full((n, 1), float(i)), b["labels"].float().unsqueeze(1), b["boxes"]], dim=1))
    yolo_targets = torch.cat(rows) if rows else torch.zeros((0, 6))
    return {
        "images": images,
        "targets": yolo_targets,                       # (M, 6): img_idx, cls, cx, cy, w, h
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "distance_m": [b["distance_m"] for b in batch],
        "track_ids": [b["track_ids"] for b in batch],
        "seq": [b["seq"] for b in batch],
        "frame": [b["frame"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if hasattr(ds, "transform"):
        ds.transform.rng = np.random.default_rng(s)


def get_dataloaders(root: str, split_dir: str = "data/splits", img_size: int = 640, batch_size: int = 16,
                    num_workers: int = 4, seed: int = 42, **dataset_kwargs) -> Dict[str, DataLoader]:
    """Returneaza {'train','val','test'} -> DataLoader. Doar train e amestecat si augmentat."""
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = KittiPedestrianDataset(root, split, split_dir, img_size, seed=seed, **dataset_kwargs)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers,
            collate_fn=detection_collate, worker_init_fn=_seed_worker, generator=g,
            pin_memory=torch.cuda.is_available(), drop_last=(split == "train"),
            persistent_workers=num_workers > 0,
        )
    return loaders
