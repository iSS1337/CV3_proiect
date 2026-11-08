"""Dataset-uri PyTorch pentru proiectul ADAS AEB (CV3).

1) KittiPedestrianDataset  - KITTI Tracking (secvente video cu bounding box-uri, track id si distanta z).
                             Folosit pentru train/val/test ale detectorului de pietoni (YOLO).
2) DashcamEventDataset     - propriul video dashcam + adnotari de evenimente (NORMAL/WARNING/EMERGENCY/IGNORE)
                             din ALPHA 3. Folosit ca set de testare a sistemului ADAS (nu are bounding box-uri).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import DetectionTransform, letterbox
from .splits import load_splits

PEDESTRIAN_CLASSES = ("Pedestrian",)


def _resolve_kitti_root(root: Path) -> Path:
    for cand in (root, root / "training"):
        if (cand / "image_02").is_dir() and (cand / "label_02").is_dir():
            return cand
    raise FileNotFoundError(
        f"Nu am gasit image_02/ si label_02/ in '{root}' (sau in '{root}/training'). "
        "Vezi README: descarca KITTI Tracking (left color images + training labels)."
    )


def parse_kitti_tracking_labels(label_file: Path, classes: Sequence[str] = PEDESTRIAN_CLASSES,
                                max_occlusion: int = 3, min_height_px: float = 0.0) -> Dict[int, List[dict]]:
    """Returneaza {frame: [ {track_id, box_xyxy, occluded, truncated, z}, ... ]} pentru clasele cerute."""
    per_frame: Dict[int, List[dict]] = {}
    for line in label_file.read_text().splitlines():
        p = line.split()
        if len(p) < 17 or p[2] not in classes:
            continue
        occ = int(p[4])
        x1, y1, x2, y2 = map(float, p[6:10])
        if occ > max_occlusion or (y2 - y1) < min_height_px:
            continue
        per_frame.setdefault(int(p[0]), []).append({
            "track_id": int(p[1]), "box": (x1, y1, x2, y2),
            "truncated": float(p[3]), "occluded": occ, "z": float(p[15]),
        })
    return per_frame


class KittiPedestrianDataset(Dataset):
    """Cadre KITTI Tracking cu pietoni.

    Un item = dict(image: Tensor[3,S,S], boxes: Tensor[N,4] (cx,cy,w,h normalizat), labels: Tensor[N] (0=pieton),
    track_ids: Tensor[N], distance_m: Tensor[N] (z KITTI, referinta pentru distanta/TTC),
    seq: str, frame: int, meta: dict).
    """

    def __init__(self, root: str | Path, split: str, split_dir: str | Path = "data/splits",
                 img_size: int = 640, augment: Optional[bool] = None,
                 only_with_pedestrians: bool = False, max_occlusion: int = 3,
                 min_height_px: float = 0.0, seed: int = 0):
        assert split in ("train", "val", "test")
        self.root = _resolve_kitti_root(Path(root))
        self.split = split
        self.sequences = load_splits(split_dir)[split]
        augment = (split == "train") if augment is None else augment
        self.transform = DetectionTransform(img_size, train=augment, seed=seed)

        self.annotations: Dict[str, Dict[int, List[dict]]] = {}
        self.index: List[Tuple[str, int]] = []
        for seq in self.sequences:
            ann = parse_kitti_tracking_labels(self.root / "label_02" / f"{seq}.txt",
                                              max_occlusion=max_occlusion, min_height_px=min_height_px)
            self.annotations[seq] = ann
            frames = sorted(int(p.stem) for p in (self.root / "image_02" / seq).glob("*.png"))
            for f in frames:
                if only_with_pedestrians and f not in ann:
                    continue
                self.index.append((seq, f))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        seq, frame = self.index[i]
        img = cv2.imread(str(self.root / "image_02" / seq / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Imagine ilizibila: {seq}/{frame:06d}.png")
        objs = self.annotations[seq].get(frame, [])
        boxes = np.array([o["box"] for o in objs], dtype=np.float32).reshape(-1, 4)
        extra = np.array([[o["track_id"], o["z"]] for o in objs], dtype=np.float32).reshape(-1, 2)
        tensor, yolo_boxes, extra, meta = self.transform(img, boxes, extra)
        n = len(yolo_boxes)
        return {
            "image": tensor,
            "boxes": torch.from_numpy(yolo_boxes).reshape(n, 4),
            "labels": torch.zeros(n, dtype=torch.long),
            "track_ids": torch.from_numpy(extra[:, 0]).long() if n else torch.zeros(0, dtype=torch.long),
            "distance_m": torch.from_numpy(extra[:, 1]) if n else torch.zeros(0),
            "seq": seq, "frame": frame, "meta": meta,
        }

    def stats(self) -> dict:
        n_ped = sum(len(v) for s in self.sequences for v in self.annotations[s].values())
        n_pos = sum(1 for s, f in self.index if f in self.annotations[s])
        return {"split": self.split, "sequences": self.sequences, "frames": len(self.index),
                "frames_with_pedestrians": n_pos, "pedestrian_boxes": n_ped}


# ----------------------------------------------------------------------------------------------
LABEL_NAMES = ("NORMAL", "WARNING", "EMERGENCY")
LABEL_IGNORE = -1


def events_to_frame_labels(events_csv: str | Path, n_frames: int) -> np.ndarray:
    """CSV de evenimente (event_id,label,start_frame,end_frame,...) -> vector de eticheta pe cadru.

    Prioritati: IGNORE (-1) > EMERGENCY (2) > WARNING (1) > NORMAL (0). Intervalele sunt inclusive.
    Evenimentele cu end_frame gol sunt respinse (adnotare neterminata).
    """
    labels = np.zeros(n_frames, dtype=np.int8)
    ignore = np.zeros(n_frames, dtype=bool)
    with open(events_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row["label"].strip().upper()
            if not row["end_frame"].strip():
                raise ValueError(f"Eveniment #{row['event_id']} neinchis (end_frame gol).")
            s, e = int(row["start_frame"]), min(int(row["end_frame"]), n_frames - 1)
            if lab == "IGNORE":
                ignore[s:e + 1] = True
            elif lab == "WARNING":
                labels[s:e + 1] = np.maximum(labels[s:e + 1], 1)
            elif lab == "EMERGENCY":
                labels[s:e + 1] = np.maximum(labels[s:e + 1], 2)
            else:
                raise ValueError(f"Eticheta necunoscuta: {lab}")
    labels[ignore] = LABEL_IGNORE
    return labels


class DashcamEventDataset(Dataset):
    """Cadre din video-ul propriu cu eticheta de situatie (NORMAL/WARNING/EMERGENCY) pe cadru.

    Cadrele din regiunile IGNORE (reluari slow-motion etc.) sunt excluse. `stride` subesantioneaza
    (59.94 FPS -> redundanta mare). Folosit DOAR pentru testare (un singur video, fara bounding box-uri).
    """

    def __init__(self, video: str | Path, events_csv: str | Path, img_size: int = 640, stride: int = 6):
        self.video, self.img_size = str(video), img_size
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Nu pot deschide video-ul: {video}")
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self.frame_labels = events_to_frame_labels(events_csv, self.n_frames)
        self.indices = [i for i in range(0, self.n_frames, stride) if self.frame_labels[i] != LABEL_IGNORE]
        self._cap: Optional[cv2.VideoCapture] = None  # deschis lazy, per worker

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video)
        idx = self.indices[i]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            raise IOError(f"Nu pot citi cadrul {idx}")
        img, _, _ = letterbox(frame, self.img_size)
        tensor = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))).float() / 255.0
        return {"image": tensor, "label": int(self.frame_labels[idx]), "frame": idx, "time_s": idx / self.fps}

    def stats(self) -> dict:
        fl = self.frame_labels
        return {"frames_total": self.n_frames, "fps": self.fps,
                **{n: int((fl == k).sum()) for k, n in enumerate(LABEL_NAMES)},
                "IGNORE": int((fl == LABEL_IGNORE).sum()), "items_after_stride": len(self)}
