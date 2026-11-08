"""Pre-procesare si augmentari pentru detectia pietonilor (compatibile cu YOLO).

Conventie: imagine BGR uint8 (H, W, 3) -> tensor float32 (3, S, S) RGB in [0, 1];
cutii in pixeli xyxy -> returnate normalizate YOLO (cx, cy, w, h) relativ la imaginea dupa letterbox.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Redimensioneaza pastrand raportul de aspect si completeaza cu gri. Returneaza (img, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, r, (pad_x, pad_y)


def xyxy_to_yolo(boxes: np.ndarray, size: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = boxes.astype(np.float32)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.stack([cx, cy, w, h], axis=1) / size


def _hflip(img, boxes):
    img = img[:, ::-1].copy()
    if len(boxes):
        w = img.shape[1]
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img, boxes


def _color_jitter(img, rng, b=0.25, c=0.25, s=0.3):
    out = img.astype(np.float32)
    out = out * (1 + rng.uniform(-c, c)) + 255 * rng.uniform(-b, b) * 0.5
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1 + rng.uniform(-s, s)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _random_scale_crop(img, boxes, rng, lo=0.75, hi=1.0):
    """Taie o sub-regiune aleatoare (zoom-in); pastreaza doar cutiile cu >= 50% din arie ramasa."""
    h, w = img.shape[:2]
    s = rng.uniform(lo, hi)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    if len(boxes) == 0:
        return crop, boxes, np.zeros(0, dtype=bool)
    b = boxes.copy()
    b[:, [0, 2]] -= x0
    b[:, [1, 3]] -= y0
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, cw)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, ch)
    area1 = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    keep = (area1 >= 0.5 * area0) & (area1 > 4)
    return crop, b[keep], keep


class DetectionTransform:
    """Pipeline: [augmentari doar la train] -> letterbox -> normalizare -> tensor."""

    def __init__(self, size: int = 640, train: bool = False, seed: int | None = None):
        self.size, self.train = size, train
        self.rng = np.random.default_rng(seed)

    def __call__(self, img_bgr: np.ndarray, boxes_xyxy: np.ndarray, extra: np.ndarray | None = None):
        """extra: array (N, k) aliniat cu cutiile (ex. distanta z din KITTI), filtrat odata cu ele."""
        boxes = boxes_xyxy.astype(np.float32).reshape(-1, 4)
        extra = np.zeros((len(boxes), 0), np.float32) if extra is None else extra
        if self.train:
            if self.rng.random() < 0.5:
                img_bgr, boxes = _hflip(img_bgr, boxes)
            if self.rng.random() < 0.5:
                img_bgr, boxes, keep = _random_scale_crop(img_bgr, boxes, self.rng)
                extra = extra[keep] if keep.size else extra
            img_bgr = _color_jitter(img_bgr, self.rng)

        h0, w0 = img_bgr.shape[:2]
        img, r, (px, py) = letterbox(img_bgr, self.size)
        if len(boxes):
            boxes = boxes * r
            boxes[:, [0, 2]] += px
            boxes[:, [1, 3]] += py
            boxes = boxes.clip(0, self.size)
        tensor = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))).float() / 255.0
        meta = {"orig_size": (h0, w0), "scale": r, "pad": (px, py)}
        return tensor, xyxy_to_yolo(boxes, self.size), extra, meta
