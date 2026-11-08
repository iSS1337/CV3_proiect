"""Genereaza un set KITTI-Tracking sintetic (mic) + un video cu evenimente, pentru teste fara date reale."""
from pathlib import Path

import cv2
import numpy as np


def make_kitti(root: Path, n_seq: int = 6, n_frames: int = 12, size=(375, 1242)):
    rng = np.random.default_rng(0)
    for s in range(n_seq):
        seq = f"{s:04d}"
        (root / "image_02" / seq).mkdir(parents=True, exist_ok=True)
        (root / "label_02").mkdir(exist_ok=True)
        lines = []
        x = 300 + 40 * s
        for f in range(n_frames):
            img = rng.integers(60, 120, (*size, 3), dtype=np.uint8)
            x1, y1, x2, y2 = x + 5 * f, 150, x + 5 * f + 40, 260
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 200, 30), -1)
            cv2.imwrite(str(root / "image_02" / seq / f"{f:06d}.png"), img)
            lines.append(f"{f} 1 Pedestrian 0 0 0 {x1} {y1} {x2} {y2} 1.7 0.5 0.5 0 1.5 {20 - f} 0")
            lines.append(f"{f} 2 Car 0 0 0 800 180 1000 300 1.5 1.8 4 0 1.5 30 0")
            lines.append(f"{f} -1 DontCare -1 -1 -10 10 10 50 50 -1 -1 -1 -1000 -1000 -1000 -10")
        (root / "label_02" / f"{seq}.txt").write_text("\n".join(lines) + "\n")


def make_video(path: Path, n_frames: int = 120, fps: int = 30, size=(180, 320)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]))
    for i in range(n_frames):
        frame = np.full((*size, 3), i * 2 % 255, np.uint8)
        w.write(frame)
    w.release()
