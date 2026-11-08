"""Genereaza clipuri scurte de demonstratie din Dataset-urile proiectului (se pot pune in repository).

1) dashcam_demo.mp4 : un eveniment EMERGENCY din video-ul propriu, cu eticheta pe cadru data de DashcamEventDataset.
2) kitti_demo.mp4   : o secventa din split-ul de test KITTI, cu cutiile si distanta incarcate de KittiPedestrianDataset.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv3 import DashcamEventDataset, KittiPedestrianDataset  # noqa: E402
from cv3.dataset import LABEL_NAMES  # noqa: E402

COLORS = {0: (60, 180, 60), 1: (0, 165, 255), 2: (0, 0, 220)}  # BGR


def to_bgr(t):
    return np.ascontiguousarray((t.permute(1, 2, 0).numpy()[..., ::-1] * 255).astype(np.uint8))


def pick_event(events_csv, frame_labels, event_id=None):
    """Alege un eveniment EMERGENCY. Implicit: cel cu cele mai multe cadre EMERGENCY ramase dupa IGNORE
    (reluarile slow-motion sunt marcate IGNORE si nu sunt potrivite pentru demo)."""
    rows = [r for r in csv.DictReader(open(events_csv, newline="", encoding="utf-8")) if r["label"] == "EMERGENCY"]
    if event_id is not None:
        rows = [r for r in rows if int(r["event_id"]) == event_id]
    if not rows:
        raise SystemExit("Nu exista eveniment EMERGENCY potrivit.")

    def visible(r):
        return int((frame_labels[int(r["start_frame"]):int(r["end_frame"]) + 1] == 2).sum())

    r = max(rows, key=visible)
    if visible(r) == 0:
        raise SystemExit("Evenimentul ales este acoperit de IGNORE.")
    return int(r["event_id"]), int(r["start_frame"]), int(r["end_frame"])


def dashcam_demo(video, events, out, event_id, pad_s, width):
    ds = DashcamEventDataset(video, events, img_size=640, stride=1)
    eid, s, e = pick_event(events, ds.frame_labels, event_id)
    pad = int(pad_s * ds.fps)
    lo, hi = max(0, s - pad), min(ds.n_frames - 1, e + pad)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    step = max(1, round(ds.fps / 30))
    h = None
    writer = None
    for f in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (f - lo) % step:
            continue
        lab = int(ds.frame_labels[f])
        frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
        name = "IGNORE" if lab < 0 else LABEL_NAMES[lab]
        col = COLORS.get(lab, (128, 128, 128))
        cv2.rectangle(frame, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(frame, f"{name}   frame {f}   t={f / ds.fps:6.2f}s   event #{eid}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.rectangle(frame, (0, 44), (width, 52), col, -1)
        if writer is None:
            h = frame.shape[0]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), ds.fps / step, (width, h))
        writer.write(frame)
    cap.release()
    writer.release()
    print("scris", out, f"(frames {lo}..{hi}, event #{eid} {s}..{e})")


def kitti_demo(root, split_dir, out, split, max_frames, fps):
    ds = KittiPedestrianDataset(root, split, split_dir, augment=False, only_with_pedestrians=True)
    # secventa cu cei mai multi pietoni
    counts = {}
    for sq, fr in ds.index:
        counts[sq] = counts.get(sq, 0) + len(ds.annotations[sq][fr])
    best = max(counts, key=counts.get)
    items = [i for i, (sq, _) in enumerate(ds.index) if sq == best][:max_frames]
    writer = None
    for i in items:
        it = ds[i]
        img = to_bgr(it["image"])
        px, py = it["meta"]["pad"]
        h0, w0 = it["meta"]["orig_size"]
        r = it["meta"]["scale"]
        S = img.shape[0]
        for (cx, cy, w, h), d in zip(it["boxes"].tolist(), it["distance_m"].tolist()):
            x1, y1, x2, y2 = [int(v * S) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d:.1f}m", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        img = img[py:py + int(round(h0 * r))]
        cv2.rectangle(img, (0, 0), (260, 28), (0, 0, 0), -1)
        cv2.putText(img, f"{split} seq {it['seq']} frame {it['frame']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    print("scris", out, f"(seq {best}, {len(items)} cadre)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--events", default="data/events/ped_test_video_events.csv")
    ap.add_argument("--event-id", type=int, help="implicit: cel mai lung EMERGENCY")
    ap.add_argument("--kitti-root")
    ap.add_argument("--split-dir", default="data/splits")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="docs/demo")
    ap.add_argument("--pad-s", type=float, default=2.0, help="secunde inainte si dupa eveniment")
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--kitti-frames", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.video:
        dashcam_demo(a.video, a.events, out / "dashcam_demo.mp4", a.event_id, a.pad_s, a.width)
    if a.kitti_root:
        kitti_demo(a.kitti_root, a.split_dir, out / "kitti_demo.mp4", a.split, a.kitti_frames, 10)
