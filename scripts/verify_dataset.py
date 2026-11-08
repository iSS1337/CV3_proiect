"""Verificare existenta + functionalitate Dataset/DataLoader. Scrie vizualizari in outputs/verify/."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv3 import DashcamEventDataset, KittiPedestrianDataset, detection_collate, get_dataloaders  # noqa: E402
from cv3.dataset import LABEL_NAMES  # noqa: E402

def to_bgr(t):
    return np.ascontiguousarray((t.permute(1, 2, 0).numpy()[..., ::-1] * 255).astype(np.uint8))


def draw(img, boxes_yolo, dist=None):
    s = img.shape[0]
    for k, (cx, cy, w, h) in enumerate(boxes_yolo.tolist()):
        x1, y1, x2, y2 = [int(v * s) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if dist is not None:
            cv2.putText(img, f"{dist[k]:.1f}m", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split-dir", default="data/splits")
    ap.add_argument("--video", help="ped_test_video.mp4 (optional)")
    ap.add_argument("--events", default="data/events/ped_test_video_events.csv")
    ap.add_argument("--out", default="outputs/verify")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


    print("== KITTI splits ==")
    for sp in ("train", "val", "test"):
        st = KittiPedestrianDataset(a.kitti_root, sp, a.split_dir).stats()
        print(f"{sp:5s}: {st['frames']} cadre, {st['frames_with_pedestrians']} cu pietoni, {st['pedestrian_boxes']} cutii")

    sets = {sp: set(KittiPedestrianDataset(a.kitti_root, sp, a.split_dir).sequences) for sp in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "Scurgere intre split-uri!"
    print("OK: split-urile nu au secvente comune")

    loaders = get_dataloaders(a.kitti_root, a.split_dir, batch_size=a.batch_size, num_workers=a.workers)
    for sp, dl in loaders.items():
        b = next(iter(dl))
        assert b["images"].shape[1:] == (3, 640, 640)
        assert 0 <= b["images"].min() and b["images"].max() <= 1
        if len(b["targets"]):
            assert b["targets"][:, 2:].min() >= 0 and b["targets"][:, 2:].max() <= 1, "cutii in afara [0,1]"
        print(f"{sp}: batch images {tuple(b['images'].shape)}, targets {tuple(b['targets'].shape)}")

    # vizualizari: cadre CU pietoni, repartizate uniform in fiecare split
    for sp in ("train", "val", "test"):
        ds = KittiPedestrianDataset(a.kitti_root, sp, a.split_dir, augment=False, only_with_pedestrians=True)
        idx = np.linspace(0, len(ds) - 1, 4).astype(int)
        vb = detection_collate([ds[int(i)] for i in idx])
        tiles = [draw(to_bgr(im), bx, d) for im, bx, d in zip(vb["images"], vb["boxes"], vb["distance_m"])]
        cv2.imwrite(str(out / f"batch_{sp}.jpg"), np.hstack(tiles))

    if a.video:
        ds = DashcamEventDataset(a.video, a.events)
        print("== Dashcam ==", ds.stats())
        for lab in range(3):
            match = [i for i, f in enumerate(ds.indices) if ds.frame_labels[f] == lab]
            idx = match[len(match) // 2] if match else None  # cadru din mijlocul categoriei
            if idx is not None:
                cv2.imwrite(str(out / f"dashcam_{LABEL_NAMES[lab]}.jpg"), to_bgr(ds[idx]["image"]))
    print("Vizualizari in", out)


if __name__ == "__main__":
    main()
