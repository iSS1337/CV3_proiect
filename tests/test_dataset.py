import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synthetic import make_kitti, make_video  # noqa: E402
from cv3 import DashcamEventDataset, KittiPedestrianDataset, get_dataloaders, make_sequence_splits  # noqa: E402
from cv3.splits import load_splits, save_splits  # noqa: E402
from cv3.dataset import events_to_frame_labels, LABEL_IGNORE  # noqa: E402


@pytest.fixture(scope="module")
def kitti(tmp_path_factory):
    d = tmp_path_factory.mktemp("kitti")
    make_kitti(d)
    sizes = {f"{i:04d}": 12 for i in range(6)}
    splits = make_sequence_splits(sizes, seed=1)
    save_splits(splits, d / "splits")
    return d, d / "splits", splits


def test_splits_disjoint_complete_nonempty(kitti):
    _, _, sp = kitti
    allseq = sp["train"] + sp["val"] + sp["test"]
    assert sorted(allseq) == [f"{i:04d}" for i in range(6)]
    assert len(set(allseq)) == len(allseq)
    assert all(len(v) >= 1 for v in sp.values())


def test_splits_deterministic():
    s = {f"{i:04d}": 10 + i for i in range(21)}
    assert make_sequence_splits(s, seed=42) == make_sequence_splits(s, seed=42)
    r = make_sequence_splits(s)
    tot = sum(s.values())
    assert 0.55 < sum(s[q] for q in r["train"]) / tot < 0.85


def test_item_shapes_and_ranges(kitti):
    root, sd, _ = kitti
    ds = KittiPedestrianDataset(root, "val", sd, img_size=320)
    it = ds[0]
    assert it["image"].shape == (3, 320, 320) and it["image"].dtype == torch.float32
    assert it["boxes"].shape == (1, 4)          # doar Pedestrian; Car/DontCare sunt filtrate
    assert 0 <= it["boxes"].min() and it["boxes"].max() <= 1
    assert it["distance_m"].shape == (1,) and it["distance_m"][0] > 0


def test_box_matches_pixels_without_augmentation(kitti):
    root, sd, _ = kitti
    ds = KittiPedestrianDataset(root, "test", sd, img_size=640, augment=False)
    it = ds[0]
    cx, cy, w, h = (it["boxes"][0] * 640).tolist()
    patch = it["image"][:, int(cy), int(cx)]       # centrul cutiei = dreptunghiul verde (RGB ~ 30,200,30)
    assert patch[1] > 0.6 and patch[0] < 0.3


def test_train_augment_keeps_boxes_valid(kitti):
    root, sd, _ = kitti
    ds = KittiPedestrianDataset(root, "train", sd, seed=3)
    for i in range(len(ds)):
        b = ds[i]["boxes"]
        assert b.numel() == 0 or (b.min() >= 0 and b.max() <= 1 and (b[:, 2:] > 0).all())


def test_dataloader_batch(kitti):
    root, sd, _ = kitti
    dl = get_dataloaders(str(root), str(sd), img_size=256, batch_size=4, num_workers=0)
    b = next(iter(dl["val"]))
    assert b["images"].shape[1:] == (3, 256, 256)
    assert b["targets"].shape[1] == 6 and len(b["boxes"]) == b["images"].shape[0]


def test_events_and_video(tmp_path):
    csv = tmp_path / "e.csv"
    csv.write_text("event_id,label,start_frame,end_frame,start_s,end_s,note\n"
                   "1,WARNING,10,30,,,\n2,EMERGENCY,30,40,,,\n3,IGNORE,80,99,,,\n")
    lab = events_to_frame_labels(csv, 120)
    assert lab[9] == 0 and lab[10] == 1 and lab[30] == 2 and lab[40] == 2 and lab[41] == 0 and lab[90] == LABEL_IGNORE
    v = tmp_path / "v.mp4"
    make_video(v)
    ds = DashcamEventDataset(v, csv, img_size=128, stride=5)
    assert all(ds.frame_labels[f] != LABEL_IGNORE for f in ds.indices)
    it = ds[len(ds) - 1]
    assert it["image"].shape == (3, 128, 128)


def test_unclosed_event_rejected(tmp_path):
    csv = tmp_path / "e.csv"
    csv.write_text("event_id,label,start_frame,end_frame,start_s,end_s,note\n1,WARNING,10,,,,\n")
    with pytest.raises(ValueError):
        events_to_frame_labels(csv, 50)


def test_splits_balance_pedestrians():
    # 21 secvente cu pietoni foarte inegal repartizati (ca in KITTI)
    rng = np.random.default_rng(5)
    frames = {f"{i:04d}": int(rng.integers(100, 800)) for i in range(21)}
    objs = {k: int(rng.choice([0, 20, 50, 400, 2000])) for k in frames}
    sp = make_sequence_splits(frames, seed=42, seq_objects=objs, min_objects=200)
    tot = sum(objs.values())
    assert sum(objs[q] for q in sp["val"]) >= 200 and sum(objs[q] for q in sp["test"]) >= 200
    assert sp == make_sequence_splits(frames, seed=42, seq_objects=objs, min_objects=200)
    assert sorted(sum(sp.values(), [])) == sorted(frames)
