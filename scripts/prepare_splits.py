"""Genereaza data/splits/{train,val,test}.txt din KITTI Tracking (split pe secvente, seed fix)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cv3.dataset import _resolve_kitti_root, parse_kitti_tracking_labels  # noqa: E402
from cv3.splits import make_sequence_splits, save_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="Folder KITTI Tracking (contine image_02/ si label_02/ sau training/)")
ap.add_argument("--out", default="data/splits")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--min-val-test-boxes", type=int, default=500, help="minim cutii pieton in val si in test")
a = ap.parse_args()

root = _resolve_kitti_root(Path(a.root))
sizes, peds = {}, {}
for lab in sorted((root / "label_02").glob("*.txt")):
    seq = lab.stem
    sizes[seq] = len(list((root / "image_02" / seq).glob("*.png")))
    peds[seq] = sum(len(v) for v in parse_kitti_tracking_labels(lab).values())
splits = make_sequence_splits(sizes, seed=a.seed, seq_objects=peds, min_objects=a.min_val_test_boxes)
save_splits(splits, a.out)
tot = sum(sizes.values())
for k, seqs in splits.items():
    fr = sum(sizes[s] for s in seqs)
    nb = sum(peds[s] for s in seqs)
    print(f"{k:5s}: {len(seqs):2d} secvente | {fr:5d} cadre ({100*fr/tot:4.1f}%) | {nb:5d} cutii ({100*nb/sum(peds.values()):4.1f}%) | {seqs}")
