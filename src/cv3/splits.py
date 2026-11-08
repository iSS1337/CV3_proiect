"""Impartirea pe train/val/test, la nivel de SECVENTA (nu de cadru).

Cadrele consecutive dintr-un video sunt aproape identice; daca ar fi amestecate intre split-uri,
metricile (mAP, recall) ar fi umflate artificial. De aceea se imparte pe secvente intregi.
Rezultatul se salveaza in data/splits/*.txt, pentru reproductibilitate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def make_sequence_splits(
    seq_sizes: Dict[str, int],
    ratios=(0.70, 0.15, 0.15),
    seed: int = 42,
    seq_objects: Optional[Dict[str, int]] = None,
    n_trials: int = 50000,
    min_objects: int = 0,
) -> Dict[str, List[str]]:
    """Imparte secventele in train/val/test, pe secvente intregi, echilibrand CADRELE si (optional) PIETONII.

    seq_sizes:   {secventa: numar_cadre}
    seq_objects: {secventa: numar_cutii_pieton}; daca e dat, eroarea de proportii se calculeaza si pe pietoni.
    Cautare aleatoare cu seed fix (deterministica): se genereaza n_trials alocari, se alege cea cu eroarea minima
    (suma abaterilor absolute fata de `ratios`, pe cadre si pe pietoni). Fiecare split are cel putin o secventa;
    daca min_objects > 0, val si test trebuie sa aiba cel putin atatea cutii.
    """
    if len(seq_sizes) < 3:
        raise ValueError("Sunt necesare cel putin 3 secvente pentru train/val/test.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Proportiile trebuie sa insumeze 1.")

    names = sorted(seq_sizes)
    n = len(names)
    frames = np.array([seq_sizes[q] for q in names], dtype=float)
    objs = np.array([seq_objects[q] for q in names], dtype=float) if seq_objects else None
    r = np.array(ratios)
    rng = np.random.default_rng(seed)

    # alocari candidate: probabilitati = ratios, un vector de etichete (0/1/2) per secventa
    assign = rng.choice(3, size=(n_trials, n), p=r)
    onehot = np.stack([(assign == k) for k in range(3)], axis=2).astype(float)      # (T, n, 3)
    share_f = np.einsum("tnk,n->tk", onehot, frames) / frames.sum()
    cost = np.abs(share_f - r).sum(axis=1)
    if objs is not None and objs.sum() > 0:
        obj_cnt = np.einsum("tnk,n->tk", onehot, objs)
        cost += np.abs(obj_cnt / objs.sum() - r).sum(axis=1)
        valid_obj = (obj_cnt[:, 1] >= min_objects) & (obj_cnt[:, 2] >= min_objects)
    else:
        valid_obj = np.ones(n_trials, dtype=bool)
    nonempty = (onehot.sum(axis=1) >= 1).all(axis=1)
    valid = nonempty & valid_obj
    if not valid.any():
        raise ValueError("Nicio alocare valida (verifica min_objects / numarul de secvente).")
    cost[~valid] = np.inf
    best = assign[int(np.argmin(cost))]

    keys = ["train", "val", "test"]
    return {k: sorted(nm for nm, a in zip(names, best) if a == i) for i, k in enumerate(keys)}


def save_splits(splits: Dict[str, List[str]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, seqs in splits.items():
        (out_dir / f"{k}.txt").write_text("\n".join(seqs) + "\n", encoding="utf-8")


def load_splits(split_dir: str | Path) -> Dict[str, List[str]]:
    split_dir = Path(split_dir)
    return {
        k: [l.strip() for l in (split_dir / f"{k}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        for k in ("train", "val", "test")
    }
