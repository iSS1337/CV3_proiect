# CV3 - ADAS AEB: detectia pietonilor si estimarea riscului de coliziune (SIVA 421)

## Commit #1: Dataset, DataLoader, split-uri

```
src/cv3/dataset.py     KittiPedestrianDataset (train/val/test) si DashcamEventDataset (video propriu)
src/cv3/transforms.py  letterbox 640, normalizare, augmentari la train (flip, crop/zoom, culoare)
src/cv3/splits.py      split 70/15/15 pe secvente, salvat in data/splits/
src/cv3/dataloader.py  get_dataloaders(), collate pentru numar variabil de cutii (format YOLO)
scripts/               prepare_splits.py, verify_dataset.py
tests/                 teste pe date sintetice
data/events/           adnotarile video-ului propriu (NORMAL/WARNING/EMERGENCY/IGNORE)
```

### Date
- Detectie (train/val/test): KITTI Tracking, "left color images" si "training labels"
  (https://www.cvlibs.net/datasets/kitti/eval_tracking.php). Se dezarhiveaza in `data/kitti/`, astfel incat sa existe
  `data/kitti/training/image_02/<seq>/*.png` si `data/kitti/training/label_02/<seq>.txt`.
  Se pastreaza doar clasa `Pedestrian`. Distanta `z` din etichete ramane ca referinta pentru validarea distantei si TTC.
- Video propriu: `vids/ped_test_video.mp4` + `data/events/ped_test_video_events.csv`. Adnotarile sunt pe intervale de cadre,
  fara bounding box-uri, deci video-ul e folosit doar ca set de test pentru sistemul ADAS, nu pentru mAP.
  Regiunile IGNORE (reluari slow-motion) sunt excluse.

### Split-uri (KITTI Tracking, pe secvente, seed 42)
| Split | Secvente | Cadre | Cutii pieton |
|---|---|---|---|
| train | 14 | 5611 (70.1%) | 7639 (66.6%) |
| val | 4 | 1307 (16.3%) | 1711 (14.9%) |
| test | 3 | 1090 (13.6%) | 2120 (18.5%) |

Split-ul se face pe secvente intregi, ca sa nu ajunga cadre aproape identice in split-uri diferite. Se echilibreaza si cadrele,
si pietonii (minim 500 de cutii in val si in test).

Video propriu: 30069 cadre NORMAL, 9275 WARNING, 3647 EMERGENCY, 582 IGNORE.

### Verificare
`docs/verify/` contine cadre cu cutiile incarcate de DataLoader (cu distanta KITTI in metri) pentru train/val/test si cadre
din video-ul propriu pentru WARNING si EMERGENCY.

### Clipuri demo
`docs/demo/dashcam_demo.mp4`: eveniment EMERGENCY din video-ul propriu, cu eticheta data de DashcamEventDataset pe fiecare cadru.
`docs/demo/kitti_demo.mp4`: secventa din split-ul de test, cu cutiile si distanta incarcate de KittiPedestrianDataset.
Se genereaza cu `python scripts/make_demo.py --video vids/ped_test_video.mp4 --kitti-root data/kitti`.

### Rulare
```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/prepare_splits.py --root data/kitti
python scripts/verify_dataset.py --kitti-root data/kitti --video vids/ped_test_video.mp4
pytest tests -q
```
