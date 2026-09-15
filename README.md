# IMU-Based Gait Event Detection: Unified Grid Evaluation

Analysis code for **"Unified Grid Evaluation of IMU-Based Gait Event Detection"** (ICCAS 2026)
and the extended presentation *"Robustness Benchmarking of Multi-Site IMU-Based Gait Event
Detection Algorithms Under Practical Perturbations"* (IFAC 2026, Busan).

Jihwan Park, Hyungseung Yang, Byunghyun Kang (corresponding)
IRLab, Dept. of AI & Robotics, Sejong University

## What this repository does

Detects heel strike (HS) and toe-off (TO) from wearable IMUs and evaluates every combination of

- **5 sensor sites**: foot, shank, thigh (affected), thigh (sound), lumbar
- **9 channels**: 3-axis accelerometer, 3-axis gyroscope, 3 Euler angles
- **4 classical algorithms**: threshold-based (classic / adaptive), band-pass filter, template matching

against video-labelled ground truth, then measures how each combination degrades under
practical perturbations (noise, sensor misalignment, packet loss, drift, sampling-rate reduction,
and their compounds). Deep baselines (1D-CNN, 1D-LSTM) are evaluated under the same
perturbation framework.

## Layout

```
.
├── src/                    core library
│   ├── config.py           subjects, sensors, channels, tolerances, paths
│   ├── algorithms.py       TB_classic / TB_adaptive / BPF / TM detectors
│   ├── algorithms_patch.py Nyquist-adaptive cutoff for resampled signals
│   ├── pipeline.py         Euler conversion, grid evaluation, F1 matching, statistics
│   ├── pipeline_phase2.py  perturbation robustness (degradation ratio, DR)
│   ├── pipeline_phase3.py  1D-CNN / 1D-LSTM models and training
│   ├── pipeline_phase4.py  deep-model evaluation under perturbations
│   ├── perturbations.py    perturbation definitions (P1–P7, C1–C4)
│   ├── perturbations_mc.py 9-channel Monte-Carlo perturbations, P7 time-axis correction
│   ├── traditional_ref.py  classical baselines under the deep-model conditions
│   ├── trials_adapter.py   resolves data locations (PC vs. container)
│   ├── visualize*.py       figures
│   └── run.py              phase runner
├── imu_loader.py           xlsx → per-sensor arrays
├── build_cache.py          raw cache (results/raw_cache/*.npz)
├── phase1_*.py             grid evaluation audit / run / tables
├── make_cells27.py         cells for Table 3 correction
├── phase2_corrected.py     per-site perturbation re-evaluation
├── table3_final.py         Table 3 final
├── train_one.py            train one deep model (site × arch)
├── run_phase4.py           deep-model perturbation sweep
├── make_tables.py / make_figs.py   paper tables and figures
├── tools/                  video–IMU synchronisation and manual labelling utilities
├── figures/                presentation figures
├── imu/                    raw IMU xlsx (not tracked)
├── output/synced/          ground-truth csv (not tracked)
└── results/                generated outputs (not tracked)
```

## Data

Four participants, treadmill walking at 2–5 km/h, 8 trials each (32 trials, ≈7,200 labelled
events). Sensors: EBIMU24GV6 at ≈90.9 Hz. Ground truth: 60 fps video, manually labelled,
aligned to the foot gyroscope zero-crossing.

Raw data and ground-truth files are **not** included. Place them as

```
imu/260519_PJH_IMU.xlsx ...            (see src/config.py SUBJECT_FILES)
output/synced/260519_PJH_*_synced.csv
```

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. GPU is not required; the full pipeline runs on CPU in roughly 40 minutes
after the cache is built.

## Reproducing the paper

```bash
# 0. raw cache (once, ~25 min)
python build_cache.py

# 1. grid evaluation: 5 sites × 9 channels × 4 algorithms
python phase1_grid.py
python phase1_tables.py

# 2. perturbation robustness (classical algorithms)
python make_cells27.py
python phase2_corrected.py shank
python phase2_corrected.py foot
python phase2_corrected.py thigh_aff
python phase2_corrected.py lumbar
python table3_final.py

# 3. deep baselines and their robustness
python train_one.py shank cnn
python train_one.py shank lstm
python train_one.py foot cnn
python train_one.py foot lstm
python run_phase4.py shank
python run_phase4.py foot

# 4. tables and figures
python make_tables.py
python make_figs.py
```

Legacy single-command entry point (phase 1–3):

```bash
python -m src.run --phase 1+2
```

## Key results (from the paper)

- Best single-channel classical detector: foot gyroscope with BPF, F1 = 0.860 on HS
- Shank gyroscope Z with threshold detection is the most robust classical choice across perturbations
- Compound perturbations degrade approximately multiplicatively (DR ≈ product of single DRs)
- Deep models reach F1 > 0.98 in-distribution but are sensitive to normalisation statistics under sensor misalignment

## Citation

```
@inproceedings{park2026unified,
  title     = {Unified Grid Evaluation of IMU-Based Gait Event Detection},
  author    = {Park, Jihwan and Yang, Hyungseung and Kang, Byunghyun},
  booktitle = {International Conference on Control, Automation and Systems (ICCAS)},
  year      = {2026}
}
```

## License

MIT
