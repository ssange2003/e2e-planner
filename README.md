# Structured Planner

Learned driving planner for the RC vehicle (Jetson Orin Nano + RealSense camera).

---

## Concept

### The Perception → Planner Split

Classic end-to-end driving feeds raw camera pixels directly into a neural network that outputs steering and throttle. That works but requires enormous amounts of data, is sensitive to lighting and visual domain, and is hard to debug.

This project takes a different approach:

```
Camera ──► YOLO              ──► object list (class, distance, position…)  ─┐
       ──► BiSeNet (LaneSeg) ──► 6×12 lane-pixel grid                      ─┤──► PLANNER ──► steering
                                                                             │               throttle
                               ego state (prev steering/throttle)          ─┘
```

**Your colleague** owns the left side (perception — camera, YOLO, lane segmentation).
**This module** owns the right side (planner — structured numbers in, actuation out).

The planner never touches pixels. It sees a fixed-size vector of normalised numbers describing the world and outputs two numbers: steering and throttle.

### Why this is better for this project

| Property | End-to-end (pixels → control) | This planner (features → control) |
|---|---|---|
| Data needed | Thousands of frames | Hundreds of rows |
| Sensitive to lighting | Yes | No (lane grid is binary mask) |
| Augmentation | Hard (image transforms) | Easy (perturb numbers / flip grid) |
| Model size | Millions of params | ~100 k params |
| Inference time on Jetson | 10–50 ms | < 1 ms |
| Debuggable | Hard | Read the CSV |

---

## Input / Output

### Planner Input (per frame, 114 floats + 1 scenario token)

**Object block** — top 5 closest YOLO detections, padded with zeros if fewer:

| Feature | Description | Range |
|---|---|---|
| `valid` | 1 if slot has a real object, 0 if padding | {0, 1} |
| `class_norm` | YOLO class ID ÷ (N_CLASSES − 1) | [0, 1] |
| `conf` | Detection confidence | [0, 1] |
| `dist_norm` | Distance ÷ 5 m | [0, 1] |
| `lat_offset` | Signed lateral offset from lane centre, normalised by lane width | (−∞, ∞) |
| `width_norm` | Bounding box width ÷ frame width | [0, 1] |
| `height_norm` | Bounding box height ÷ frame height | [0, 1] |
| `lane_overlap` | Fraction of lane width the object covers | [0, 1] |

5 objects × 8 features = **40 values**

**Lane block** — 6×12 spatial grid pooled from the BiSeNet segmentation mask:

The binary lane mask is resized to a coarse 64×112 image, then divided into 6 rows × 12 columns. Each cell stores the mean lane-pixel fraction [0.0–1.0]. Row 0 = far (top of image), Row 5 = near (bottom).

```
Far   [0.0][0.0][0.3][0.8][0.8] … ← road curves right ahead
      [0.0][0.1][0.5][0.9][0.9] …
      [0.0][0.2][0.7][1.0][1.0] …
      [0.0][0.3][0.8][1.0][1.0] …
      [0.1][0.4][0.9][1.0][1.0] …
Near  [0.2][0.5][1.0][1.0][1.0] … ← nearly centred now
```

6 rows × 12 cols = **72 values** (row-major: `lane_r0c0`, `lane_r0c1`, … `lane_r5c11`)

**Ego state** — previous cycle's output:

| Feature | Description |
|---|---|
| `ego_steering` | Previous steering command |
| `ego_throttle` | Previous throttle ÷ MAX_THROTTLE |

**2 values**

**Scenario token** — integer that tells the planner what it is supposed to be doing:

| Value | Name | When to use |
|---|---|---|
| 0 | LANE_FOLLOW | Normal track driving |
| 1 | LEFT_TURN | Turning left at junction |
| 2 | RIGHT_TURN | Turning right at junction |
| 3 | GO_STRAIGHT | Straight through intersection / past stop line |
| 4 | PULL_OVER | Pulling over to roadside (emergency stop) |
| 5 | PARKING | Parking manoeuvre |

### Planner Output

| Output | Range | Notes |
|---|---|---|
| `steering` | [−1, 1] | Negative = left, positive = right |
| `throttle` | [0, 1] | Multiplied by `MAX_THROTTLE` before sending to JetRacer |

---

## Model Architecture

```
objects  (40) ── Linear(40→128) ── LayerNorm ── ReLU ── Linear(128→128) ── ReLU ── Linear(128→64) ── ReLU ──┐
lane     (72) ── Linear(72→128) ── ReLU ────────────────────────────── Linear(128→128) ── ReLU ── Linear(128→64) ── ReLU ──┤
ego       (2) ── Linear(2→32)   ── ReLU ────────────────────────────────────────────────────────────────────────────────────┤ concat (168)
scenario  (1) ── Embedding(6,8) ─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                              Linear(168→256) ── ReLU ── Dropout(0.2)
                              Linear(256→128) ── ReLU ── Dropout(0.1)
                              Linear(128→64)  ── ReLU
                                    ├── Linear(64→1) ── Tanh()    → steering ∈ [−1, 1]
                                    └── Linear(64→1) ── Sigmoid() → throttle ∈ [ 0, 1]
```

Total trainable parameters: ~100,000. Trains in minutes on the Jetson.

---

## File Map

```
e2e-planner/
├── planner_model.py          ← shared definitions — model, feature builders, CSV schema
│                               import from this in everything else
│
├── collect_data_planner.py   ← Step 1: drive manually and log structured features
├── augment.py                ← Step 2: synthetically expand the dataset
├── train_planner.py          ← Step 3: train the planner model
├── evaluate.py               ← Step 4: offline error metrics + plots
├── planner_inference.py      ← Step 5: run the trained model on the vehicle
│
├── lane_seg.py               ← BiSeNet wrapper (loads model directly, no LKAS)
├── camera.py                 ← RealSense camera wrapper
├── planner_viewer.py         ← Web viewer for collection and inference
├── yolo_config.py            ← YOLO model path and thresholds
├── gamepads.py               ← Gamepad / controller input (optional)
├── dedup.py                  ← CSV deduplication utility
│
├── doc/
│   ├── ARCHITECTURE.md       ← lane feature design history and roadmap
│   └── WORKFLOW.md           ← end-to-end workflow notes
│
├── requirements.txt          ← Jetson dependencies (see install notes below)
├── requirements_desktop.txt  ← desktop-only deps (training / evaluation)
└── TROUBLESHOOTING.md
```

---

## Step-by-Step Guide

### Prerequisites

**PyTorch (Jetson — must install from Jetson AI Lab, not PyPI):**
```bash
python3 -m pip install torch torchvision \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

After installing torch, install the missing CUDA sparse solver (required on JetPack 6.x):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install libcudss0-cuda-12
echo "/usr/lib/aarch64-linux-gnu/libcudss/12" | sudo tee /etc/ld.so.conf.d/cudss.conf
sudo ldconfig
```

Tested: torch==2.10.0, torchvision==0.25.0, JetPack 6.2, CUDA 12.6.

**Other dependencies:**
```bash
pip install -r requirements.txt
# lkas and jetracer already installed as editable packages
```

---

### Step 1 — Collect Data

The collector runs standalone — no LKAS process required. BiSeNet is loaded directly via `lane_seg.py`.

```bash
# Normal track driving
python collect_data_planner.py --scenario 0

# Turning left at junction
python collect_data_planner.py --scenario 1

# Turning right at junction
python collect_data_planner.py --scenario 2

# Straight through intersection
python collect_data_planner.py --scenario 3

# Pull-over
python collect_data_planner.py --scenario 4

# Parking
python collect_data_planner.py --scenario 5
```

Open the web viewer in a browser: `http://<jetson-ip>:8082`

**Controls in the browser:**
- `←` / `→` — steer left / right (hold the key)
- `↓` — stop (throttle = 0)
- `0`–`5` — switch scenario token live
- `Space` — toggle recording ON/OFF (red badge = recording)
- `Ctrl+C` in terminal — quit and save

**Tips:**
- Collect at least ~300 rows per scenario before augmenting
- Cover edge cases: sharp corners, obstacle on left side, obstacle on right side, clear straight
- Check the live counter in the terminal to confirm rows are being saved
- If BiSeNet is not detecting lanes, a warning is printed after 30 consecutive no-lane saved rows — check camera angle and lighting

**Output:** `data/planner_data.csv` — one row per saved frame, appended across sessions.

**What each row contains:**
```
frame_id | obj0_valid … obj4_lane_overlap (40 cols) |
lane_r0c0 … lane_r5c11 (72 cols) |
ego_steering | ego_throttle | scenario | target_steering | target_throttle
```

---

### Step 2 — Augment

Expands the dataset ~8× using physically meaningful transforms:

```bash
python augment.py
# or specify paths explicitly:
python augment.py --input data/planner_data.csv --output data/augmented_data.csv
```

**What augmentation does:**

| Transform | Physical meaning |
|---|---|
| Identity | Keep original |
| Mirror | Horizontal flip — negate lateral offsets, steering, and flip lane grid columns |
| Distance noise | Simulate RealSense depth noise (σ = 3 cm normalised) |
| Lateral jitter | Simulate YOLO box jitter |
| Confidence noise | Simulate varying detection confidence |
| Object dropout | Simulate a missed detection (one object randomly removed) |
| Distance scale | Simulate depth calibration drift (±15%) |
| Mirror + noise | Combination of mirror and distance noise |

**Output:** `data/augmented_data.csv`

```
Before: 300 rows  →  After: ~2400 rows  (×8)
```

---

### Step 3 — Train

```bash
python train_planner.py

# Optional flags:
python train_planner.py \
    --csv    data/augmented_data.csv \
    --epochs 100 \
    --lr     3e-4 \
    --batch-size 64 \
    --output planner_model.pth
```

**Training uses augmented_data.csv by default, falls back to planner_data.csv if augmentation was skipped.**

During training you will see:
```
Epoch   Train Loss    Val Loss   Steer MAE   Thtl MAE  LR
    1   0.123456    0.134567    0.2341      0.0412   3.00e-04
    2   0.098765    0.112345    0.1987      0.0381   3.00e-04  ★ (best saved)
  ...
```

`★` marks epochs where the model improved on validation — the best checkpoint is saved automatically.

**Output:** `planner_model.pth`

Training typically converges in 30–80 epochs on ~2000 rows. On the Jetson Orin Nano this takes 2–5 minutes.

---

### Step 4 — Evaluate (offline)

Before putting the model on the vehicle, check its offline accuracy:

```bash
python evaluate.py

# Optional flags:
python evaluate.py \
    --csv     data/planner_data.csv \
    --model   planner_model.pth \
    --out-dir data/eval
```

**Output — printed to terminal:**
```
OVERALL RESULTS
  Samples          : 300
  Steering MAE     : 0.0821
  Steering RMSE    : 0.1134
  Throttle MAE     : 0.0043
  Throttle RMSE    : 0.0061

PER-SCENARIO RESULTS
  Scenario               N   Steer MAE   Steer RMSE   Thtl MAE
  LANE_FOLLOW          120      0.0412       0.0634     0.0021
  LEFT_TURN             80      0.1204       0.1543     0.0061
  PULL_OVER             50      0.0934       0.1123     0.0078
```

**Output — plots saved to `data/eval/`:**
- `steering_scatter.png` — predicted vs ground truth scatter
- `throttle_scatter.png` — same for throttle
- `steering_error_hist.png` — error distribution histogram
- `per_scenario_mae.png` — bar chart comparing scenarios
- `timeseries.png` — prediction tracking over 200 frames

**Reading the results:**
- Steering MAE < 0.10 is good
- If one scenario has much higher error → collect more data for that scenario
- A biased error histogram (not centred at 0) → the model is systematically off in one direction

---

### Step 5 — Inference on Vehicle

```bash
# Simulation first (no motor output):
python planner_inference.py --scenario 0

# Enable motors once you've verified the steering looks correct in the web viewer:
python planner_inference.py --scenario 0 --motor

# Left turn at junction:
python planner_inference.py --scenario 1 --motor

# Use a different model file:
python planner_inference.py --model planner_model.pth --scenario 0 --motor
```

**Web viewer:** `http://<jetson-ip>:8082`

The annotation overlay shows:
- Scenario name (colour-coded)
- Current predicted steering and throttle
- YOLO bounding boxes with distances
- Lane grid overlay (green cells = lane pixels)

**Terminal output (updated every second):**
```
[LANE_FOLLOW]  steer=+0.023  thr=0.200  objs=2  lane=YES  FPS=18.3
```

---

## System Diagram

```
                    DATA COLLECTION
┌─────────────────────────────────────────────────────┐
│  python collect_data_planner.py --scenario 0         │
│    RealSense ──► YOLO (CPU) ──► object features     │
│    RealSense ──► BiSeNet (GPU) ──► lane grid (72)   │
│    web viewer ────────────► human steering/throttle  │
│    all ────────────────────► planner_data.csv        │
└─────────────────────────────────────────────────────┘

                    OFFLINE PIPELINE
  planner_data.csv
       │
       ▼
  augment.py ──► augmented_data.csv (×8)
       │
       ▼
  train_planner.py ──► planner_model.pth
       │
       ▼
  evaluate.py ──► data/eval/*.png + summary

                    INFERENCE
┌─────────────────────────────────────────────────────┐
│  python planner_inference.py --scenario 0 --motor    │
│    RealSense ──► YOLO (CPU) ──► object features     │
│    RealSense ──► BiSeNet (GPU) ──► lane grid (72)   │
│    ego state ──────────────► ego features            │
│    --scenario flag ─────────► scenario token         │
│    all ─────────────────────► PlannerModel           │
│                                    │                 │
│                          [steering, throttle]        │
│                               │          │           │
│                          JetRacer    web viewer      │
└─────────────────────────────────────────────────────┘
```

---

## Iterating — What to Do When Performance Is Poor

**High steering error on a specific scenario:**
1. `python evaluate.py` — confirm which scenario is worst in `per_scenario_mae.png`
2. Collect more data for that scenario: `python collect_data_planner.py --scenario <N>`
3. Re-run augment + train + evaluate

**Model steers in the wrong direction consistently:**
- Check the mirror augmentation is working: mirrored rows should have negated steering and flipped lane grid columns
- Verify the JetRacer hardware inversion (`car.steering = -final_steering`) is correct for your vehicle

**Throttle always too high or too low:**
- Check `MAX_THROTTLE` in `planner_model.py` matches the `FULL_THROTTLE` value used in `planner_viewer.py`
- Default is `0.35`

**No lane detection (lane grid all zeros):**
- BiSeNet is not detecting lanes — check camera angle and lighting
- The model still operates but without lane information; collect dedicated data with BiSeNet running so the model learns both conditions

**FPS too low during inference:**
- YOLO runs on CPU (GPU is reserved for BiSeNet); reduce `YOLO_SKIP` to run YOLO less frequently
- The planner forward pass itself is < 1 ms — YOLO and BiSeNet are the bottlenecks

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Constants Reference

All shared constants live in `planner_model.py`. Change them there and they propagate everywhere.

| Constant | Default | Meaning |
|---|---|---|
| `N_MAX_OBJECTS` | 5 | Max YOLO detections tracked per frame |
| `OBJ_FEATURES` | 8 | Features per object slot |
| `GRID_ROWS` | 6 | Lane grid rows (far → near) |
| `GRID_COLS` | 12 | Lane grid columns (left → right) |
| `LANE_FEATURES` | 72 | Total lane grid cells (GRID_ROWS × GRID_COLS) |
| `MAX_DIST_M` | 5.0 | Distance normalisation ceiling (metres) |
| `MAX_THROTTLE` | 0.35 | Physical throttle ceiling for JetRacer |
| `FRAME_W` | 848 | Camera resolution width |
| `FRAME_H` | 480 | Camera resolution height |
| `N_YOLO_CLASSES` | 80 | YOLO class count (COCO default) |
| `N_SCENARIOS` | 6 | Scenario token vocabulary size |
