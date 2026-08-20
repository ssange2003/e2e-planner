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

## LiDAR Sector Layout

The RPLidar returns 1000 points over 360°, which `process_lidar_to_5_sectors()`
collapses into five scalars. Index-to-angle conversion is `0.36° / index`:

| Sector | Index range | Angle | Role |
|---|---|---|---|
| `lidar_s0` | 83 – 166 | +29.9° … +59.8° | Left **side** |
| `lidar_s1` | 27 – 83 | +9.7° … +29.9° | Left **oblique** (front-left) |
| `lidar_s2` | 972 – 27 | −10.1° … +9.7° | **Front** |
| `lidar_s3` | 916 – 972 | −30.2° … −10.1° | Right **oblique** |
| `lidar_s4` | 833 – 916 | −60.1° … −30.2° | Right **side** |

**Why the distinction between side and oblique matters:**

For a wall `d` metres to the side, a ray at angle θ from the forward axis
reaches it at range `d / sin(θ)`. The closest reading a sector can produce is
therefore governed by its **largest** angle:

```
s0 :  d / sin(59.8°)  =  1.16 × d      (true side)
s1 :  d / sin(29.9°)  =  2.01 × d      (oblique)

s1 always reads 1.73× farther than s0 for the same wall.
```

Measured on `s_curve_course1.csv`: side `min(s0,s4)` = 0.305 m,
oblique `min(s1,s3)` = 0.500 m — a ratio of 1.64, matching the 1.73 prediction.

This is why **`SIDE_LIDAR_COLS` must be `s0`/`s4`**. Using `s1`/`s3` as "side"
makes the S-curve test unsatisfiable: with a 0.45 m threshold, `s1 & s3` fired
on **0.0%** of frames in every course, while `s0 & s4` fires on **59.0%** in the
S-curve course and **0.0%** in the open courses.

**Front vs side are answers to different questions:**

```
FRONT (s2)          "Is my path blocked?"        → STOP evidence
SIDE  (s0, s4)      "Is the corridor narrow?"    → S-curve / avoidance only
OBLIQUE (s1, s3)    "Is something cutting in?"   → avoidance only
```

The paper-cup S-curve has ~40 cm lanes, so the side sectors sit at 0.2–0.4 m
during *perfectly normal* driving. Feeding them into the STOP test would label
the entire S-curve as a stop. Only the front sector may promote a zero-throttle
event to STOP.

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
├── augment/                  ← Step 2: scenario-aware labelling + augmentation
│   ├── augment.py            ←   pipeline orchestration + CLI
│   ├── config.py             ←   every offline threshold, one place
│   ├── dataset_loader.py     ←   CSV load + filename → scenario prior
│   ├── sensor_evidence.py    ←   raw sensors → physical facts (no judgement)
│   ├── scenario.py           ←   prior + evidence → per-frame scenario label
│   ├── label_processor.py    ←   noise interpolation + protected smoothing
│   ├── augmentation.py       ←   scenario-aware transforms
│   ├── imu_check.py          ←   offline IMU calibration check
│   └── test_pipeline.py      ←   synthetic-case regression suite (23 checks)
├── train_planner.py          ← Step 3: train the planner model
├── evaluate.py               ← Step 4: offline error metrics + plots
├── planner_inference.py      ← Step 5: run the trained model on the vehicle
│
├── lane_seg.py               ← BiSeNet wrapper (loads model directly, no LKAS)
├── camera.py                 ← RealSense wrapper (+ D435i IMU, `--imu-check`)
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

Augmentation is no longer a single script — it is a five-stage pipeline that
**classifies what each frame actually was** before deciding how to treat it.

```bash
PYTHONPATH=. python augment/augment.py --input-dir data --output data/augmented_data.csv

# or list files explicitly:
PYTHONPATH=. python augment/augment.py --inputs data/normal_course1.csv data/stop_course1.csv
```

`PYTHONPATH=.` is required because `augment/config.py` imports the shared
constants from `planner_model.py` in the parent directory.

**Pipeline stages:**

```
CSV ──► DatasetLoader       filename → scenario prior, per-file _src_idx
    ──► SensorEvidence      raw sensors → physical facts only, no judgement
    ──► ScenarioClassifier  prior + evidence → per-frame scenario label
    ──► LabelProcessor      interpolate noise, smooth only unprotected frames
    ──► Augmentor           scenario-aware transforms
    ──► Training CSV
```

Each stage has exactly one responsibility. `SensorEvidence` never decides
*what a situation is* — it only reports distances and flags, because the same
"side at 0.25 m" is normal in the S-curve and a threat elsewhere. Only
`ScenarioClassifier` has the context needed to make that call.

**Filename is the prior.** Files must be named `{scenario}_course{N}.csv`:

| Prefix | Meaning |
|---|---|
| `normal_` | Ordinary lane-following |
| `avoidance_` | Steering around an obstacle without stopping |
| `s_curve_` | Narrow corridor (paper cups) — side sensors close by design |
| `stop_` | Deliberate full stop in front of an obstacle |
| `recovery_` | Restart after the obstacle is removed |
| `noise_` | Deliberately bad driving — excluded from training entirely |

The prior is applied **asymmetrically**. It can only ever *preserve* labels,
never delete them:

```
filename says stop, sensors show nothing   →  preserve (trust the human)
filename says normal, sensors show threat  →  preserve (trust the sensors)
```

A human can mislabel a file, so the prior is only trusted in the direction that
cannot destroy data. A frame is never forced into a scenario just because the
filename said so — `avoidance_course1.csv` measures only 17.5% actual avoidance,
the rest being ordinary driving before and after the manoeuvre.

**Scenario-aware augmentation:**

| Scenario | Mirror | Distance scale | Rationale |
|---|---|---|---|
| `normal`, `s_curve` | Yes | Yes | Fully symmetric, no directional meaning |
| `avoidance` | **No** | Yes | Which side you passed on *is* the behaviour |
| `stop`, `recovery` | Yes | **No** | Scaling distance breaks the "how close → why I stopped" link |
| `noise` | — | — | Dropped from the dataset before augmentation |

Policy is evaluated **per frame, not per file**. Applying it per file cost 742
rows of mirror augmentation on `avoidance_course1.csv` when only 198 rows were
genuinely directional.

**Zero-throttle events are classified, not blanket-deleted.**

Frames where throttle drops below `ZERO_EVENT_THRESH` are grouped into events,
and each event is judged as a whole:

```
evidence present (front obstacle / approach trend)  →  preserve as STOP
IMU says the car was still rolling                  →  gamepad glitch, interpolate
at a file boundary (no context either side)         →  preserve, never guess
```

**Output:** `data/augmented_data.csv`

```
Before: 2240 rows  →  After: ~17600 rows  (×7.9)

normal 1259 · noise 387 · avoidance 160 · s_curve 325 · stop 45 · recovery 64
```

Counts are printed at every run — read them. If a `stop_*.csv` file yields zero
`stop` frames, the classification is wrong and no amount of training will fix it.

---

### Step 2b — Verify IMU Calibration

Only relevant once you have collected data with a **D435i** (the plain D435 has
no IMU). Skip this if `imu_check.py` reports the columns are missing — the
pipeline falls back to LiDAR-only evidence automatically.

```bash
# live check, vehicle stationary — confirms mount axes
python camera.py --imu-check

# offline check against collected data — confirms thresholds
python augment/imu_check.py data/normal_course1.csv
```

**Why the IMU is needed at all:**

A gamepad glitch and a deliberate stop produce *identical* readings on every
external sensor. The scene ahead of the car is the same in both cases. Only
ego-motion separates them:

| Situation | throttle | LiDAR / camera | IMU |
|---|---|---|---|
| Glitch (still coasting) | 0 | same as normal driving | **motion detected** |
| Real stop | 0 | same as normal driving | **stationary** |

The IMU is **not integrated** — double-integrating acceleration drifts within
seconds. Instead `imu_motion` is the standard deviation of `|accel|` over a
0.2 s window: a rolling car vibrates, a stopped car does not. There is no
accumulating error because there is no accumulation.

**What to read from `imu_check.py`:**

```
[2] separation ratio    ≥ 3.0×   → IMU judgement is trustworthy
                        1.5–3.0× → borderline, choose the threshold carefully
                        < 1.5×   → vibration cannot separate; consider an encoder

[3] knee of the moving-ratio curve → the real MOTOR_DEAD_ZONE_MAX
```

Stage `[3]` is the only reliable way to determine ESC breakaway. The throttle
histogram alone cannot: the 0.825–0.850 bin is the mode of the entire dataset,
so the current 0.85 threshold sits on top of a peak with no natural gap on
either side. The IMU answers it directly by showing at which throttle the car
actually starts moving.

Put the recommended value into `augment/config.py` **before** running Step 2 —
the shipped `IMU_MOTION_THRESH = 0.15` is a placeholder with no measurement
behind it.

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

**The car will not move after training (throttle output stuck near zero):**

This is the single failure mode the whole augment pipeline exists to prevent.
It is a data problem, not a training problem — more epochs make it worse.

*Mechanism.* The gamepad throttle passes through a dead zone in
`collect_data_planner.py`. With a single threshold, a stick that hovers near the
boundary snaps to exactly `0.0` and back every frame, while the driver is still
driving. Measured across 2240 collected rows:

```
frames at exactly 0.0        520  (23.2%)
smallest non-zero value      0.11863
                             ^ nothing at all exists between 0 and 0.119
```

Continuous human input cannot produce that gap — it is a staircase carved by the
dead-zone code. The next transition confirms it: after a zero run ends, throttle
jumps straight to a median of **0.609** in one frame. A real restart ramps
`0 → 0.4 → 0.7 → 0.95`; a dead-zone artefact snaps back to where the thumb
already was.

*Why the model then refuses to move.* Those glitch frames are visually and
geometrically **identical** to the frames around them — same lane grid, same
LiDAR, same objects. So the dataset now contains contradictory labels on
effectively identical inputs: "drive at 0.85" and "stop" for the same scene.
MSE regression has exactly one way to minimise loss against contradictory
targets — it predicts their mean. With 23% of labels pulled to zero, the
throttle head collapses toward the floor and the car crawls or stalls.

*Fixes, in order of leverage:*

1. **Stop generating the artefact.** `collect_data_planner.py` now uses a
   hysteresis dead zone (`THROTTLE_ENGAGE_TH` 0.08 / `THROTTLE_RELEASE_TH` 0.03)
   so the stick must fall much further to release than it did to engage.
2. **Filter what already exists.** Step 2 classifies zero-throttle events and
   interpolates the ones with no physical evidence behind them. Check the
   `noise` count in the augment output — above ~20% means the dead zone is still
   too aggressive.
3. **Measure ego-motion.** With a D435i, the classifier stops guessing: if the
   car was rolling during a zero-throttle event, it is a glitch regardless of
   what the LiDAR saw. See Step 2b.

*Diagnosis.* Check the label distribution actually being trained on:

```bash
python -c "import pandas as pd; d=pd.read_csv('data/augmented_data.csv'); print((d.target_throttle==0).mean())"
```

Above ~0.10 and the throttle head will be biased low. Also verify the model is
not simply predicting a constant — if `evaluate.py` shows throttle MAE close to
the standard deviation of the labels, it has learned the mean and nothing else.

**Real stops and restarts are missing from the dataset:**

`stop` and `recovery` are the two behaviours most easily destroyed by smoothing
and the hardest to recover afterwards, so they need deliberate collection:

```bash
python collect_data_planner.py     # save as stop_course1.csv / recovery_course1.csv
```

Drive the restart **slowly and on purpose** (`0 → 0.4 → 0.7 → 0.85 → 0.95`).
Ordinary track laps never contain that ramp — every zero-to-motion transition in
them is a dead-zone artefact instead, which is precisely what gets filtered out.

**Scenario classification depends on the filename instead of the sensors:**

The prior is meant to assist evidence, not replace it. Test by stripping the
filenames and re-classifying:

```bash
PYTHONPATH=".;augment" python -c "
import sys; sys.path.insert(0,'augment')
from dataset_loader import DatasetLoader
from sensor_evidence import compute_sensor_evidence
from scenario import ScenarioClassifier
import glob
df = DatasetLoader().load(glob.glob('data/*_course*.csv'))
ev = compute_sensor_evidence(df, df['_src_idx'], 5)
df2 = df.copy(); df2['_scenario_type'] = 'normal'
d = ScenarioClassifier(5).classify(df2, ev)
for f in df['_source_file'].unique():
    m = (df['_source_file'] == f).to_numpy()
    print(f, round(d['is_s_curve'].to_numpy()[m].mean()*100, 1))
"
```

An `s_curve_*` file should still score above 60% with the prior removed, and
`normal_*` files below 5%. If the S-curve file drops to near zero, the geometry
is not discriminating and the classifier will fail on any new course — this is
exactly the symptom that exposed the `s1/s3` vs `s0/s4` sector error.

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
| `IMU_COLUMNS` | 3 cols | D435i derived features — **not** part of `csv_columns()` |

`IMU_COLUMNS` is deliberately kept out of `csv_columns()`. Collection writes the
extended schema via `csv_columns_ext()`, while training and augmentation still
validate against the base schema with a subset check
(`missing = expected - set(df.columns)`). Old CSVs without IMU and new CSVs with
it both load, and because model inputs are assembled from explicitly named
columns, **adding the IMU changes neither the model dimensions nor requires
retraining**.

### Augment Pipeline Constants

Thresholds used only for offline labelling live in `augment/config.py`. Nothing
here runs on the vehicle — `planner_inference.py` does not import `augment/` at
all, so tuning these has exactly zero effect on the control loop.

Each value carries a tag: `[실측]` measured against collected data,
`[추론]` physically reasoned but not calibrated, `[미검증]` weakly supported.

| Constant | Default | Meaning |
|---|---|---|
| `ZERO_EVENT_THRESH` | 0.10 | Below this, a frame is a zero-throttle candidate. Safe anywhere in 0.001–0.10 — the data is empty across that whole span |
| `MOTOR_DEAD_ZONE_MAX` | 0.85 | ESC breakaway. **Unverified** — sits on the distribution mode; determine it with `imu_check.py` |
| `LIDAR_DANGER_M` | 0.45 | Absolute front threat distance. The former 0.30 never fired once in 2240 rows |
| `LIDAR_CLEAR_M` | 0.50 | Threat-clear distance. Differs from `DANGER` on purpose — the gap is hysteresis |
| `LIDAR_RATIO` | 0.60 | Relative threat vs per-file baseline. Always ANDed with `LIDAR_CLEAR_M` |
| `FRONT_LIDAR_COL` | `lidar_s2` | The only sector allowed to promote an event to STOP |
| `SIDE_LIDAR_COLS` | `s0`, `s4` | True side. Never STOP evidence — see LiDAR Sector Layout |
| `OBLIQUE_LIDAR_COLS` | `s1`, `s3` | Front-oblique. Avoidance evidence only |
| `S_CURVE_SIDE_CLOSE_M` | 0.45 | Both sides closer than this = narrow corridor |
| `S_CURVE_FRONT_OPEN_M` | 0.45 | Front clear of this = passable. Matches `LIDAR_DANGER_M` so the two tests cannot contradict |
| `MIN_STOP_FRAMES` | 2 | Frames before an event is promoted to STOP. 0.2 s at `SAVE_FPS=10` |
| `APPROACH_TREND_WINDOW` | 5 | Look-back for the approach trend. 0.5 s at `SAVE_FPS=10` |
| `LANE_BASELINE_MIN` | 0.05 | Below this baseline, lane evidence is ignored — BiSeNet reads 0.035 in the S-curve |
| `CAMERA_EVIDENCE_ENABLED` | `False` | YOLO has no paper-cup class, so this signal is structurally always false |
| `IMU_MOTION_THRESH` | 0.15 | **Placeholder.** Calibrate with `imu_check.py` before trusting it |
| `IMU_YAW_THRESH` | 0.30 | Turning threshold in rad/s. Bypasses the gamepad's discrete steering |

Two constants were tuned to the same value on purpose: `LIDAR_DANGER_M` and
`S_CURVE_FRONT_OPEN_M` are both 0.45 so that a single boundary decides whether
the path ahead is blocked. Splitting them creates frames that are simultaneously
"threatened" and "open", which lights up STOP and S_CURVE at once.

Anything at `SAVE_FPS` resolution is worth double-checking when you change the
collection rate — several of these constants were originally written assuming
30 fps while the collector actually saves at 10 fps, making them 3× longer than
their comments claimed.
