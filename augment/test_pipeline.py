#!/usr/bin/env python3
"""
test_pipeline.py — synthetic 데이터로 핵심 케이스 검증
======================================================

CASE A: 측면만 가까움(0.25m) + 정면 열림(1.0m) + throttle 정상 → STOP 아님
CASE B: 정면 0.25m + throttle 0                                → STOP 후보
CASE C: STOP 후 0→0.4→0.7→0.9→0.95                              → recovery 보존
CASE D: throttle 유지 + 장애물 접근                              → avoidance 보호
CASE E: noise 파일                                              → 학습 데이터 제외
"""

import sys
import numpy as np
import pandas as pd

from config import csv_columns, GRID_ROWS, GRID_COLS, N_MAX_OBJECTS
from sensor_evidence import compute_sensor_evidence
from scenario import ScenarioClassifier
from label_processor import LabelProcessor
from augmentation import Augmentor


def make_frames(n, front, side_l, side_r, throttle, steering,
                scenario_type="normal", lane_val=0.02):
    """테스트용 DataFrame 생성."""
    cols = csv_columns()
    df = pd.DataFrame(0.0, index=range(n), columns=cols)
    df["frame_id"] = np.arange(1, n + 1)
    df["lidar_s0"] = 5.0
    df["lidar_s4"] = 5.0
    df["lidar_s2"] = front
    df["lidar_s1"] = side_l
    df["lidar_s3"] = side_r
    df["target_throttle"] = throttle
    df["target_steering"] = steering
    df["scenario"] = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            df[f"lane_r{r}c{c}"] = lane_val
    df["_src_idx"] = 0
    df["_scenario_type"] = scenario_type
    df["_course_id"] = 1
    df["_source_file"] = f"{scenario_type}_course1.csv"
    return df


def classify(df):
    ev = compute_sensor_evidence(df, df["_src_idx"], 5)
    return ScenarioClassifier(5).classify(df, ev)


results = []


def check(name, condition, detail=""):
    results.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))


print("=" * 68)
print("  Pipeline verification")
print("=" * 68)

# ── CASE A: 측면만 가까움 → STOP 되면 안 됨 ─────────────────────────
print("\nCASE A: side 0.25m / front 1.0m / throttle 유지 → STOP 아니어야 함")
n = 40
df = make_frames(n, front=1.0, side_l=0.25, side_r=0.25,
                 throttle=0.90, steering=0.0, scenario_type="s_curve")
d = classify(df)
check("STOP 프레임 0개", int(d["is_stop"].sum()) == 0,
      f"stop={int(d['is_stop'].sum())}")
check("S_CURVE로 인식", int(d["is_s_curve"].sum()) > 0,
      f"s_curve={int(d['is_s_curve'].sum())} frames")
check("보호 대상에 포함", int(d["is_protected"].sum()) > 0,
      f"protected={int(d['is_protected'].sum())}")

# ── CASE B: 정면 근접 + throttle 0 → STOP 후보 ──────────────────────
print("\nCASE B: front 0.25m / throttle 0 (10프레임) → STOP 후보")
front = np.concatenate([np.full(10, 1.5), np.full(10, 0.25), np.full(20, 1.5)])
thr = np.concatenate([np.full(10, 0.90), np.full(10, 0.0), np.full(20, 0.90)])
df = make_frames(40, front=front, side_l=3.0, side_r=3.0,
                 throttle=thr, steering=0.0, scenario_type="stop")
d = classify(df)
check("STOP 검출됨", int(d["is_stop"].sum()) >= 10,
      f"stop={int(d['is_stop'].sum())} frames")
check("노이즈로 지워지지 않음", int(d["is_noise_event"].sum()) == 0,
      f"noise={int(d['is_noise_event'].sum())}")

# ── CASE C: STOP 후 점진적 재출발 → recovery 보존 ───────────────────
print("\nCASE C: STOP 후 throttle 0→0.4→0.7→0.9→0.95 → recovery 보존")
front = np.concatenate([np.full(5, 1.5), np.full(10, 0.25), np.full(25, 2.0)])
thr = np.concatenate([
    np.full(5, 0.90), np.full(10, 0.0),
    [0.4, 0.7, 0.9, 0.95], np.full(21, 0.95),
])
df = make_frames(40, front=front, side_l=3.0, side_r=3.0,
                 throttle=thr, steering=0.0, scenario_type="recovery")
d = classify(df)
n_rec = int(d["is_recovery"].sum())
check("recovery 검출됨", n_rec >= 1, f"recovery={n_rec} frames")

proc = LabelProcessor(5).process(df, d)
transition = proc["target_throttle"].iloc[15:19].to_numpy()
orig_transition = np.array([0.4, 0.7, 0.9, 0.95])
check("재출발 transition 원본 보존",
      np.allclose(transition, orig_transition, atol=0.02),
      f"{np.round(transition, 3).tolist()}")

# ── CASE D: throttle 유지 + 장애물 접근 → avoidance ─────────────────
print("\nCASE D: throttle 유지 + 정면 접근 → avoidance 보호")
front = np.concatenate([np.full(10, 2.0), np.full(10, 0.25), np.full(20, 2.0)])
steer = np.concatenate([np.zeros(10), np.full(10, 0.9), np.zeros(20)])
df = make_frames(40, front=front, side_l=3.0, side_r=3.0,
                 throttle=0.90, steering=steer, scenario_type="avoidance")
d = classify(df)
check("avoidance 검출됨", int(d["is_avoidance"].sum()) > 0,
      f"avoidance={int(d['is_avoidance'].sum())} frames")
check("STOP으로 오분류 안 됨", int(d["is_stop"].sum()) == 0,
      f"stop={int(d['is_stop'].sum())}")

proc = LabelProcessor(5).process(df, d)
peak_steer = proc["target_steering"].iloc[10:20].abs().max()
check("회피 조향 크기 보존(스무딩 안 됨)", peak_steer >= 0.89,
      f"peak|steer|={peak_steer:.3f}")

# ── CASE E: noise 파일 → 학습 데이터 제외 ───────────────────────────
print("\nCASE E: noise 파일 → 증강 결과에서 제외")
df_noise = make_frames(30, front=3.0, side_l=3.0, side_r=3.0,
                       throttle=0.90, steering=0.0, scenario_type="noise")
df_norm = make_frames(30, front=3.0, side_l=3.0, side_r=3.0,
                      throttle=0.90, steering=0.0, scenario_type="normal")
df_norm["_src_idx"] = 1
both = pd.concat([df_noise, df_norm], ignore_index=True)
d = classify(both)
aug = Augmentor(42).augment(both, d["scenario_label"])
stats = Augmentor(42).augment(both, d["scenario_label"]) is not None
a = Augmentor(42)
aug = a.augment(both, d["scenario_label"])
check("noise 행 제외됨", a.last_stats["excluded_noise_rows"] == 30,
      f"excluded={a.last_stats['excluded_noise_rows']} rows")
check("normal만 증강됨", a.last_stats["input_rows"] == 30,
      f"input={a.last_stats['input_rows']} rows → {a.last_stats['output_rows']}")

# ── 파일 경계 누수 검증 ─────────────────────────────────────────────
print("\nEXTRA: 파일 경계 누수 검증")
df_a = make_frames(20, front=3.0, side_l=3.0, side_r=3.0,
                   throttle=0.95, steering=0.0)
df_b = make_frames(20, front=3.0, side_l=3.0, side_r=3.0,
                   throttle=0.20, steering=0.0)
df_b["_src_idx"] = 1
merged = pd.concat([df_a, df_b], ignore_index=True)
d = classify(merged)
proc = LabelProcessor(5).process(merged, d)
# 파일 A의 마지막 프레임이 파일 B 값에 오염되지 않아야 함
last_a = proc["target_throttle"].iloc[19]
check("파일 A 끝 프레임이 B에 오염 안 됨", last_a > 0.90,
      f"A[last]={last_a:.3f} (B는 0.20)")

# ── 결과 ────────────────────────────────────────────────────────────
print()
print("=" * 68)
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print(f"  RESULT: {n_pass}/{n_total} passed")
print("=" * 68)
sys.exit(0 if n_pass == n_total else 1)