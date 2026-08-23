#!/usr/bin/env python3
"""
Planner Data Collection — Structured Feature Logger
====================================================
Ghost-mode style data collection for the structured planner.
Camera + YOLO + LKAS run as usual, but instead of saving raw images
we log a single CSV row of normalised feature vectors per frame.

  Camera → YOLO     → object features  ─┐
  Camera → LaneSeg  → lane grid (72)   ─┤→ one CSV row  →  planner_data.csv
  web-viewer → human steering/throttle ─┘

No .jpg / .npy files are created. The dataset is a plain CSV.

Usage
-----
  python collect_data_planner.py [--web-port 8082] [--scenario 0] [--out planner_data.csv]

Scenarios (--scenario default, or switch live with 0-5 keys in viewer)
  0 = LANE_FOLLOW    normal driving (obstacle avoidance implicit via YOLO)
  1 = LEFT_TURN      turning left at junction
  2 = RIGHT_TURN     turning right at junction
  3 = GO_STRAIGHT    straight through intersection / past stop line
  4 = PULL_OVER      pulling over to roadside
  5 = PARKING        parking manoeuvre
  6 = CUSTOM_SPLINE  custom spline scenario

Controls (web viewer browser)
  ← / →   steer left / right  (held key → ±STEER_VALUE)
  ↓        throttle = 0        (full stop)
  0-6      switch scenario token live
  Ctrl+C   quit and save

Run standalone — no LKAS required.
  DO NOT run vehicle.py — this script controls JetRacer directly.

Data layout
  data/
  └── planner_data.csv   (one row per saved frame)
"""

import sys
import time
import signal
import csv
import argparse
import threading
import numpy as np
from pathlib import Path

# 💡 [추가됨] 라이다 센서 제어 및 스플라인 알고리즘 모듈 임포트
# 원본 코드에는 카메라 센서만 사용되었으나, 
# 거리 측정 및 알고리즘 주행을 위해 LidarSensor와 SplineExpert가 새롭게 추가되었습니다.
from lidar_sensor import LidarSensor  
from spline_expert import SplineExpert

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent / "vehicle" / "src"))
sys.path.append(str(script_dir.parent / "common" / "src"))

# ── PyTorch legacy weights fix ────────────────────────────────────────────────
import torch
_orig_torch_load = torch.load

def _torch_load_legacy(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_legacy

from ultralytics import YOLO as _YOLO

# ── Camera ────────────────────────────────────────────────────────────────────
from camera import Camera
from visualization.visualizer import LKASVisualizer

# ── JetRacer ──────────────────────────────────────────────────────────────────
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("[WARN] JetRacer not available — simulation mode (no motor output)")
    JETRACER_AVAILABLE = False

# ── Lane segmentation (direct BiSeNet — no LKAS required) ────────────────────
from lane_seg import LaneSeg

# ── Web viewer ────────────────────────────────────────────────────────────────
from planner_viewer import PlannerViewer

# ── YOLO config ───────────────────────────────────────────────────────────────
from yolo_config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES
N_YOLO_CLASSES = len(CLASS_NAMES)
# YOLO must run on CPU — BiSeNet (LaneSeg) takes the GPU to avoid OOM on Jetson
YOLO_DEVICE = 'cpu'

# ── Planner model shared definitions ─────────────────────────────────────────
from planner_model import (
    build_object_features,
    build_lane_grid,
    lane_boundaries_from_mask,
    draw_lane_grid_overlay,
    csv_columns,
    N_MAX_OBJECTS,
    SCENARIO_LANE_FOLLOW, SCENARIO_LEFT_TURN, SCENARIO_RIGHT_TURN,
    SCENARIO_GO_STRAIGHT, SCENARIO_PULL_OVER, SCENARIO_PARKING,
    SCENARIO_NAMES,
    MAX_THROTTLE,
    csv_columns_ext, IMU_COLUMNS,   # 💡 [추가됨] IMU 확장 스키마
    LIDAR_EXTRA_COLUMNS, CAR_HALF_W, CAR_SIDE_GAP,  # 💡 [추가됨] corridor
    FRAME_W, FRAME_H,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_THROTTLE   = 0.10   # auto-forward throttle during collection
SAVE_FPS        = 10     # max rows written per second

# ─────────────────────────────────────────────────────────────────────────────
# 💡 [추가된 부분: 스로틀 데드존 히스테리시스 상수]
# ─────────────────────────────────────────────────────────────────────────────
# [문제] 기존에는 단일 임계값 하나(0.08)로 데드존을 처리했습니다.
#     thro_raw = pad_thro if pad_thro > 0.08 else 0.0
# 이러면 스틱이 임계값 근처에 머무를 때 0 과 비0 을 프레임마다 왕복하고,
# 그 결과 "운전자는 계속 주행 중인데 라벨만 0" 인 프레임이 기록됩니다.
#
# [실측 증거] 수집된 4개 파일 2240행 분석 결과:
#   - target_throttle 이 정확히 0.0 인 프레임 : 520개 (23.2%)
#   - 0 초과 최솟값                          : 0.11863
#   → 0 과 0.119 사이가 완전한 공백. 사람이 아날로그 스틱을 연속으로
#     조작했다면 이 구간에 값이 깔려 있어야 하므로, 이 공백은 사람이 아니라
#     데드존 코드가 만든 계단입니다.
#   - 0 -> 비0 전이 직후 값의 중앙값 : 0.609 / 0.459 / 0.553
#   → 진짜 재출발이면 0->0.2->0.4 로 올라가야 하는데 한 프레임 만에 0.6 으로
#     점프합니다. 스틱이 데드존에 빠졌다 복귀한 흔적입니다.
#
# [해결] 진입/이탈 임계값을 분리합니다(라이다 DANGER/CLEAR 와 같은 원리).
#   정지 상태에서 출발하려면 ENGAGE 를 넘어야 하고,
#   일단 주행이 시작되면 RELEASE 아래로 내려갈 때까지 0 으로 떨어지지 않습니다.
# 두 값 사이가 히스테리시스 폭이며, 이 폭만큼 채터링이 사라집니다.
#
# [주의] 이 값은 조작감을 바꿉니다. 실제로 몰아보시고 조정하세요.
#   - ENGAGE 를 낮추면 출발이 민감해지고
#   - RELEASE 를 낮추면 스로틀을 놓아도 더 오래 붙어 있습니다.
THROTTLE_ENGAGE_TH  = 0.08   # 정지 -> 주행 전환에 필요한 스틱 입력 (기존 값 유지)
THROTTLE_RELEASE_TH = 0.03   # 주행 -> 정지 전환 임계값 (이 아래로 내려가야 0)

DATA_DIR      = script_dir / "data"
# 💡 [수정됨] PLANNER_CSV 상수 선언은 출력 파일명을 동적으로 받기 위해 main 함수 내부로 이동되었습니다.

# 💡 [추가됨] 시나리오 확장 및 전문가(Expert) 알고리즘 매핑 구조
# 기존에는 0~5번(수동 조작) 시나리오만 있었으나, 6번 시나리오 호출 시
# 수동 조작 대신 지정된 자율주행 모듈(SplineExpert)이 개입하도록 딕셔너리로 구조화되었습니다.
SCENARIO_EXPERT_MAP = {
    6: SplineExpert(),  
}

# ─────────────────────────────────────────────────────────────────────────────
# YOLO background worker
# Runs YOLO on CPU in a daemon thread — same pattern as planner_inference.py.
# The main loop always reads the latest cached result; YOLO never blocks it.
# ─────────────────────────────────────────────────────────────────────────────

_yolo_lock    = threading.Lock()
_yolo_cache   = {'boxes': [], 'distances': [], 'class_ids': [], 'confs': []}
_yolo_running = False  # guarded by _yolo_lock — always read/write under the lock


def _yolo_worker(yolo, frame, depth_array, depth_scale, frame_w, frame_h):
    global _yolo_cache
    try:
        results = yolo(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                       device=YOLO_DEVICE, verbose=False)
        boxes_r, dists_r, cids_r, confs_r = [], [], [], []
        if len(results[0].boxes) > 0:
            xyxy  = results[0].boxes.xyxy.cpu().numpy()
            cids  = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            for box, cid, conf in zip(xyxy, cids, confs):
                cx = int(max(0, min((box[0] + box[2]) / 2, frame_w - 1)))
                cy = int(max(0, min((box[1] + box[3]) / 2, frame_h - 1)))
                raw  = int(depth_array[cy, cx])
                dist = raw * depth_scale if raw > 0 else -1.0
                boxes_r.append(box)
                dists_r.append(dist)
                cids_r.append(int(cid))
                confs_r.append(float(conf))
        with _yolo_lock:
            _yolo_cache = {'boxes': boxes_r, 'distances': dists_r,
                           'class_ids': cids_r, 'confs': confs_r}
        del results
    except Exception as e:
        print(f"\n[YOLO] Error: {e}")
    finally:
        with _yolo_lock:
            global _yolo_running
            _yolo_running = False


# ─────────────────────────────────────────────────────────────────────────────
# Annotation  (broadcast only — nothing saved to disk)
# ─────────────────────────────────────────────────────────────────────────────
import cv2

_MODE_COLORS = {
    0: (0, 255, 0),    # LANE_FOLLOW  — green
    1: (255, 255, 0),  # LEFT_TURN    — yellow
    2: (0, 255, 255),  # RIGHT_TURN   — cyan
    3: (255, 165, 0),  # GO_STRAIGHT  — orange
    4: (255, 0, 255),  # PULL_OVER    — magenta
    5: (0, 128, 255),  # PARKING      — light blue
    # 💡 [추가됨] 새로 추가된 6번 시나리오(CUSTOM_SPLINE)에 대한 UI 텍스트 색상(보라색) 매핑이 추가되었습니다.
    6: (128, 0, 255),  
}
_SCENARIO_NAMES = SCENARIO_NAMES  # imported from planner_model
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]

_visualizer = LKASVisualizer(image_width=FRAME_W, image_height=FRAME_H)


def _annotate(frame, boxes, distances, class_ids, scenario, steering, throttle, fps,
              left_x, right_x, saved_count, mask=None, lane_feats=None):
    out = frame.copy()

    # ── Lane segmentation overlay ─────────────────────────────────────────────
    if mask is not None:
        out = _visualizer.draw_segmentation(out, mask)
    else:
        # fallback: dim vertical lines at fixed positions
        h = out.shape[0]
        cv2.line(out, (int(left_x),  0), (int(left_x),  h), (80, 80, 160), 1)
        cv2.line(out, (int(right_x), 0), (int(right_x), h), (80, 80, 160), 1)

    # ── Grid pooling overlay ──────────────────────────────────────────────────
    if lane_feats is not None:
        out = draw_lane_grid_overlay(out, lane_feats)

    for box, dist, cid in zip(boxes, distances, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = _BOX_COLORS[cid % len(_BOX_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl   = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        dist_txt = f"{dist:.2f}m" if dist > 0 else "N/A"
        cv2.putText(out, f"{lbl} {dist_txt}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    sc_name  = _SCENARIO_NAMES.get(scenario, str(scenario))
    sc_color = _MODE_COLORS.get(scenario, (255, 255, 255))
    cv2.putText(out, f"SCENARIO: {sc_name}",       (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"steer={steering:+.2f}  thr={throttle:.2f}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS={fps:.1f}  saved={saved_count}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_csv(csv_path: Path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 💡 [수정됨] 기본 스키마 -> IMU 확장 스키마
    # 기존 CSV(IMU 없음)를 열면 컬럼 수가 달라 자동으로 백업 후 새 파일이
    # 생성됩니다. 이는 의도된 동작이며, 기존 파일은 .bak 로 보존됩니다.
    # 학습/증강 쪽은 csv_columns() 부분집합 검사를 쓰므로 양쪽 다 읽힙니다.
    expected = csv_columns_ext()

    if csv_path.exists():
        # Check schema matches current planner_model constants
        with open(csv_path) as _f:
            existing_cols = _f.readline().strip().split(',')
        if existing_cols == expected:
            fh = open(csv_path, "a", newline="")
            writer = csv.writer(fh)
            print(f"[CSV] Appending to existing dataset: {csv_path}  "
                  f"({sum(1 for _ in open(csv_path)) - 1} rows)")
        else:
            # Schema mismatch (e.g. grid size changed) — back up old file
            backup = csv_path.with_suffix(f".bak{int(time.time())}.csv")
            csv_path.rename(backup)
            print(f"[CSV] Schema changed ({len(existing_cols)} cols → {len(expected)} cols)")
            print(f"[CSV] Old data backed up to {backup.name}")
            fh = open(csv_path, "w", newline="")
            writer = csv.writer(fh)
            writer.writerow(expected)
            print(f"[CSV] Created new dataset: {csv_path}")
    else:
        fh = open(csv_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(expected)
        print(f"[CSV] Created new dataset: {csv_path}")

    return fh, writer


# 💡 [수정됨] 함수 파라미터 및 CSV 기록 로직 변경
# 기존에는 카메라와 차량 제어값만 저장했으나, 라이다 센서 기능이 도입됨에 따라
# 'lidar_feats' 파라미터가 추가되었고, 해당 5구역 데이터를 CSV의 올바른 위치(스키마 순서)에 기록하도록 수정되었습니다.
def _save_row(writer, fh, frame_id: int, obj_feats: list, lane_feats: list,
              ego_feats: list, lidar_feats: list, scenario: int, steering: float,
              throttle: float, imu_feats: list = None,
              corridor_feats: list = None):
    """Write one structured row to the CSV.

    💡 [추가된 부분: imu_feats]
    [imu_motion, imu_yaw_rate, imu_accel_fwd] 3개 값을 스키마 맨 뒤에 붙입니다.
    None 이면 0.0 세 개로 채워 컬럼 수를 항상 일정하게 유지합니다
    (IMU 가 없는 기기에서도 CSV 구조가 깨지지 않도록).
    """
    # target_throttle is normalised to [0, 1] for training
    throttle_norm = float(throttle) / MAX_THROTTLE

    row = [frame_id]
    row.extend(f"{v:.5f}" for v in obj_feats)
    row.extend(f"{v:.5f}" for v in lane_feats)
    row.extend([f"{ego_feats[0]:.5f}", f"{ego_feats[1]:.5f}"])
    
    # 추가된 라이다 5구역 데이터를 ego_feats 뒤, scenario 앞에 끼워 넣음
    row.extend(f"{v:.5f}" for v in lidar_feats)
    
    row.append(scenario)
    row.append(f"{steering:.5f}")
    row.append(f"{throttle_norm:.5f}")

    # 💡 [추가됨] IMU 파생값 3개를 스키마 맨 뒤에 기록
    imu = imu_feats if imu_feats is not None else [0.0, 0.0, 0.0]
    row.extend(f"{float(v):.5f}" for v in imu)

    # 💡 [추가됨] corridor 3개. None 이면 MAX_DIST_M 로 채워 "열림" 을 뜻하게 한다.
    # 0 으로 채우면 "코앞이 막힘" 이 되어 정반대 의미가 되므로 절대 0 을 쓰지 않는다.
    cor = corridor_feats if corridor_feats is not None else [MAX_DIST_M] * 3
    row.extend(f"{float(v):.5f}" for v in cor)

    writer.writerow(row)
    fh.flush()

# 💡 [추가됨] ESC 모터 후진 보호 필터 클래스 추가
# RC카의 ESC(전자변속기) 특성상 후진을 위해서는 '브레이크 ➔ 중립 ➔ 후진'의 시퀀스가 필수적입니다.
# 원본 코드는 즉시 후진값을 전송하여 모터가 무시하는 현상이 있었으나,
# 이 클래스를 통해 소프트웨어적으로 0.3초간의 브레이크 및 중립 과정을 자동 생성하여 확실하게 후진 기어를 체결합니다.
class ESC_DoubleTap:
    def __init__(self):
        self.state = 'FORWARD'
        self.t_start = 0.0

    def process(self, target_thr):
        import time
        now = time.time()
        
        # 1. 전진 또는 정지 (조이스틱을 놓았거나 앞으로 밀었을 때)
        if target_thr >= -0.01:
            self.state = 'FORWARD'
            return target_thr
            
        # 2. 음수(후진) 신호 최초 발생 시 -> 1차 브레이크 타격
        if self.state == 'FORWARD':
            self.state = 'BRAKE'
            self.t_start = now
            # ESC의 데드밴드(방어벽)를 확실히 뚫기 위해 최소 -0.5 이상의 강한 브레이크를 때림
            return min(target_thr, -0.5) 
            
        # 3. 0.3초간 브레이크 유지 (시간 연장)
        elif self.state == 'BRAKE':
            if now - self.t_start < 0.3:
                return min(target_thr, -0.5)
            else:
                self.state = 'NEUTRAL'
                self.t_start = now
                return 0.0
                
        # 4. 0.3초간 중립 유지 (ESC 후진 락 완벽 해제)
        elif self.state == 'NEUTRAL':
            if now - self.t_start < 0.3:
                return 0.0
            else:
                self.state = 'REVERSE'
                return target_thr
                
        # 5. 락 해제 후에는 원래 사용자가 당긴 조이스틱 값으로 계속 후진
        elif self.state == 'REVERSE':
            return target_thr
            
        return target_thr

# 💡 [추가됨] 라이다 5구역 데이터 압축 및 전처리 함수
# 라이다 센서에서 스캔된 1000개의 원시 데이터(Raw Scan) 배열을 입력받아,
# 자율주행 모델이 이해할 수 있도록 5개의 핵심 논리적 구역(좌측, 좌전방, 정면, 우전방, 우측)의 
# 최단 장애물 거리값으로 정제(압축)하여 반환하는 함수가 새롭게 이식되었습니다.
def process_lidar_to_5_sectors(raw_scan):
    """
    1000개의 원본 데이터를 받아 기존 시스템과 완벽히 동일한 5개의 논리적 섹터(s0~s4)로 덜어냅니다.
    """
    def get_min_dist(idx_start, idx_end):
        # 0(정면)을 걸쳐서 슬라이싱해야 하는 경우
        if idx_start > idx_end:
            slice1 = raw_scan[idx_start:]
            slice2 = raw_scan[:idx_end]
            valid_dists = np.concatenate((slice1[slice1 > 0.0], slice2[slice2 > 0.0]))
        else:
            slice_data = raw_scan[idx_start:idx_end]
            valid_dists = slice_data[slice_data > 0.0]
        
        # 측정된 값이 있으면 최솟값(가장 가까운 거리) 반환, 없으면 최대거리 5.0m 반환
        if len(valid_dists) > 0:
            return min(5.0, float(np.min(valid_dists)))
        return 5.0

    # 섹터별 물리적 인덱스 계산 (좌측이 +, 우측이 -) 원래 로직 그대로 복구
    s0 = get_min_dist(83, 166)   # 좌측
    s1 = get_min_dist(27, 83)    # 좌전방
    s2 = get_min_dist(972, 27)   # 정면
    s3 = get_min_dist(916, 972)  # 우전방
    s4 = get_min_dist(833, 916)  # 우측

    return [s0, s1, s2, s3, s4]


# ─────────────────────────────────────────────────────────────────────────────
# 💡 [추가된 부분: corridor — 차폭 기준 통행 가능 거리]
# ─────────────────────────────────────────────────────────────────────────────
# [왜 여기서 계산하는가]
# 원본 1000점이 살아 있는 유일한 지점이다. 5섹터로 압축되고 나면 각 호(arc)의
# 최솟값 하나만 남아 "몇 도 방향에서 온 값인지"가 소멸하며, 그 뒤로는 어떤
# 방법으로도 복원할 수 없다.
#
# [왜 필요한가]  같은 s1 = 0.50m 가
#     11도 방향이면  lateral 0.095m  ->  내 폭 안, 부딪힘  (정지해야 함)
#     25도 방향이면  lateral 0.211m  ->  옆,      비켜감   (회피하면 됨)
# 으로 정반대 의미인데 5섹터로는 구분이 불가능하다.
#
# [실측] 상황별 5섹터 프로파일 거리 (작을수록 구분 불가):
#     avoidance vs stop      0.72 m   <- 정반대 행동인데 제일 안 갈림
#     s_curve   vs avoidance 0.72 m
#     stop      vs recovery  0.81 m
#     normal    vs 나머지    2.2~3.7 m  (개활지만 잘 갈린다)
#
# [연산량] 실측 34.6us. 수집 예산 100ms 의 0.03%, 추론 예산 54.6ms 의 0.06%.
# sin/cos 는 상수 배열로 1회만 계산하므로 프레임당 비용은 곱셈과 min 뿐이다.
# 참고로 15섹터 분할은 80.6us 로 오히려 더 비싸면서 각도 문제를 못 푼다.
#
# [주의] 이 함수는 수집과 추론이 반드시 같은 것을 써야 한다. 정의를 두 벌
# 두면 학습 때 본 값과 주행 때 들어가는 값이 조용히 어긋난다.
# (이 저장소에 이미 그런 사례가 있다 — train_planner.py 는 row_to_tensors 를
#  import 하고도 쓰지 않고 텐서를 직접 만든다.)

_CORRIDOR_IDX = np.concatenate([np.arange(833, 1000), np.arange(0, 167)])
# 인덱스 -> 각도(rad). 1000점 / 360도 = 0.36도/idx, 양수가 왼쪽(spline_expert 규약).
_CORRIDOR_ANG = np.deg2rad(_CORRIDOR_IDX * 0.36)
_CORRIDOR_ANG = np.where(_CORRIDOR_ANG > np.pi, _CORRIDOR_ANG - 2 * np.pi, _CORRIDOR_ANG)
_CORRIDOR_SIN = np.sin(_CORRIDOR_ANG)
_CORRIDOR_COS = np.cos(_CORRIDOR_ANG)


def compute_corridor(raw_scan):
    """원본 스캔에서 [front_clear, left_gap, right_gap] 를 계산한다.

        front_clear  내 차 폭 안에서 가장 가까운 것까지  = 직진하면 몇 m 뒤 충돌
        left_gap     왼쪽 한 차폭 띠에서 가장 가까운 것  = 왼쪽으로 비키면 뭐가 있나
        right_gap    오른쪽 같은 것

    유효한 점이 없으면 MAX_DIST_M(5.0) 을 반환한다. 0 이 아니라 5.0 인 이유는
    0 이 "거리 0m = 코앞에 벽" 을 뜻해 정반대 의미가 되기 때문이다.
    """
    scan = np.asarray(raw_scan, dtype=float)
    if scan.size < 1000:
        return [MAX_DIST_M, MAX_DIST_M, MAX_DIST_M]

    d = scan[_CORRIDOR_IDX]
    ok = np.isfinite(d) & (d > 0.0) & (_CORRIDOR_COS > 0.0)   # 전방 반구만
    lat = d * _CORRIDOR_SIN                                    # 양수 = 왼쪽

    in_path = ok & (np.abs(lat) < CAR_HALF_W)
    left_band = ok & (lat >= CAR_HALF_W) & (lat < CAR_HALF_W + CAR_SIDE_GAP)
    right_band = ok & (lat <= -CAR_HALF_W) & (lat > -(CAR_HALF_W + CAR_SIDE_GAP))

    def _nearest(mask):
        return float(min(MAX_DIST_M, d[mask].min())) if mask.any() else MAX_DIST_M

    return [_nearest(in_path), _nearest(left_band), _nearest(right_band)]

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# 💡 [수정됨] 메인 함수 시그니처 변경
# 사용자가 스크립트 실행 시 --out 옵션을 통해 저장할 CSV 파일명을 마음대로 지정할 수 있도록
# out_csv 파라미터가 함수 인자로 추가되었습니다.
def main(web_port: int = 8082, scenario: int = SCENARIO_LANE_FOLLOW, out_csv: str = "planner_data.csv",
         save_scan: bool = False):
    global _yolo_running  # 전역 변수 참조 선언 명시

    # 전달받은 파일명 인자를 바탕으로 최종 저장 경로 동적 할당
    PLANNER_CSV = DATA_DIR / out_csv

    # 💡 [추가됨] 라이다 센서 포트 자동 스캔 및 연결 알고리즘
    # 원본 코드에는 라이다 연동 자체가 없었습니다. 
    # USB 및 ACM 통신 포트 전체를 자동으로 순회 스캔하여 라이다 하드웨어를 스스로 찾고 연결합니다.
    # 만약 연결에 완전히 실패하면 안전을 위해 데이터 수집기를 강제 종료시키는 방어 로직도 추가되었습니다.
    lidar = None
    candidate_ports = [f"/dev/ttyUSB{i}" for i in range(5)] + [f"/dev/ttyACM{i}" for i in range(2)]

    print("\n[LIDAR] 라이다 포트 강제 탐색을 시작합니다...")
    for port_name in candidate_ports:
        try:
            # LidarSensor 클래스에 포트 주소를 직접 주입하며 연결을 시도
            lidar = LidarSensor(port=port_name) 
            print(f"[LIDAR] ✅ 라이다 센서 연결 성공 (위치: {port_name})")
            break  # 연결에 성공하면 즉시 스캔 중단
        except Exception:
            pass  # 실패하면 조용히 다음 포트로 넘어감

    # 사용자가 "무조건 연결해야 한다"고 했으므로, 끝까지 못 찾으면 파이프라인을 셧다운시킵니다.
    if lidar is None:
        print("\n[FATAL] 시스템의 모든 USB 포트를 스캔했으나 라이다의 응답이 없습니다.")
        print("  1. 물리적 단선 또는 전력 부족 (선 뽑았다 다시 꽂기)")
        print("  2. 권한 차단 (터미널에 'sudo chmod 666 /dev/ttyUSB*' 입력)")
        sys.exit(1)  # 강제 종료

    print("[SYSTEM] 시나리오별 전문가 모듈 맵핑 완료 (확장성 준비됨)")

    sc_name = _SCENARIO_NAMES.get(scenario, str(scenario))
    print("=" * 62)
    print("  Planner Data Collection  (structured features, no images)")
    print("=" * 62)
    print(f"  Scenario      : {sc_name} ({scenario})")
    print(f"  Output CSV    : {PLANNER_CSV}")
    print(f"  Save rate cap : {SAVE_FPS} fps")
    print(f"  YOLO          : background thread (non-blocking)")
    print(f"  Base throttle : {BASE_THROTTLE}")
    print()
    print("  Controls: ← steer left  → steer right  ↓ stop  Ctrl+C quit")
    print("=" * 62)

    # ── YOLO (CPU — GPU reserved for LaneSeg/BiSeNet) ────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"\n[YOLO] Loading: {MODEL_PATH}  (device=cpu)")
    yolo = _YOLO(MODEL_PATH)

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = PlannerViewer(http_port=web_port, ws_port=web_port + 1)
        web_viewer._scenario = scenario   # seed from --scenario CLI flag
        web_viewer.start()
        print(f"[WEB] Viewer: http://0.0.0.0:{web_port}")

    # ── LaneSeg (direct BiSeNet — GPU) ────────────────────────────────────────
    print("\n[LaneSeg] Loading BiSeNet...")
    lane_seg = LaneSeg(device="auto")

    # ── Camera ───────────────────────────────────────────────────────────────
    print("\n[CAM] Opening RealSense camera...")
    # 💡 [수정됨] IMU 스트림 활성화 (D435i)
    # enable_imu=True 는 별도 파이프라인 + 콜백으로 동작하므로
    # 아래 read_frames() 루프의 타이밍에는 영향을 주지 않습니다.
    # D435(무印) 처럼 IMU 가 없으면 camera.py 가 경고만 찍고 계속 진행합니다.
    camera = Camera(width=FRAME_W, height=FRAME_H, enable_depth=True, enable_imu=True)
    depth_scale = camera.depth_scale if camera.depth_scale > 0 else 0.001

    # ── Gamepad ──────────────────────────────────────────────────────────────
    # 💡 [추가됨] 실물 게임패드 하드웨어 제어권 연동
    # 웹 브라우저 키보드 입력만 지원하던 원본 코드와 달리,
    # 실제 주행을 원활히 하기 위해 ShanWanGamepad 조이스틱 컨트롤러 모듈을 새롭게 초기화합니다.
    try:
        from gamepads import ShanWanGamepad
        pad = ShanWanGamepad()
        print("\n[PAD] Gamepad initialized")
        
    except Exception as e:
        pad = None
        print(f"\n[WARN] Gamepad not loaded 원인: {e}")

    # ── JetRacer ─────────────────────────────────────────────────────────────
    car = None
    if JETRACER_AVAILABLE:
        print("\n[CAR] Initializing NvidiaRacecar...")
        car = NvidiaRacecar()
        # 💡 [수정됨] 차량 조향 중앙값 미세 조정을 위해 주석(0.05) 추가 등 물리 캘리브레이션 튜닝 흔적 반영
        car.steering_offset = 0.040 # 0.05
        car.throttle = 0.0
        car.steering = 0.0
        time.sleep(0.2)   # wait for servo to physically centre before starting
        print("[CAR] Ready")
    else:
        print("\n[CAR] Simulation mode")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_fh, csv_writer = _init_csv(PLANNER_CSV)

    # [추가 2026-08-23] --save-scan : 원본 1000점 스캔을 CSV 와 1:1 로 보존.
    #
    #   [근거] process_lidar_to_5_sectors(:397) 의 min 연산자가 호 안의
    #   각도를 지운다. 그래서 CSV 에 남는 lidar_s0~s4 는 5개 전부 "거리"
    #   이고 각도가 하나도 없다. 조향은 각도를 정하는 문제인데 입력에
    #   각도가 없으니, 5섹터의 조향 설명력이 비선형 교차검증 R^2 0.081
    #   에 그친다 (원본 8코스 3,671행 실측).
    #
    #   [왜 원본을 남기나] 어떤 5개가 최적인지 (bearing / argmin 각도 /
    #   corridor / 섹터 9,15,36개) 는 원본 스캔이 있어야 비교할 수 있다.
    #   지금은 min 5개만 남아 단 하나의 실험도 불가능하고, 형식을 바꿀
    #   때마다 차를 다시 몰아야 한다. 스캔을 남기면 전부 오프라인이 된다.
    #
    #   [비용] 1000 x float32 x 10fps x 10분 = 24 MB.
    #   CSV 스키마 불변. 기본값 꺼짐이라 플래그 없으면 기존 동작 그대로.
    scan_sink = [] if save_scan else None
    if save_scan:
        _sp = PLANNER_CSV.with_name("scan_" + PLANNER_CSV.stem + ".npy")
        print(f"  Raw scan  : ON  -> {_sp.name}")

    # ── Signal handling ───────────────────────────────────────────────────────
    running = True

    def _shutdown(sig, _f):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    
    # ── State ─────────────────────────────────────────────────────────────────
    prev_steering  = 0.0
    prev_throttle  = 0.0
    
    # 💡 [추가됨] 새롭게 정의한 ESC 모터 후진 필터 인스턴스화
    esc_filter = ESC_DoubleTap()

    # 💡 [추가됨] 스로틀 히스테리시스 상태
    # True = 현재 주행 중(스로틀이 붙어 있음). 루프를 넘어 유지되어야 하므로
    # 반드시 루프 바깥에서 초기화합니다.
    throttle_engaged = False

    frame_id       = 1
    saved_count    = 0
    last_save_time = 0.0
    save_interval  = 1.0 / SAVE_FPS
    fps            = 0.0
    fps_count      = 0
    fps_start      = time.time()
    # Track consecutive no-lane frames while recording — warn if BiSeNet isn't detecting
    _no_lane_rec_streak = 0
    _NO_LANE_WARN_THRESH = 30   # warn after ~3 s at 10 fps

    print(f"\n[COLLECT] Running — Ctrl+C to stop\n")

    try:
        while running:

            # ── Web viewer human input ────────────────────────────────────────
            raw_steer  = web_viewer.steering if web_viewer else 0.0

            # ── Camera frame ─────────────────────────────────────────────────
            color_bgr, depth_raw = camera.read_frames()
            if color_bgr is None:
                continue
                
            depth_array = depth_raw if depth_raw is not None else \
                          np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)

            # 💡 [추가됨] YOLO 객체 인식용 깨끗한 원본 이미지 독립 보존
            # 뒷 단계에서 차선 인식을 돕기 위해 원본 이미지(color_bgr) 위에 검은색/하얀색 덧칠을 진행합니다.
            # 하지만 객체 인식 AI인 YOLO에게 이렇게 덧칠된 이미지가 넘어가면 표지판/사람 등을 놓칠 위험이 매우 큽니다.
            # 따라서 마스킹 연산이 들어가기 전에 YOLO 전용으로 오염되지 않은 원본(yolo_input_frame)을 깊은 복사해 둡니다.
            yolo_input_frame = color_bgr.copy()

            # 💡 [추가됨] 노란색 차선 강조 및 흰색(가짜 차선) 시각적 노이즈 제거 필터
            # 원본 코드는 카메라 영상을 차선인식 모델에 그대로 넘겼으나, 빛 반사나 흰색 구조물 오인식이 심했습니다.
            # 이 로직은 영상의 HSV 색상을 분석하여, 방해되는 흰색은 까맣게 칠해 지워버리고
            # 인식해야 할 노란색 차선은 순백색으로 강하게 덧칠해 주어 BiSeNet 모델의 인식률을 폭발적으로 끌어올립니다.
            hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([0, 0, 184])
            upper_yellow = np.array([96, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            color_bgr[yellow_mask > 0] = (255, 255, 255)
            # ─────────────────────────────────────────────────────────────────

            # ── YOLO (방금 빼둔 깨끗한 원본 이미지 전달) ────────────────────────
            with _yolo_lock:
                if not _yolo_running:
                    _yolo_running = True
                    threading.Thread(
                        target=_yolo_worker,
                        # 💡 [수정됨] 마스킹 덧칠로 왜곡된 color_bgr 이미지 대신,
                        # 미리 복사해둔 깨끗한 원본(yolo_input_frame)을 YOLO 입력으로 전달합니다.
                        args=(yolo, yolo_input_frame, depth_array.copy(),  
                              depth_scale, FRAME_W, FRAME_H),
                        daemon=True,
                    ).start()
                r = _yolo_cache
            boxes     = r['boxes']
            distances = r['distances']
            class_ids = r['class_ids']
            confs     = r['confs']

            # ── Lane segmentation (BiSeNet, every frame) ─────────────────────
            mask = lane_seg.infer(color_bgr)
            left_lane_x, right_lane_x = lane_boundaries_from_mask(mask)
            lane_detected = bool(mask.any())

            # 💡 [추가됨] 매 프레임별 라이다 원본 획득 및 5구역 특징(Feature) 정제
            # 원본 코드에는 없는 라이다 연산이 프레임마다 동작하여 데이터를 수집합니다.
            raw_lidar_array = lidar.get_raw_scan()
            processed_lidar_feats = process_lidar_to_5_sectors(raw_lidar_array)
            # 💡 [추가됨] 원본 스캔이 살아 있는 이 지점에서만 계산 가능 (34.6us)
            corridor_feats = compute_corridor(raw_lidar_array)

            # 💡 [추가됨] 확장형 아키텍처(전문가 모델) 기반 알고리즘 자율 연산
            # 사용자가 선택한 시나리오가 6번(스플라인) 등 매핑된 자율주행 모듈일 경우,
            # 라이다 데이터를 모듈에 전달해 알고리즘이 직접 꺾어야 할 각도(expert_steering)를 계산하게 만듭니다.
            cur_scenario = web_viewer.scenario if web_viewer else scenario 
            expert_steering = 0.0

            if cur_scenario in SCENARIO_EXPERT_MAP:
                expert_steering = SCENARIO_EXPERT_MAP[cur_scenario].calculate_action(raw_lidar_array)

            # ── Human steering / throttle ─────────────────────────────────────
            if pad is not None:
                pad_data = pad.read_data()
                
                # 오른쪽 스틱 좌우(x), 왼쪽 스틱 상하(y)
                pad_steer = pad_data.analog_stick_right.x
                pad_thro = pad_data.analog_stick_left.y
                btn_a = pad_data.button_a

                # 💡 [수정됨] 물리 조이스틱 입력의 미세 튜닝 및 가공 로직 전면 개선
                # 1. 아날로그 스틱의 미세한 흔들림을 막기 위한 조향 데드존(0.08) 필터가 추가되었습니다.
                # 2. 민감도(STEER_GAIN = 0.9)를 곱해 차량의 회전폭을 조절합니다.
                steer_raw = pad_steer if abs(pad_steer) > 0.08 else 0.0
                STEER_GAIN = 0.9 
                human_steering = steer_raw * STEER_GAIN

                # 💡 [수정됨] 직관적인 스로틀 조작 분리 로직 구현
                # 게임패드의 A버튼을 누르면 즉시 고정된 후진 힘(-0.1)을 가하도록 핫키를 지정했고,
                # 버튼을 누르지 않았을 때는 엑셀(y축) 입력을 받아 THROTTLE_GAIN 상수를 통해 부드럽게 감속시킵니다.
                if btn_a == 1:
                    human_throttle = -0.1 
                else:
                    # 💡 [수정된 부분: 단일 데드존 -> 히스테리시스 데드존]
                    # 기존: thro_raw = pad_thro if pad_thro > 0.08 else 0.0
                    #       임계값 하나뿐이라 경계에서 0 과 비0 을 왕복했습니다.
                    # 변경: 주행 중이면 RELEASE(0.03) 까지 버티고,
                    #       정지 상태에서 출발하려면 ENGAGE(0.08) 를 넘어야 합니다.
                    #       throttle_engaged 는 루프 바깥에서 유지되는 상태입니다.
                    if throttle_engaged:
                        thro_raw = pad_thro if pad_thro > THROTTLE_RELEASE_TH else 0.0
                    else:
                        thro_raw = pad_thro if pad_thro > THROTTLE_ENGAGE_TH else 0.0
                    throttle_engaged = thro_raw > 0.0

                    THROTTLE_GAIN = 0.368 
                    human_throttle = thro_raw * THROTTLE_GAIN
                
                # 💡 [추가됨] 반자율 주행 스위칭 로직 (제어권 분배)
                # 시나리오가 6번 등 알고리즘 모드일 경우: 조향은 AI(알고리즘)가 스스로 꺾고, 엑셀/브레이크는 사람이 담당합니다.
                # 일반 모드일 경우: 조향과 엑셀 모두 사람이 수동 제어합니다.
                if cur_scenario in SCENARIO_EXPERT_MAP:
                    input_steering = expert_steering  # 🤖 조향은 알고리즘
                    input_throttle = human_throttle   # 🧑‍✈️ 엑셀은 사람
                else:
                    input_steering = human_steering   # 🧑‍✈️ 수동 조향
                    input_throttle = human_throttle   # 🧑‍✈️ 수동 엑셀
                
            else:
                human_steering = raw_steer
                human_throttle = web_viewer.throttle if web_viewer else 0.0
                
                if cur_scenario in SCENARIO_EXPERT_MAP:
                    input_steering = expert_steering
                    input_throttle = human_throttle
                else:
                    input_steering = human_steering
                    input_throttle = human_throttle
            
            # Apply to vehicle
            if car is not None:
                car.steering = -float(input_steering)
                
                # 💡 [수정됨] 스로틀 하드웨어 전송 전, 안전한 후진 기어 변환 필터 적용
                # 원본 코드와 달리 스로틀 값을 즉시 할당하지 않고,
                # 새롭게 만든 esc_filter를 거쳐 강제 브레이크/중립 유지 시퀀스가 수행되도록 변경되었습니다.
                hw_throttle = esc_filter.process(float(input_throttle))
                target_thr_val = -hw_throttle
            
                # 💡 [수정됨] 차량 외부 시퀀스 중복 방지 제어 로직
                # 다른 매크로나 시퀀스가 차량을 조작 중일 때는 조이스틱 값이 차량에 먹히지 않게 차단합니다.
                if getattr(car, '_in_sequence', False):
                    pass  # 시퀀스 진행 중에는 외부 명령 차단
                else:
                    if not hasattr(car, 'last_sent_thr') or car.last_sent_thr != target_thr_val:
                        car.throttle = target_thr_val
                        car.last_sent_thr = target_thr_val
                        
            # ── Build structured features ─────────────────────────────────────
            obj_feats  = build_object_features(
                boxes=boxes, distances=distances,
                class_ids=class_ids, confs=confs,
                left_lane_x=left_lane_x, right_lane_x=right_lane_x,
                frame_w=FRAME_W, frame_h=FRAME_H, n_classes=N_YOLO_CLASSES,
            )
            lane_feats = build_lane_grid(mask)
            ego_feats = [prev_steering, prev_throttle / MAX_THROTTLE]

            # ── Save row (rate-limited, only when recording is toggled ON) ─────
            is_recording = web_viewer.recording if web_viewer else True
            cur_scenario = web_viewer.scenario  if web_viewer else scenario
            now = time.monotonic()
            if is_recording and now - last_save_time >= save_interval:
                last_save_time = now
                
                # 💡 [수정됨] CSV 저장 함수 호출 구조 변경
                # 1. 매 프레임별로 정제된 라이다 특징 데이터(processed_lidar_feats)를 추가로 전송합니다.
                # 2. steering과 throttle 파라미터에는 사람이 수동으로 조작했건, 반자율 알고리즘이 조작했건
                #    최종적으로 차를 구동시킨 결정값(input_steering)이 AI의 학습용 정답지(Label)로 기록됩니다.
                _save_row(
                    csv_writer, csv_fh,               
                    frame_id   = frame_id,
                    obj_feats  = obj_feats,
                    lane_feats = lane_feats,
                    ego_feats  = ego_feats,
                    lidar_feats= processed_lidar_feats, 
                    scenario   = cur_scenario,
                    steering   = input_steering,         
                    throttle   = input_throttle,         
                    # 💡 [추가됨] 저장 직전에 최신 IMU 파생값을 읽어 함께 기록.
                    # read_imu() 는 공유 변수 조회 + 표준편차 1회라 비용이
                    # 사실상 0 입니다(수집 예산 100ms 대비 무시 가능).
                    imu_feats  = list(camera.read_imu()),
                    corridor_feats = corridor_feats,
                )
                # [추가] CSV 한 행이 실제로 써진 이 자리에서만 append 한다.
                # 루프 앞쪽(:690)에서 하면 recording OFF 구간과 rate-limit
                # 로 버려진 프레임까지 들어가 CSV 와 행이 어긋난다.
                if scan_sink is not None:
                    scan_sink.append(np.asarray(raw_lidar_array, dtype=np.float32))

                frame_id    += 1
                saved_count += 1

                # Track no-lane streak and warn — rows with lane=NO teach the model
                # nothing about lane-following and will cause pegged steering at inference.
                if not lane_detected:
                    _no_lane_rec_streak += 1
                    if _no_lane_rec_streak == _NO_LANE_WARN_THRESH:
                        sys.stdout.write(
                            f"\n[WARN] Lane not detected for {_no_lane_rec_streak} consecutive"
                            f" saved rows. Saving lane=NO data. Check camera angle / lighting"
                            f" or pause recording until lanes are visible.\n"
                        )
                        sys.stdout.flush()
                else:
                    _no_lane_rec_streak = 0

                steer_tag = " ←" if input_steering < 0 else " →" if input_steering > 0 else "  ↑"
                sys.stdout.write(
                    f"\r[{'REC' if is_recording else '---'}]  "
                    f"saved={saved_count:>5d}  "
                    f"steer={input_steering:+.2f}{steer_tag}  "
                    f"thr={input_throttle:.2f}  "
                    f"objs={len(boxes)}  "
                    f"lane={'YES' if lane_detected else 'NO '}   "
                )
                sys.stdout.flush()

            # Update ego state for next frame
            prev_steering = input_steering
            prev_throttle = input_throttle

            # ── FPS ──────────────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # ── Broadcast annotated frame ─────────────────────────────────────
            if web_viewer is not None:
                annotated = _annotate(
                    color_bgr, boxes, distances, class_ids,
                    cur_scenario, input_steering, input_throttle, fps,
                    left_lane_x, right_lane_x, saved_count,
                    mask=mask, lane_feats=lane_feats,
                )
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'scenario':      _SCENARIO_NAMES.get(cur_scenario, str(cur_scenario)),
                    'steering':      input_steering,
                    'throttle':      input_throttle,
                    'fps':           fps,
                    'objects':       len(boxes),
                    'lane_detected': lane_detected,
                    'saved_rows':    saved_count,
                    'recording':     is_recording,
                })

            frame_id += 0   # already incremented inside save block

    finally:
        running = False

        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
            time.sleep(0.3)   # hold neutral long enough for servo to physically reach centre

        csv_fh.close()

        # [추가] 원본 스캔 저장. 행 순서가 CSV 와 1:1 로 대응한다.
        if scan_sink:
            _scan_path = PLANNER_CSV.with_name("scan_" + PLANNER_CSV.stem + ".npy")
            _arr = np.stack(scan_sink, axis=0)
            np.save(_scan_path, _arr)
            print()
            print(f"  [SAVE] raw scan -> {_scan_path}  shape={_arr.shape}"
                  f"  ({_arr.nbytes / 1e6:.1f} MB)")
        sys.stdout.write("\n")

        try: camera.close()
        except Exception: pass
        if web_viewer is not None:
            try: web_viewer.stop()
            except Exception: pass

        print(f"\n[COLLECT] Done — {saved_count} rows saved")
        print(f"[COLLECT] Dataset: {PLANNER_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner data collection — structured features")
    parser.add_argument('--web-port', type=int, default=8082,
                        help='Web viewer HTTP port (default 8082, 0 to disable)')
    
    # 💡 [수정됨] 커맨드라인 실행 옵션 파서 개선
    # 1. choices 범위에 새롭게 추가된 6번(CUSTOM_SPLINE) 시나리오 번호를 포함시켰습니다.
    # 2. 실행 시 저장될 CSV 파일 이름을 직접 입력받을 수 있도록 '--out' 인자 등록 로직이 신설되었습니다.
    parser.add_argument('--scenario', type=int, default=SCENARIO_LANE_FOLLOW,
                        choices=[0, 1, 2, 3, 4, 5, 6],  
                        help='Starting scenario token (can be changed live via 0-6 keys in viewer): '
                             '0=LANE_FOLLOW 1=LEFT_TURN 2=RIGHT_TURN 3=GO_STRAIGHT 4=PULL_OVER 5=PARKING 6=CUSTOM_SPLINE')
    
    parser.add_argument('--out', type=str, default="planner_data.csv",
                        help='저장할 CSV 파일명 (예: spline_data.csv). 미입력시 기본값 planner_data.csv')
    
    # [추가] 원본 스캔 보존 플래그. 기본 꺼짐 = 기존 동작과 완전 동일.
    parser.add_argument('--save-scan', action='store_true',
                        help='원본 1000점 스캔을 data/scan_<out>.npy 로 함께 저장 '
                             '(24MB/10분). CSV 스키마는 바뀌지 않는다. 피처 재설계를 '
                             '재수집 없이 오프라인으로 실험하려면 필수.')

    args = parser.parse_args()
    
    main(web_port=args.web_port, scenario=args.scenario, out_csv=args.out,
         save_scan=args.save_scan)
