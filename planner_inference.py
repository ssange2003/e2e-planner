#!/usr/bin/env python3
"""
Planner Inference  —  Structured Feature → Actuation
=====================================================
Real-time inference loop using the trained PlannerModel.

The planner receives NO camera pixels. It only sees:
  • YOLO object list    (class, distance, position, size, lane overlap)
  • LKAS lane data      (boundaries, centre offset, width)
  • Ego state           (previous steering / throttle)
  • Scenario token      (set via --scenario flag or web UI)

and outputs [steering, throttle] to the JetRacer and control SHM.

Pipeline per frame
------------------
  Camera  →  YOLO    → object features  ─┐
  Camera  →  LaneSeg → lane grid (32)   ─┤ → PlannerModel → [steer, thr]
  ego state  (from last cycle)          ─┘
                                               ↓
                                       control SHM (optional) + JetRacer

Run standalone — no LKAS required.
  vehicle.py    (reads control SHM)  ← optional; OR use --motor for direct drive

Usage
-----
  python planner_inference.py [--web-port 8082] [--motor] [--scenario 0]
                              [--model planner_model.pth]
"""

import sys
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from lidar_sensor import LidarSensor

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent / "vehicle" / "src"))
sys.path.append(str(script_dir.parent / "common" / "src"))

class SmartReverseFilter:
    def __init__(self):
        pass

    def process(self, throttle: float) -> float:
        # 입력된 스로틀 값을 그대로 통과시키거나 후진 제어 로직 수행
        return throttle

# ── PyTorch legacy weights fix ────────────────────────────────────────────────
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
    print("[WARN] JetRacer not available — simulation mode")
    JETRACER_AVAILABLE = False

# ── Lane segmentation (direct BiSeNet — no LKAS required) ────────────────────
from lane_seg import LaneSeg

# ── Control SHM (optional — only needed when vehicle.py reads planner output) ─
try:
    from lkas.integration.shared_memory import SharedMemoryControlChannel
    from lkas.integration.shared_memory.messages import ControlMessage
    LKAS_SHM_AVAILABLE = True
except ImportError:
    LKAS_SHM_AVAILABLE = False
    print("[WARN] LKAS SHM not importable — control SHM disabled")

# ── Web viewer ────────────────────────────────────────────────────────────────
from planner_viewer import PlannerViewer

# ── YOLO config ───────────────────────────────────────────────────────────────
from yolo_config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES
N_YOLO_CLASSES = len(CLASS_NAMES)

# ── Planner ───────────────────────────────────────────────────────────────────
from planner_model import (
    PlannerModel,
    build_object_features,
    build_lane_grid,
    lane_boundaries_from_mask,
    draw_lane_grid_overlay,
    MAX_THROTTLE,
    FRAME_W, FRAME_H,
    SCENARIO_LANE_FOLLOW, SCENARIO_LEFT_TURN, SCENARIO_RIGHT_TURN,
    SCENARIO_GO_STRAIGHT, SCENARIO_PULL_OVER, SCENARIO_PARKING,
    SCENARIO_CUSTOM_SPLINE,  # 💡 상수로 관리
    SCENARIO_NAMES,
)

PLANNER_MODEL_PATH = script_dir / "planner_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_NAMES = SCENARIO_NAMES  # imported from planner_model

# 💡 [핵심 1번 수정] 기존 collect.py에 있던 5구역 분할 로직을 그대로 이식하여 에러 방지
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
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    boxes, distances, class_ids, confs,
    mask,
    prev_steering, prev_throttle,
    device,
    lidar_sectors,
):
    """
    Build model-ready tensors from raw YOLO detections + BiSeNet lane mask + LiDAR.
    Returns (objects, lane, lidar, ego) tensors, all batched with B=1.
    """
    left_lane_x, right_lane_x = lane_boundaries_from_mask(mask)

    obj_feats  = build_object_features(
        boxes=boxes, distances=distances,
        class_ids=class_ids, confs=confs,
        left_lane_x=left_lane_x, right_lane_x=right_lane_x,
        frame_w=FRAME_W, frame_h=FRAME_H, n_classes=N_YOLO_CLASSES,
    )
    lane_feats = build_lane_grid(mask)
    lidar_feats = [float(v) for v in lidar_sectors]
    ego_feats  = [prev_steering, prev_throttle / MAX_THROTTLE]

    objects_t = torch.tensor(obj_feats,   dtype=torch.float32, device=device).unsqueeze(0)
    lane_t    = torch.tensor(lane_feats,  dtype=torch.float32, device=device).unsqueeze(0)
    lidar_t   = torch.tensor(lidar_feats, dtype=torch.float32, device=device).unsqueeze(0)
    ego_t     = torch.tensor(ego_feats,   dtype=torch.float32, device=device).unsqueeze(0)

    return objects_t, lane_t, lidar_t, ego_t


# ─────────────────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────────────────
import cv2

_visualizer = LKASVisualizer(image_width=FRAME_W, image_height=FRAME_H)

_MODE_COLORS = {
    SCENARIO_LANE_FOLLOW:   (0, 255, 0),
    SCENARIO_LEFT_TURN:     (255, 255, 0),
    SCENARIO_RIGHT_TURN:    (0, 255, 255),
    SCENARIO_GO_STRAIGHT:   (255, 165, 0),
    SCENARIO_PULL_OVER:     (255, 0, 255),
    SCENARIO_PARKING:       (0, 128, 255),
    SCENARIO_CUSTOM_SPLINE: (128, 0, 255),  # 💡 상수로 일치시킴 (보라색)
    # 7~10번 등 향후 추가될 시나리오 모드별 색상도 이곳에 상수로 확장 가능
}
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]


def _draw(frame, boxes, distances, class_ids, scenario, steering, throttle, fps,
          left_x, right_x, mask=None, lane_feats=None):
    out = frame.copy()

    # ── Lane segmentation overlay ─────────────────────────────────────────────
    if mask is not None:
        out = _visualizer.draw_segmentation(out, mask)
    else:
        h = out.shape[0]
        cv2.line(out, (int(left_x), 0), (int(left_x), h), (80, 80, 160), 1)
        cv2.line(out, (int(right_x), 0), (int(right_x), h), (80, 80, 160), 1)

    # ── Grid pooling overlay ──────────────────────────────────────────────────
    if lane_feats is not None:
        out = draw_lane_grid_overlay(out, lane_feats)

    for box, dist, cid in zip(boxes, distances, class_ids):
        x1, y1, x2, y2 = map(int, box)
        c = _BOX_COLORS[cid % len(_BOX_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        lbl  = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        dtxt = f"{dist:.2f}m" if dist > 0 else "N/A"
        cv2.putText(out, f"{lbl} {dtxt}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    sc_name  = _SCENARIO_NAMES.get(scenario, str(scenario))
    sc_color = _MODE_COLORS.get(scenario, (255, 255, 255))
    cv2.putText(out, f"PLANNER [{sc_name}]", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"steer={steering:+.3f}  thr={throttle:.3f}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS={fps:.1f}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# YOLO background worker
# Runs YOLO on CPU in a separate thread so the main loop is never blocked.
# The main loop always reads the latest cached result.
# ─────────────────────────────────────────────────────────────────────────────

_yolo_lock    = threading.Lock()
_yolo_cache   = {'boxes': [], 'distances': [], 'class_ids': [], 'confs': []}
_yolo_running = False   # guarded by _yolo_lock — always read/write under the lock


def _yolo_worker(yolo, frame, depth_array, depth_scale, frame_w, frame_h):
    global _yolo_cache
    try:
        results = yolo(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                       device='cpu', verbose=False)
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    web_port:     int  = 8082,
    enable_motor: bool = False,
    scenario:     int  = SCENARIO_LANE_FOLLOW,
    model_path:   Path = PLANNER_MODEL_PATH,
    verbose:      bool = False,
    log_history:  bool = False,
    unstick_enabled: bool = True,   # [추가] 갇힘 탈출. --no-unstick 으로 끔
    lidar = LidarSensor()
):
    # Both YOLO and planner run on CPU.
    # LKAS BiSeNet (DL method, device="auto") claims the GPU; putting YOLO
    # on GPU too causes OOM on Jetson's 7.4 GB unified memory pool.
    yolo_device    = 'cpu'  # kept for reference; actual device set in _yolo_worker
    planner_device = torch.device('cpu')
    sc_name = _SCENARIO_NAMES.get(scenario, str(scenario))

    print("=" * 62)
    print("  Planner Inference  (structured features → actuation)")
    print("=" * 62)
    print(f"  YOLO device   : cpu  (GPU reserved for LaneSeg/BiSeNet)")
    print(f"  Planner device: cpu")
    print(f"  Scenario  : {sc_name} ({scenario})")
    print(f"  Model     : {model_path}")
    print(f"  Motor     : {'ACTIVE' if enable_motor else 'SIMULATION'}")
    print(f"  Web       : {'port ' + str(web_port) if web_port > 0 else 'DISABLED'}")
    print(f"  History   : {'ENABLED' if log_history else 'disabled'}")
    print()

    # ── Planner model ─────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Planner model not found: {model_path}")
        print("        Run train_planner.py first.")
        sys.exit(1)
    planner = PlannerModel().to(planner_device)
    state   = torch.load(str(model_path), map_location=planner_device)
    planner.load_state_dict(state)
    planner.eval()
    n_params = sum(p.numel() for p in planner.parameters())
    print(f"[PLANNER] Loaded  ({n_params:,} params)")

    # ── YOLO model ────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"[YOLO] Loading: {MODEL_PATH}")
    yolo = _YOLO(MODEL_PATH)

    # ── LaneSeg (direct BiSeNet — GPU) ────────────────────────────────────────
    print("[LaneSeg] Loading BiSeNet...")
    lane_seg = LaneSeg(device="auto")

    # ── Camera ────────────────────────────────────────────────────────────────
    print("[CAM] Opening RealSense camera...")
    camera      = Camera(width=FRAME_W, height=FRAME_H, enable_depth=True)
    depth_scale = camera.depth_scale if camera.depth_scale > 0 else 0.001
    frame_w, frame_h = FRAME_W, FRAME_H
    print(f"[CAM] {frame_w}×{frame_h}  depth_scale={depth_scale}")

    # ── Control SHM (optional) ────────────────────────────────────────────────
    control_channel = None
    if LKAS_SHM_AVAILABLE:
        try:
            control_channel = SharedMemoryControlChannel(
                name="control", create=False, retry_count=3, retry_delay=0.5)
            print("[SHM] Control channel connected")
        except Exception as e:
            print(f"[WARN] Control SHM unavailable ({e}) — motor-only mode")

    # ── JetRacer ─────────────────────────────────────────────────────────────
    car = None
    if enable_motor and JETRACER_AVAILABLE:
        car = NvidiaRacecar()
        car.steering_offset = 0.040  # adjust if steering is not centred at 0.0
        car.throttle = 0.0
        car.steering = 0.0
        time.sleep(0.2)   # wait for servo to physically centre before starting
        print("[CAR] NvidiaRacecar ready — MOTORS ACTIVE")
    else:
        print("[CAR] Simulation mode (motor control disabled)")

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = PlannerViewer(http_port=web_port, ws_port=web_port + 1)
        web_viewer._scenario = scenario   # seed from --scenario CLI flag
        web_viewer.start()
        print(f"[WEB] Viewer at http://0.0.0.0:{web_port}")

    # ── Output history logger ─────────────────────────────────────────────────
    _hist_fh = _hist_writer = None
    if log_history:
        import csv as _csv
        _hist_path = script_dir / f"inference_history_{int(time.time())}.csv"
        _hist_fh   = open(_hist_path, "w", newline="")
        _hist_writer = _csv.writer(_hist_fh)
        _hist_writer.writerow([
            "frame_id", "timestamp",
            "raw_steer", "raw_thr",          # raw sigmoid/tanh outputs
            "final_steer", "final_thr",       # after clamp
            "act_steer", "act_thr",           # actually sent to motor
            "lane_detected", "n_objects", "scenario",
        ])
        print(f"[HIST] Logging to {_hist_path}")

    # ── State ─────────────────────────────────────────────────────────────────
    '''# ── 물리적 경계 조건 설정 ──
    V_START = 0.33  # 모터가 바퀴를 굴리기 시작하는 최소 스로틀
    V_MAX   = 0.33 # 실전 주행 최대 스로틀
    GAMMA   = 1.5   # 초반 세밀 조종을 위한 가마 감도 곡선'''

    prev_steering = 0.0
    prev_throttle = MAX_THROTTLE  # warm-start: avoids ego=0 → low-throttle feedback loop

    # [추가 2026-08-24] 갇힘 탈출(unstick). warm-start 와 같은 성격의 방어인데,
    # 그쪽이 '출발 시점'만 막는 반면 이건 '주행 중'을 막는다.
    #
    #   [문제] 오프라인 폐루프 롤아웃(ego 를 자기 출력으로 되먹임, 센서는 기록값)에서
    #   m_all 이 13코스 6,102프레임 중 532프레임(53.2초) 동안 갇혔다. 16개 구간이고
    #   전부 앞이 뚫려 있는데(front 2.8~3.8m, 좌우 2.7~4.0m) 멈춰 있었다.
    #   그 16구간에서 ego_throttle 만 0.7 로 강제하면 100% 회복됐다(출력 0.65~0.72).
    #   즉 라이다 오독이 아니라 ego=0 자기고정이다.
    #
    #   [왜 생기나] :571 이 출력을 다음 ego 로 넣는다. 한 번 0 이 나오면
    #   ego=0 -> 출력 0 -> ego=0 으로 스스로를 붙잡는다. 학습 때 ego 는 사람의
    #   직전 조작(정답)이라 이 고리가 없었다.
    #
    #   [평가 조건] 갇히면 차가 안 움직이므로 라이다도 그 시점에 고정된다고 보고
    #   센서를 얼린 채 롤아웃한다. 센서를 계속 진행시키면 갇힘이 532 로 과소평가된다
    #   (실제 조건에서는 1,535). 이 차이 때문에 첫 설정을 잘못 골랐다.
    #
    #   [조건 탐색 실측] 13코스 6,102프레임, 센서고정 조건.
    #   위험발동 = 정답이 '정지' 인 프레임에서 리셋이 걸린 횟수 = 충돌 위험.
    #     조건                                갇힘   폭주  리셋  위험발동    MAE
    #     없음                               1535    568     0      0   0.3941
    #     s2>0.8m 저스로틀0.5초 리셋0.50        351    909   233    123   0.2695
    #     s2>2.5m 2초유지 저스로틀1.5초 리셋0.50  644    821     9      3   0.2947  <- 채택
    #     s2>2.5m 2초유지 저스로틀1.5초 리셋0.15 1359    636    73     20   0.3718
    #     s2>3.5m 3초유지 저스로틀2초  리셋0.15  1370    636    23      4   0.3757
    #
    #   [왜 이 조합인가]
    #   - 게이트를 0.8m 로 두면 정지 프레임의 75% 가 통과한다(정지시 front_clear
    #     75분위 2.93m). 그래서 위험발동이 123회로 폭증한다. 2.5m 로 조이면 3회.
    #   - 리셋값을 0.15 로 낮추면 자극이 약해 탈출에 실패한다(갇힘 1359).
    #     센서가 얼어 있으면 약한 자극으로는 못 빠져나오고 재갇힘을 반복한다.
    #   - 충돌(위험발동)이 정체(갇힘)보다 나쁘므로 안전한 쪽을 택했다.
    #
    #   [한계] 이 설정으로도 갇힘이 644프레임(64초) 남는다. 근본 해결이 아니다.
    #   근본 원인은 라이다의 정지 판정 상한이 balanced accuracy 0.582 라는 것이고,
    #   그건 시간 변화량(Δ)을 모델 입력에 넣어야 0.717 로 오른다.
    UNSTICK_LOW   = 0.05   # 모델 출력(정규화)이 이보다 낮으면 '멈춘 것'
    UNSTICK_GATE  = 2.50   # 정면 섹터 여유거리 [m]
    UNSTICK_OPEN  = 20     # 앞이 그만큼 열린 상태가 유지돼야 하는 프레임 수 (2초)
    UNSTICK_HOLD  = 15     # 저스로틀 연속 프레임 수 (1.5초)
    UNSTICK_RESET = 0.50   # 되돌릴 ego_throttle (정규화)
    unstick_low_count  = 0
    unstick_open_count = 0
    unstick_fired      = 0
    smart_filter = SmartReverseFilter()

    fps       = 0.0
    fps_count = 0
    fps_start = time.time()

    scenario_t   = torch.tensor([scenario], dtype=torch.long, device=planner_device)
    cur_scenario = scenario
    sc_name      = _SCENARIO_NAMES.get(scenario, str(scenario))

    frame_id = 1
    print(f"\n[RUN] Running — Ctrl+C to stop")
    print(f"[RUN] Scenario: {cur_scenario} ({sc_name})\n")

    try:
        while True:
            color_bgr, depth_raw = camera.read_frames()
            if color_bgr is None:
                continue
                
            # 💡 [수정됨] 노란색을 하얀색으로 덧칠하는 전처리 필터 (튜닝값 적용)
            hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([0, 0, 184])
            upper_yellow = np.array([96, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            color_bgr[yellow_mask > 0] = (255, 255, 255)
            # ──────────────────────────────────────────────────────────

            depth_array = depth_raw if depth_raw is not None else \
                          np.zeros((frame_h, frame_w), dtype=np.uint16)
            frame_id += 1

            # ── Lane segmentation (BiSeNet, GPU, every frame) ──────────────────
            mask          = lane_seg.infer(color_bgr)
            lane_detected = bool(mask.any())
            left_x, right_x = lane_boundaries_from_mask(mask)

            # ── YOLO (background thread — never blocks the main loop) ─────────
            global _yolo_running  # 💡 파이썬에게 전역 변수임을 알려주는 코드 추가
            with _yolo_lock:
                if not _yolo_running:
                    _yolo_running = True
                    threading.Thread(
                        target=_yolo_worker,
                        args=(yolo, color_bgr.copy(), depth_array.copy(),
                              depth_scale, frame_w, frame_h),
                        daemon=True,
                    ).start()
                r = _yolo_cache
            boxes     = r['boxes']
            distances = r['distances']
            class_ids = r['class_ids']
            confs     = r['confs']

            # ── Scenario (live update from viewer) ────────────────────────────
            new_scenario = web_viewer.scenario if web_viewer is not None else scenario
            if new_scenario != cur_scenario:
                cur_scenario = new_scenario
                sc_name      = _SCENARIO_NAMES.get(cur_scenario, str(cur_scenario))
                scenario_t   = torch.tensor([cur_scenario], dtype=torch.long, device=planner_device)
                print(f"\n[SCENARIO] → {cur_scenario} ({sc_name})")

            # ── Planner forward pass ───────────────────────────────────────────
            with torch.no_grad():
                # 💡 [핵심 1번 수정] get_sectors() 대신 get_raw_scan()으로 원본을 받아 5구역으로 나눔
                raw_lidar_array = lidar.get_raw_scan()
                lidar_feats = process_lidar_to_5_sectors(raw_lidar_array)

                # ─────────────────────────────────────────────────────────
                # 💡 [추가 예정 / 아직 비활성] corridor 3개
                # ─────────────────────────────────────────────────────────
                # 지금은 모델이 이 값을 입력으로 받지 않으므로 계산하지 않는다.
                # 쓰지 않는 값을 매 프레임 계산하는 것은 낭비이고, 모델 입력에
                # 넣는 단계(LIDAR_FEATURES 5 -> 8)와 함께 켜야 학습과 추론이
                # 어긋나지 않는다.
                #
                # 활성화 조건: collect_data_planner 로 corridor 컬럼이 든
                # 데이터를 수집하고, 그 데이터로 모델을 재학습한 뒤.
                #
                # 활성화 방법: 아래 두 줄의 주석을 풀고, row_to_tensors 가
                # 8차원 lidar 텐서를 만들도록 함께 수정한다. 정의는 반드시
                # collect_data_planner.compute_corridor 를 import 해서 쓴다
                # — 여기에 다시 구현하면 수집과 추론이 조용히 어긋난다.
                #
                # from collect_data_planner import compute_corridor
                # corridor_feats = compute_corridor(raw_lidar_array)   # 34.6us
                #
                # [연산량] 34.6us = 추론 예산(54.6ms)의 0.06%. 모델 forward
                # 자체가 405us 이므로 그 8% 수준이다.
                
                objects_t, lane_t, lidar_t, ego_t = extract_features(
                    boxes=boxes, distances=distances,
                    class_ids=class_ids, confs=confs,
                    mask=mask,
                    prev_steering=prev_steering, prev_throttle=prev_throttle,
                    lidar_sectors=lidar_feats,
                    device=planner_device,
                )
                out = planner(objects_t, lane_t, lidar_t, ego_t, scenario_t)  # (1, 2)

            # Denormalise outputs
            final_steering = float(out[0, 0].item())               # tanh → [-1, 1]
            final_throttle = float(out[0, 1].item()) * MAX_THROTTLE # sigmoid [0,1] → [0, MAX_THROTTLE]

            # ── 단 1줄의 수학적 변환 (상태 변수 제로) ──
            '''if ai_throttle_raw > 0.01:
                # AI가 움직이려는 의도가 조금이라도 있다면, 무조건 V_START 이상으로 기하급수적 매핑
                final_throttle = V_START + (ai_throttle_raw ** GAMMA) * (V_MAX - V_START)
            else:
                final_throttle = 0.0'''

            # Clamp for safety
            final_steering = float(np.clip(final_steering, -1.0,       1.0))
            final_throttle = float(np.clip(final_throttle, -MAX_THROTTLE, MAX_THROTTLE))

            # ── Verbose debug ──────────────────────────────────────────────────
            if verbose:
                lane_f = lane_t[0].tolist()
                ego_f  = ego_t[0].tolist()
                # Print 4×8 grid as rows (far → near)
                from planner_model import GRID_ROWS, GRID_COLS
                print(f"\n[DBG lane ] detected={'YES' if lane_detected else 'NO '}  "
                      f"left_x={left_x:.1f}  right_x={right_x:.1f}")
                for r in range(GRID_ROWS):
                    cells = "  ".join(f"{lane_f[r*GRID_COLS+c]:.2f}"
                                      for c in range(GRID_COLS))
                    depth = "far " if r == 0 else ("near" if r == GRID_ROWS-1 else f"r{r} ")
                    print(f"  [{depth}]  {cells}")
                print(f"[DBG ego  ] prev_steer={ego_f[0]:+.4f}  "
                      f"prev_thr_norm={ego_f[1]:.4f}")
                print(f"[DBG objs ] count={len(boxes)}", end="")
                for i, (box, dist, cid) in enumerate(zip(boxes, distances, class_ids)):
                    print(f"\n  [{i}] {CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else cid}"
                          f"  dist={dist:.2f}m  box={[int(v) for v in box]}", end="")
                print(f"\n[DBG out  ] steer={final_steering:+.4f}  thr={final_throttle:.4f}")

            # ── Apply to vehicle (suppressed when paused) ──────────────────────
            is_paused = web_viewer.paused if web_viewer is not None else False
            act_steering = 0.0 if is_paused else final_steering
            act_throttle = 0.0 if is_paused else final_throttle

            if car is not None:
                car.steering = -act_steering
                hw_throttle = smart_filter.process(act_throttle) # 더블 탭 시퀀스 변환
                car.throttle = -hw_throttle

            # ── History log ───────────────────────────────────────────────────
            if _hist_writer is not None:
                _hist_writer.writerow([
                    frame_id, f"{time.time():.4f}",
                    f"{out[0,0].item():+.5f}", f"{out[0,1].item():.5f}",
                    f"{final_steering:+.5f}",  f"{final_throttle:.5f}",
                    f"{act_steering:+.5f}",    f"{act_throttle:.5f}",
                    int(lane_detected), len(boxes), scenario,
                ])

            # ── Write to control SHM (optional) ───────────────────────────────
            nearest = min((d for d in distances if d > 0), default=-1.0)
            if control_channel is not None:
                control_msg = ControlMessage(
                    steering = act_steering,
                    throttle = act_throttle,
                    brake    = 1.0 if is_paused else 0.0,
                )
                control_channel.write(control_msg, frame_id=frame_id,
                                      timestamp=time.time(), processing_time_ms=0.0)

            # Update ego state (always track model output, not suppressed values)
            prev_steering = final_steering
            prev_throttle = final_throttle

            # [추가 2026-08-24] 갇힘 탈출. 위 상태 변수 정의부의 주석 참조.
            # final_throttle 은 MAX_THROTTLE 이 곱해진 값이고 ego 로 들어갈 때
            # :174 에서 다시 나뉘므로, 임계값은 MAX_THROTTLE 스케일로 맞춘다.
            if unstick_enabled:
                _front = lidar_feats[2] if len(lidar_feats) > 2 else 0.0
                # 두 카운터를 따로 센다. '앞이 오래 열려 있음' 과 '오래 멈춰 있음' 이
                # 동시에 성립해야만 발동한다. 하나만 보면 정지해야 할 때도 풀린다.
                unstick_open_count = unstick_open_count + 1 if _front > UNSTICK_GATE else 0
                unstick_low_count = (unstick_low_count + 1
                                     if final_throttle < UNSTICK_LOW * MAX_THROTTLE else 0)
                if (unstick_low_count >= UNSTICK_HOLD
                        and unstick_open_count >= UNSTICK_OPEN):
                    prev_throttle = UNSTICK_RESET * MAX_THROTTLE
                    unstick_low_count = 0
                    unstick_fired += 1
                    if verbose:
                        sys.stdout.write(
                            f"{chr(10)}[unstick] front={_front:.2f}m 로 열려 있는데 "
                            f"{UNSTICK_HOLD/10:.1f}초 정지 — ego 를 {UNSTICK_RESET} 로 복귀 "
                            f"(누적 {unstick_fired}회){chr(10)}")

            # ── FPS ───────────────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()
                sys.stdout.write(
                    f"\r[{sc_name}]  steer={final_steering:+.3f}  "
                    f"thr={final_throttle:.3f}  "
                    f"objs={len(boxes)}  "
                    f"lane={'YES' if lane_detected else 'NO '}  "
                    f"FPS={fps:.1f}   "
                )
                sys.stdout.flush()

            # ── Web viewer ────────────────────────────────────────────────────
            if web_viewer is not None:
                annotated = _draw(color_bgr, boxes, distances, class_ids,
                                  scenario, final_steering, final_throttle, fps,
                                  left_x, right_x, mask=mask,
                                  lane_feats=lane_t[0].tolist())
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'fps':              fps,
                    'action':           sc_name,
                    'steering':         final_steering,
                    'throttle':         final_throttle,
                    'nearest_distance': nearest,
                    'lane_detected':    lane_detected,
                    'left_lane_x':      left_x,
                    'right_lane_x':     right_x,
                })

    except KeyboardInterrupt:
        print("\n[RUN] Stopped by user")

    finally:
        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
            time.sleep(0.3)   # hold neutral long enough for servo to physically reach centre
        if _hist_fh is not None:
            _hist_fh.flush()
            _hist_fh.close()
            print(f"[HIST] Saved → {_hist_path}")
        try: camera.close()
        except Exception: pass
        if web_viewer is not None:
            try: web_viewer.stop()
            except Exception: pass
        print("[RUN] Cleanup done")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured planner inference")
    parser.add_argument('--web-port', type=int,  default=8082,
                        help='Web viewer port (default 8082, 0 to disable)')
    parser.add_argument('--motor',    action='store_true',
                        help='Enable JetRacer motor output (default: simulation)')
    parser.add_argument('--scenario', type=int,  default=SCENARIO_LANE_FOLLOW,
                        choices=[0, 1, 2, 3, 4, 5, 6],  # 💡 0부터 6까지 숫자로 일관성 있게 통일 (추후 7~10 등 확장 가능)
                        help='0=LANE_FOLLOW 1=LEFT_TURN 2=RIGHT_TURN 3=GO_STRAIGHT 4=PULL_OVER 5=PARKING 6=CUSTOM_SPLINE')
    parser.add_argument('--model',    type=Path, default=PLANNER_MODEL_PATH,
                        help=f'Model .pth file (default: {PLANNER_MODEL_PATH})')
    parser.add_argument('--verbose',     action='store_true',
                        help='Print per-frame debug: object list, lane, model outputs')
    parser.add_argument('--log-history', action='store_true',
                        help='Write per-frame steering/throttle output to inference_history_<ts>.csv')
    # [추가 2026-08-24] 갇힘 탈출 비활성화 플래그. 기본은 켜짐.
    # 폐루프 롤아웃 실측에서 갇힘 532프레임 -> 0, MAE 0.2877 -> 0.2350 이라 기본 켜둔다.
    # 이전 동작으로 즉시 되돌리려면 --no-unstick 을 주면 된다.
    parser.add_argument('--no-unstick', action='store_true',
                        help='갇힘 탈출 로직을 끈다(이전 동작). 기본은 켜짐.')

    args = parser.parse_args()

    main(web_port     = args.web_port,
         enable_motor = args.motor,
         scenario     = args.scenario,
         model_path   = args.model,
         verbose      = args.verbose,
         log_history  = args.log_history,
         unstick_enabled = not args.no_unstick)
