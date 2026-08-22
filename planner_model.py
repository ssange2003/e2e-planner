#!/usr/bin/env python3
"""
Planner Model — Structured Feature Planner
===========================================
Takes YOLO object detections + lane segmentation data + ego state
and outputs (steering, throttle) — no camera feed.

This is the Tesla-style perception→planner split:
  [YOLO + Lane Seg]  →  structured features  →  PlannerModel  →  [steering, throttle]

Feature layout
--------------
Objects   : N_MAX_OBJECTS rows × OBJ_FEATURES cols  (sorted by distance, padded with zeros)
  [valid, class_norm, confidence, dist_norm, lat_offset, width_norm, height_norm, lane_overlap]

Lane      : LANE_FEATURES cols  (6×12 spatial grid from BiSeNet mask)
  [lane_r0c0 … lane_r3c7]  per-cell lane fraction [0.0–1.0], row-major

Ego state : EGO_FEATURES cols
  [prev_steering, prev_throttle]

Scenario  : int  (task context token, embedded)
  0 = LANE_FOLLOW  1 = OBSTACLE_AVOID  2 = PARKING  3 = STOP

Output    : [steering, throttle]  both are raw floats (no activation clamp here)
  steering  ∈ [-1, 1]   (matches JetRacer convention)
  throttle  ∈ [ 0, 1]   (scaled by MAX_THROTTLE in the driver)
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Feature dimensions  (must stay in sync across collect / augment / train / infer)
# ─────────────────────────────────────────────────────────────────────────────
N_MAX_OBJECTS = 5      # objects tracked per frame (padded to this length)

# 💡 [추가된 부분: 라이다 특징 차원 정의]
# 원본 코드에는 없던 라이다 5구역(s0~s4) 데이터를 입력받기 위해 특징 차원(5)이 신설되었습니다.
LIDAR_FEATURES = 5  

OBJ_FEATURES  = 8     # features per object slot
GRID_ROWS     = 6     # spatial grid rows (far → near)
GRID_COLS     = 12    # spatial grid columns (left → right)
LANE_FEATURES = GRID_ROWS * GRID_COLS   # 72 — spatial grid of lane fractions
EGO_FEATURES  = 2     # ego state features (prev_steering, prev_throttle)

# 💡 [변경된 부분: 시나리오 어휘 사전 크기 확장]
# 원본은 6개였으나, 6번(스플라인) 및 미래 확장 시나리오(최대 10개)까지 넉넉히 수용하도록 N_SCENARIOS가 10으로 늘어났습니다.
N_SCENARIOS   = 10    
SCENARIO_DIM  = 8     # embedding dimension for scenario token

OBJECT_BLOCK_DIM = N_MAX_OBJECTS * OBJ_FEATURES   # 40

# 💡 [변경된 부분: 전체 평탄화 차원(Total Flat Dim) 계산 수정]
# 라이다 5차원이 추가됨에 따라 전체 데이터 크기가 기존보다 5만큼 늘어나도록 수식이 보완되었습니다.
TOTAL_FLAT_DIM   = OBJECT_BLOCK_DIM + LANE_FEATURES + EGO_FEATURES + LIDAR_FEATURES # 114

# Fallback lane boundary x-coordinates when the mask contains no lane pixels
FIXED_LEFT_LANE_X  = 255
FIXED_RIGHT_LANE_X = 485

# Scenario tokens
SCENARIO_LANE_FOLLOW = 0   # normal driving — obstacle avoidance implicit via YOLO features
SCENARIO_LEFT_TURN   = 1   # turning left at junction
SCENARIO_RIGHT_TURN  = 2   # turning right at junction
SCENARIO_GO_STRAIGHT = 3   # straight through intersection / past stop line
SCENARIO_PULL_OVER   = 4   # pulling over to roadside (emergency stop)
SCENARIO_PARKING     = 5   # parking manoeuvre

# 💡 [추가된 부분: 커스텀 스플라인 시나리오 토큰]
# 사용자가 추가한 알고리즘 자율주행 모드(시나리오 6번)에 대한 상수가 새롭게 선언되었습니다.
SCENARIO_CUSTOM_SPLINE = 6 

SCENARIO_NAMES = {
    SCENARIO_LANE_FOLLOW:   "LANE_FOLLOW",
    SCENARIO_LEFT_TURN:     "LEFT_TURN",
    SCENARIO_RIGHT_TURN:    "RIGHT_TURN",
    SCENARIO_GO_STRAIGHT:   "GO_STRAIGHT",
    SCENARIO_PULL_OVER:     "PULL_OVER",
    SCENARIO_PARKING:       "PARKING",
    # 💡 [추가된 부분: 시나리오 이름 맵핑]
    # 웹 뷰어 및 로그 터미널에 "CUSTOM_SPLINE"이라는 이름이 정상 출력되도록 딕셔너리에 추가되었습니다.
    SCENARIO_CUSTOM_SPLINE: "CUSTOM_SPLINE", 
}

# Normalisation constants  (shared between collection and inference)
MAX_DIST_M     = 5.0    # clip distances beyond this to 1.0

# 💡 [추가된 부분: 라이다 상한 클리핑]
# 라이다는 다른 입력과 달리 정규화 없이 raw 미터로 들어가고, 전체 값의 27.4%가
# 정확히 5.0 에 포화돼 있다. 그런데 collect_data_planner.get_min_dist 에서
# 5.0 은 "반사가 아예 없음"과 "가장 가까운 것이 5m 밖"을 겸해 의미가 모호하다.
# 게다가 이 값은 개활 테스트장에서만 나온다 — 사각 박스형 대회장에서는 어느
# 방향이든 벽이 반사를 주므로 5.0 이 거의 사라진다.
#
# [실측 근거] S구간을 제외한 6개 코스 전부 왼쪽 2~3.5m 에 구조물이 있고
# 오른쪽은 비어 있다(테스트장 왼쪽 벽). 모델이 이 좌우 비대칭을 주행 규칙이
# 아니라 세상의 법칙으로 학습한다.
#   코스별 평균 |좌우 비대칭|   원본 1.704  ->  2.0m 클립 0.010
#   S구간 좌우 대칭            +0.03/+0.05  그대로 (무손상)
#   근거리(<0.45m) 프레임      보존율 100%
#
# 2.0m 를 고른 이유: 1.2 / 1.5 / 2.0 이 지문 제거 효과가 모두 0.010 으로
# 동일하므로, 같은 효과라면 접근 정보가 가장 많이 남는 값이 맞다.
# 판단이 실제로 일어나는 거리는 avoidance p75=1.51m, s_curve p75=1.23m 라
# 2.0m 는 그 위에 있어 접근 과정을 온전히 담는다.
#
# [주의] 이 값을 바꾸면 입력 분포가 바뀐다. 다른 값으로 학습된 체크포인트는
# 에러 없이 조용히 어긋나므로 반드시 재학습해야 한다.
#
# ─────────────────────────────────────────────────────────────────────────
# [실측으로 반박됨 — 2026-08-22]  기본값을 2.0 에서 0.0(비활성)으로 되돌린다.
#
# 40에폭 대조 실험(base / ego-dropout 0.3 / lidar-clip 2.0)에서 클리핑은
# 의도한 효과를 내지 못했고 오히려 역효과가 측정되었다.
#
#   스로틀에서 ego 점유율     base 59.7%  ->  clip 68.7%   (악화)
#   S구간 lidar/ego 비율      base 0.28x  ->  clip 0.25x   (개선 없음)
#
# 해석: 라이다에서 원거리 정보를 걷어내자 모델이 남은 불확실성을 ego(직전
# 행동)로 메웠다. 지문은 지웠지만 그 자리를 관성이 채운 것이다.
# "입력을 줄이면 다른 입력을 더 본다"는 보장이 없다는 것을 다시 확인했다.
#
# 다만 클리핑 자체가 정보를 파괴하지는 않는다는 것도 함께 확인되었다:
#   장애물 정지 프레임 중 정면 2.0m 미만 비율   avoidance/s_curve/noise 100%
#   정지 전 접근 경보 시간(2.0m 진입~정지)      중앙 4.0초, 최소 1.2초
# 즉 장애물은 클립 후에도 보인다. 잃는 것은 2.0m 밖의 접근 추세뿐이다.
#
# 그래서 삭제하지 않고 옵트인 플래그로 남긴다. 박스형 대회장 데이터를
# 확보한 뒤 도메인 이식성 관점에서 재평가할 가치가 있다.
# ─────────────────────────────────────────────────────────────────────────
LIDAR_CLIP_M   = 0.0

# 💡 [수정된 부분: 최대 스로틀 상숫값 조정]
# 원본 코드의 0.35에서 실제 수집 및 주행 캘리브레이션에 맞춘 0.383으로 미세 조정되었습니다.(add modify 0.36)
MAX_THROTTLE   = 0.41 

# 💡 [추가된 부분: D435i IMU 파생 특징 정의]
# 게임패드 데드존이 만들어내는 "의도하지 않은 throttle=0"과 운전자가 실제로
# 멈춘 진짜 정지를 구분하려면, 차가 물리적으로 움직이는지를 직접 봐야 합니다.
# 라이다/카메라는 글리치 순간의 바깥 장면이 정상 주행과 동일하기 때문에
# 원리적으로 이 둘을 구분할 수 없습니다. IMU만이 자차 운동을 직접 관측합니다.
#
# [중요] 이 컬럼들은 csv_columns() 스키마에 포함되지 않습니다. 기존에 수집된
# CSV(IMU 없음)와 새로 수집할 CSV(IMU 있음)를 모두 읽을 수 있어야 하기 때문입니다.
# train_planner.py 의 스키마 검증은 부분집합 검사(missing = expected - actual)라
# 컬럼이 더 있어도 통과하며, 모델 입력은 컬럼명을 직접 지정해 만들므로
# 이 컬럼을 추가해도 모델 구조 변경도 재학습도 필요하지 않습니다.
IMU_COLUMNS = [
    "imu_motion",      # 최근 윈도우 |accel| 표준편차 — 노면 진동 = 이동 중인가
    "imu_yaw_rate",    # 수직축 gyro [rad/s] — 실제 회전 각속도
    "imu_accel_fwd",   # 전방축 accel [m/s^2] — 실제 감속이 있었는가
]
IMU_FEATURES = len(IMU_COLUMNS)

FRAME_W        = 848    # must match camera config — RealSense supported: 848x480, 640x480, 640x360
FRAME_H        = 480
N_YOLO_CLASSES = 80     # COCO classes; override if using custom model


# ─────────────────────────────────────────────────────────────────────────────
# Object feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_object_features(
    boxes:      list,      # list of [x1, y1, x2, y2] in pixels
    distances:  list,      # list of float metres (-1 = invalid)
    class_ids:  list,      # list of int
    confs:      list,      # list of float [0,1]
    left_lane_x:  float,
    right_lane_x: float,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
    n_classes: int = N_YOLO_CLASSES,
) -> list:
    """
    Convert raw YOLO output to a fixed-size normalised feature block.

    Returns a flat list of N_MAX_OBJECTS * OBJ_FEATURES floats.
    Objects are sorted closest-first; excess slots are zero-padded.

    Per-slot layout:
      [valid, class_norm, conf, dist_norm, lat_offset, w_norm, h_norm, lane_overlap]
    """
    lane_width  = max(right_lane_x - left_lane_x, 1.0)
    lane_center = (left_lane_x + right_lane_x) / 2.0

    # Build raw records and sort by distance (valid distances first, then invalids)
    records = []
    for box, dist, cid, conf in zip(boxes, distances, class_ids, confs):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cx = (x1 + x2) / 2.0

        # Lateral offset: signed, normalised by lane width  (−ve = left of centre)
        lat_offset = (cx - lane_center) / lane_width

        # Lane overlap fraction [0, 1]
        overlap = max(0.0, min(x2, right_lane_x) - max(x1, left_lane_x))
        lane_overlap = float(overlap / lane_width)

        dist_norm  = float(min(max(dist, 0.0), MAX_DIST_M) / MAX_DIST_M) if dist > 0 else 1.0
        sort_key   = dist_norm if dist > 0 else 2.0   # invalid last

        records.append((sort_key, [
            1.0,                                      # valid
            float(cid) / max(n_classes - 1, 1),       # class_norm
            float(conf),                              # confidence
            dist_norm,                                # dist_norm
            float(lat_offset),                        # lat_offset
            float((x2 - x1) / frame_w),              # width_norm
            float((y2 - y1) / frame_h),              # height_norm
            lane_overlap,                             # lane_overlap
        ]))

    records.sort(key=lambda r: r[0])

    # Pad / truncate to N_MAX_OBJECTS
    out: list[float] = []
    for i in range(N_MAX_OBJECTS):
        if i < len(records):
            out.extend(records[i][1])
        else:
            out.extend([0.0] * OBJ_FEATURES)   # zero-pad
    return out


def build_lane_grid(mask: "np.ndarray") -> list:
    """
    Convert a BiSeNet segmentation mask to a coarse spatial grid of lane fractions.

    Resizes the binary lane mask to (_GRID_H × _GRID_W) using INTER_AREA
    (which averages pixels, giving per-pixel fractions), then returns the
    mean lane fraction for each of the (GRID_ROWS × GRID_COLS) cells.

    Row 0 = far (top of image), Row GRID_ROWS-1 = near (bottom).
    Column 0 = left edge, Column GRID_COLS-1 = right edge.

    Args:
        mask: (H, W) uint8 — 0 = background, 1+ = lane  (from LaneSeg.infer)

    Returns:
        list of LANE_FEATURES (32) floats — lane fraction per cell [0.0–1.0],
        in row-major order (r0c0, r0c1, …, r3c7).
    """
    import cv2
    import numpy as _np

    _GRID_H = GRID_ROWS * 16   # 64
    _GRID_W = GRID_COLS * 14   # 112  (≈ 848×480 aspect at coarse scale)

    binary = (mask > 0).astype(_np.float32)
    small  = cv2.resize(binary, (_GRID_W, _GRID_H), interpolation=cv2.INTER_AREA)

    cell_h = _GRID_H // GRID_ROWS   # 16
    cell_w = _GRID_W // GRID_COLS   # 14

    features = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = small[r * cell_h:(r + 1) * cell_h,
                         c * cell_w:(c + 1) * cell_w]
            features.append(float(cell.mean()))
    return features


def lane_boundaries_from_mask(
    mask: "np.ndarray",
    roi_fraction: float = 0.33,
) -> tuple:
    """
    Derive approximate left/right lane boundary x-coordinates from a
    segmentation mask.  Uses the bottom roi_fraction of the image
    (near-field), finds leftmost and rightmost lane columns.

    Returns:
        (left_x, right_x) in pixels — or (FIXED_LEFT_LANE_X, FIXED_RIGHT_LANE_X)
        if no lane pixels are found in the ROI.
    """
    import numpy as _np
    h    = mask.shape[0]
    roi  = mask[int(h * (1.0 - roi_fraction)):, :]
    cols = _np.where(roi.any(axis=0))[0]
    if len(cols) < 2:
        return float(FIXED_LEFT_LANE_X), float(FIXED_RIGHT_LANE_X)
    return float(cols[0]), float(cols[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Grid visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_lane_grid_overlay(
    frame: "np.ndarray",
    grid_features: list,
    alpha: float = 0.45,
) -> "np.ndarray":
    """
    Draw the 4×8 lane-fraction grid as a semi-transparent overlay on *frame*.

    Each cell is shaded green with intensity proportional to its lane fraction.
    Grid lines are drawn in translucent white.  The fraction value is printed
    inside every cell so you can read the exact model input while driving.

    Call this *after* draw_segmentation() so the grid appears on top.

    Args:
        frame:         BGR image, already annotated with lane segmentation
        grid_features: 32 floats from build_lane_grid() — row-major [r0c0 … r3c7]
        alpha:         blend weight of the grid layer (default 0.45)

    Returns:
        New BGR image with grid overlay blended in.
    """
    import cv2
    import numpy as _np

    h, w   = frame.shape[:2]
    cell_h = h // GRID_ROWS   # 120 for 480-px frames
    cell_w = w // GRID_COLS   # 106 for 848-px frames

    overlay = frame.copy()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            frac = float(grid_features[r * GRID_COLS + c])
            y1 = r * cell_h
            y2 = y1 + cell_h
            x1 = c * cell_w
            x2 = x1 + cell_w

            # Cell fill — bright green, intensity ∝ lane fraction
            if frac > 0.02:
                intensity = int(60 + 195 * frac)   # 60 (dim) … 255 (full)
                cv2.rectangle(overlay, (x1, y1), (x2, y2),
                              (0, intensity, 0), -1)  # BGR green

            # Fraction label in cell centre
            label = f"{frac:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tx = x1 + (cell_w - tw) // 2
            ty = y1 + (cell_h + th) // 2
            cv2.putText(overlay, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # Blend cell fills with the incoming frame
    blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

    # Draw grid lines on top (always fully opaque)
    for r in range(GRID_ROWS + 1):
        y = r * cell_h
        cv2.line(blended, (0, y), (w, y), (180, 180, 180), 1, cv2.LINE_AA)
    for c in range(GRID_COLS + 1):
        x = c * cell_w
        cv2.line(blended, (x, 0), (x, h), (180, 180, 180), 1, cv2.LINE_AA)

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PlannerModel(nn.Module):
    """
    Structured-input planner.

    forward(objects, lane, ego, lidar, scenario) → (B, 2)  [steering, throttle]

    Inputs (all float32 except scenario which is long):
      objects  : (B, N_MAX_OBJECTS * OBJ_FEATURES)   40-dim
      lane     : (B, LANE_FEATURES)                  72-dim  (6×12 spatial grid)
      ego      : (B, EGO_FEATURES)                    2-dim
      lidar    : (B, LIDAR_FEATURES)                  5-dim
      scenario : (B,)                                 long
    """

    def __init__(self):
        super().__init__()

        # Object encoder
        self.obj_enc = nn.Sequential(
            nn.Linear(OBJECT_BLOCK_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Lane encoder — wider + deeper to exploit the 6×12 spatial grid
        self.lane_enc = nn.Sequential(
            nn.Linear(LANE_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Ego encoder
        self.ego_enc = nn.Sequential(
            nn.Linear(EGO_FEATURES, 32),
            nn.ReLU(inplace=True),
        )

        # 💡 [추가된 부분: 라이다 인코더 신경망 레이어]
        # 원본에는 없던 5차원 라이다 특징을 입력받아 16차원으로 고차원 임베딩하는 선형 레이어가 추가되었습니다.
        self.lidar_enc = nn.Sequential(
            nn.Linear(LIDAR_FEATURES, 16),
            nn.ReLU(inplace=True),
        )

        # Scenario embedding
        self.scenario_embed = nn.Embedding(N_SCENARIOS, SCENARIO_DIM)

        # 💡 [수정된 부분: 특징 융합(Fusion) 차원 확장]
        # 기존에는 (객체 64 + 차선 64 + 자차 32 + 시나리오 8 = 168) 이었으나,
        # 라이다 인코딩 결과물(16)이 가세하여 fused_dim이 총 184차원으로 확장되었습니다.
        fused_dim = 64 + 64 + 32 + 16 + SCENARIO_DIM
        
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Separate heads
        self.steering_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh(),                # steering  ∈ [-1, 1]
        )
        
        # 💡 [수정된 부분: 스로틀 출력 활성화 함수 변경 (Sigmoid → Tanh)]
        # 원본 코드는 [0, 1] 범위를 내뱉는 Sigmoid였으나, RC카 후진 제어(-1 ~ 1 또는 음수 전송)를 위해 Tanh로 변경되었습니다.
        self.throttle_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh(),                
        )

    # 💡 [수정된 부분: forward 메서드 입력 인자 및 연산 순서 변경]
    # 기존 (objects, lane, ego, scenario) 순서에서 세 번째 자리에 lidar 텐서가 새롭게 끼어들었습니다.
    def forward(
        self,
        objects:  torch.Tensor,   # (B, 40)
        lane:     torch.Tensor,   # (B, 72)
        lidar:    torch.Tensor,   # (B, 5)   💡 [추가] 3번째 인자로 라이다 배치 텐서 수신
        ego:      torch.Tensor,   # (B, 2)   💡 [이동] 4번째 인자로 순서 변경
        scenario: torch.Tensor,   # (B,) long
    ) -> torch.Tensor:            # (B, 2)  [steering, throttle]

        o = self.obj_enc(objects)
        l = self.lane_enc(lane)
        ld = self.lidar_enc(lidar) # 💡 [추가] 라이다 데이터 순방향 연산 실행
        e = self.ego_enc(ego)
        s = self.scenario_embed(scenario)

        # 💡 [수정된 부분: 특징 벡터 병합(Concatenate) 구조 반영]
        # 인자로 들어온 순서에 발맞추어 융합 리스트에 [o, l, ld, e, s] 형태로 라이다가 정확히 포함되었습니다.
        fused  = torch.cat([o, l, ld, e, s], dim=1)    
        trunk  = self.shared_trunk(fused)                     

        steering = self.steering_head(trunk)         
        throttle = self.throttle_head(trunk)         

        return torch.cat([steering, throttle], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# CSV column schema  (generated once, shared by all scripts)
# ─────────────────────────────────────────────────────────────────────────────

def csv_columns() -> list[str]:
    """Return ordered column names for the structured dataset CSV."""
    cols = ["frame_id"]
    for i in range(N_MAX_OBJECTS):
        cols += [
            f"obj{i}_valid",
            f"obj{i}_class_norm",
            f"obj{i}_conf",
            f"obj{i}_dist_norm",
            f"obj{i}_lat_offset",
            f"obj{i}_width_norm",
            f"obj{i}_height_norm",
            f"obj{i}_lane_overlap",
        ]
    cols += [f"lane_r{r}c{c}"
             for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    cols += ["ego_steering", "ego_throttle"]
    
    # 💡 [추가된 부분: CSV 스키마에 라이다 5개 구역 컬럼 등록]
    # 학습 데이터셋(CSV)을 저장하거나 읽어올 때 라이다 s0~s4 데이터가 빠짐없이 기록되도록 컬럼 명칭이 추가되었습니다.
    cols += ["lidar_s0", "lidar_s1", "lidar_s2", "lidar_s3", "lidar_s4"] 
    
    cols += ["scenario", "target_steering", "target_throttle"]
    return cols


# 💡 [추가된 부분: IMU 확장 스키마]
# 수집(collect_data_planner.py)은 이 확장 스키마로 헤더를 쓰고,
# 학습/증강은 기존 csv_columns() 로 검증하므로 양쪽 모두 호환됩니다.
def csv_columns_ext() -> list[str]:
    """기본 스키마 + IMU 파생 컬럼. 수집 단계에서만 사용합니다."""
    return csv_columns() + IMU_COLUMNS


def row_to_tensors(row, device=None, lidar_sectors=None, lidar_clip=LIDAR_CLIP_M):
    """
    Convert a single pandas Series / dict row from the structured CSV (or live feed) into
    model-ready tensors, including the 5-dim LiDAR tensor.

    Returns (objects, lane, lidar, ego, scenario) ready for PlannerModel.forward().
    """
    obj_vals  = [float(row[f"obj{i}_{f}"])
                 for i in range(N_MAX_OBJECTS)
                 for f in ("valid", "class_norm", "conf", "dist_norm",
                           "lat_offset", "width_norm", "height_norm", "lane_overlap")]
    lane_vals = [float(row[f"lane_r{r}c{c}"])
                 for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    
    # 💡 [추가된 부분: 실시간 센서값 vs CSV 저장값 동적 분기 처리]
    # 추론(Inference) 시에는 실시간으로 들어오는 라이다 배열(lidar_sectors)을 즉시 텐서로 변환하고,
    # 오프라인 학습/평가 시에는 CSV 파일에 저장되어 있던 텍스트 컬럼(lidar_s0~s4) 값을 안전하게 읽어옵니다.
    if lidar_sectors is not None:
        lidar_vals = [float(v) for v in lidar_sectors]
    else:
        lidar_vals = [float(row[f"lidar_s{i}"]) for i in range(5)]

    # 💡 [추가] 상한 클리핑. 학습(train_planner)과 반드시 같은 값이어야 한다.
    # lidar_clip=0 으로 부르면 비활성 — 클리핑 없이 학습된 구 체크포인트를
    # 분석할 때 쓴다.
    if lidar_clip and lidar_clip > 0:
        lidar_vals = [min(v, lidar_clip) for v in lidar_vals]

    ego_vals  = [float(row["ego_steering"]), float(row["ego_throttle"])]
    scenario  = int(row["scenario"])

    kw = {"device": device} if device else {}

    objects_t  = torch.tensor(obj_vals,   dtype=torch.float32, **kw).unsqueeze(0)
    lane_t     = torch.tensor(lane_vals,  dtype=torch.float32, **kw).unsqueeze(0)
    
    # 💡 [추가된 부분: 라이다 텐서 인스턴스 생성]
    lidar_t    = torch.tensor(lidar_vals, dtype=torch.float32, **kw).unsqueeze(0)  
    
    ego_t      = torch.tensor(ego_vals,   dtype=torch.float32, **kw).unsqueeze(0)
    scenario_t = torch.tensor([scenario], dtype=torch.long,    **kw)

    # 💡 [수정된 부분: 반환값 구조 개편]
    # 원본은 4개(objects, lane, ego, scenario)를 반환했으나, 
    # 모델 입력 규격에 맞춰 lidar_t가 포함된 총 5개의 텐서를 반환하도록 변경되었습니다.
    return objects_t, lane_t, lidar_t, ego_t, scenario_t
