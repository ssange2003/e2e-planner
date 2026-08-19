#!/usr/bin/env python3
"""
Planner Dataset Augmentation (Merged + Hierarchical Intent Filter)
==================================================================
여러 개의 CSV 파일을 읽어와 병합한 뒤, [3단계 계층형 의도 융합 필터]로
라벨(target_steering / target_throttle)을 전처리하고, 구조화된 피처 기반으로
데이터를 증강(Augmentation)합니다.

계층형 의도 융합 필터 (라벨 전처리)
-----------------------------------
1차 필터 (라이다 + 카메라 근접 위협) : s1/s2/s3 최단거리의 rolling-min +
                             접근 변화율로 물리적 위협(장애물 접근)을 감지.
                             obj{i}_valid/dist_norm(YOLO 장애물)로 카메라
                             근접 위협도 함께 판정 (신규 추가).
2차 필터 (기구학적 의도)   : 스로틀 jerk + 정지 상태 판별로 긴급 제동 /
                             재출발 엣지를 분류. 조향은 잔떨림만 스무딩
3차 필터 (차선 가시성 회복): lane grid(72셀) 합으로 카메라가 차선을 다시
                             포착하는 순간을 감지해 탈출(재출발) 엣지 완성.
                             카메라 근접 장애물이 남아있으면 재출발 보류.

엣지로 판정된 프레임은 원본 라벨을 100% 보존하며, 이후의
모터 dead-zone 보간에서도 제외됩니다(보호 마스크 공유).
모든 시계열 연산(rolling/diff/shift)은 _src_idx(파일별) 그룹 안에서만
수행되어 파일 경계 오염이 없습니다.

Augmentation strategies applied
---------------------------------
1. Identity          — original row kept as-is
2. Mirror            — lateral flip (lidar s0<->s4, s1<->s3 포함)
3. Distance noise    — Gaussian noise on dist_norm  (σ = 0.03)
4. Lateral jitter    — small noise on lat_offset and lane features (σ = 0.04)
5. Confidence noise  — noise on object confidence values (σ = 0.05)
6. Object dropout    — randomly zero out 1 object slot
7. Distance scale    — scale all dist_norm values by U(0.85, 1.15)
8. Mirror + noise    — mirror followed by distance noise

Usage
-----
  python augment.py --inputs data/lane.csv data/s_suv.csv \
                    [--output data/augmented_data.csv] \
                    [--seed 42] \
                    [--smooth 5]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from planner_model import (
    N_MAX_OBJECTS, OBJ_FEATURES, LANE_FEATURES, EGO_FEATURES,
    GRID_ROWS, GRID_COLS,
    csv_columns,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tunable thresholds  (한 곳에서만 관리 — 필터와 dead-zone이 반드시 공유)
# ─────────────────────────────────────────────────────────────────────────────

# [실측 확정] ESC breakaway 실측값: 0.85 미만은 실제로 차가 움직이지 않음.
MOTOR_DEAD_ZONE_MAX = 0.85   # 이 미만의 양(+) 스로틀은 모터가 실제로 돌지 않는 구간

# [실측 확정] 실측 데이터에서 0.75 미만 값이 거의 관찰되지 않음.
THROTTLE_IDLE_MAX   = 0.75   # 이 이하 = 완전 정지 의도 (스로틀 0 취급)
# [주의] 라이다 마운트 위치(센서 원점) 기준 거리인지, 범퍼 앞단 기준으로
# 오프셋 보정된 거리인지 미확인. 오프셋이 있다면 실제 위협 거리가 이 값과
# 다를 수 있음 — 마운트 위치 실측 후 검증 필요.
LIDAR_DANGER_M      = 0.30   # 전방/전측방 장애물 즉각 위협 거리 [m]
LIDAR_CLEAR_M       = 0.50   # 재출발 시 "길이 열렸다"고 볼 최소 거리 [m]
# [논리 추론] 게임패드 트리거/스틱은 페달과 달리 해제가 near-binary라
# 진짜 패닉 브레이크는 1~2프레임 내 -0.3~-0.9 수준으로 떨어짐. 정상 주행
# 노이즈(±0.05~0.09)와의 간격이 크므로, 오탐 마진 확보를 위해 노이즈
# 상한에서 더 떨어뜨림. 실측 로그로 노이즈 분산 재확인 시 조정 가능.
THROTTLE_JERK_BRAKE = -0.20  # 프레임 간 스로틀 급감 → 제동 변곡점
# [실측 확정] 차선이 명확히 보이는 정상 프레임의 스크린샷 실측 합계가
# 약 1.24(가려진 셀 포함 추정 1.5~2.0)로 관측됨. 기존 2.0은 이 관측값과
# 거의 같거나 높아서 정상 프레임도 "차선 안 보임"으로 오판정할 위험이 있어
# 여유를 두고 하향.
LANE_VIS_THRESH     = 1.0    # lane grid 72셀 합이 이 값 초과 → 차선 가시

# [실측 확정] planner_model.py: dist_norm = 실제거리(m) / MAX_DIST_M(5.0)
# 라이다와 동일한 물리 거리 기준(LIDAR_DANGER_M=0.30m)으로 정확히 환산.
CAMERA_OBJ_DANGER_NORM = 0.06  # = 0.30m / 5.0 (LIDAR_DANGER_M과 통일)


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 계층형 의도 융합 필터  (라벨 전처리 — 병합 직후, _src_idx 살아있을 때 호출)
# ─────────────────────────────────────────────────────────────────────────────

def apply_hierarchical_intent_filter(df: pd.DataFrame,
                                     smooth_window: int = 5,
                                     group_col: str = "_src_idx") -> pd.DataFrame:
    """
    [3단계 계층형 의도 융합 라벨 전처리]
    - 1차: 라이다(s1,s2,s3) 시계열 rolling-min + 카메라(obj valid/dist_norm)
           근접 위협으로 물리적 위협 감지 (라이다·카메라 = 메인 판단 주체)
    - 2차: 스로틀 jerk/정지 상태로 긴급 제동·재출발 엣지 판별, 조향 잔떨림 스무딩
           (throttle/steering = 서브, 타이밍 보정용)
    - 3차: lane grid 가시성 회복 + 라이다 클리어런스 + 카메라 클리어로
           탈출 엣지 완성

    엣지 프레임의 target_throttle 원본을 보존하고, 후속 dead-zone 보간에서
    제외할 수 있도록 df["_edge_protect"] (0/1) 컬럼을 남긴다.
    """
    if group_col not in df.columns:
        df = df.copy()
        df[group_col] = 0

    # ── [1단계: 라이다 시계열 퍼셉션 필터] ────────────────────────────────
    # s1(좌전방), s2(정면), s3(우전방)의 최단거리를 파일별로 rolling-min.
    # 센서 출력은 미터 단위, 결측/무한대는 5.0m로 채워져 들어온다.
    lidar_cols = ["lidar_s1", "lidar_s2", "lidar_s3"]
    if all(c in df.columns for c in lidar_cols):
        dist_min = df[lidar_cols].min(axis=1)
        dist_smooth = (
            dist_min.groupby(df[group_col], group_keys=False)
                    .apply(lambda s: s.rolling(smooth_window, center=True,
                                               min_periods=1).min())
        )
        # 접근 변화율(음수 = 장애물이 가까워지는 중) — 파일 경계에서 diff 차단
        dist_delta = dist_min.groupby(df[group_col]).diff().fillna(0.0)
    else:
        dist_smooth = pd.Series(999.0, index=df.index)
        dist_delta  = pd.Series(0.0, index=df.index)

    is_closing_in = dist_delta < 0.0   # 장애물 접근 중

    # ── [1단계 확장: 카메라 장애물(YOLO) 근접 위협] ──────────────────────
    # lane grid(차선 가시성)와는 별개 채널. obj{i}_valid + dist_norm으로
    # "카메라가 가까운 장애물을 실제로 보고 있는가"를 직접 판정한다.
    obj_valid_cols = [f"obj{i}_valid" for i in range(N_MAX_OBJECTS)]
    obj_dist_cols  = [f"obj{i}_dist_norm" for i in range(N_MAX_OBJECTS)]

    if all(c in df.columns for c in obj_valid_cols + obj_dist_cols):
        valid_mat = df[obj_valid_cols].to_numpy() > 0.5
        dist_mat  = df[obj_dist_cols].to_numpy()
        camera_obstacle_threat = pd.Series(
            (valid_mat & (dist_mat < CAMERA_OBJ_DANGER_NORM)).any(axis=1),
            index=df.index,
        )
    else:
        camera_obstacle_threat = pd.Series(False, index=df.index)

    # ── [2단계: 조향 및 스로틀 궤적 / 기구학적 의도 필터] ─────────────────
    steering_raw = df["target_steering"].copy()
    throttle_raw = df["target_throttle"].copy()

    # 조향: 잔떨림만 스무딩 (augment() 본문의 조향 스무딩은 제거됨 — 여기가 유일)
    if smooth_window > 1:
        df["target_steering"] = (
            steering_raw.groupby(df[group_col], group_keys=False)
                        .apply(lambda s: s.rolling(smooth_window, center=True,
                                                   min_periods=1).mean())
                        .clip(-1.0, 1.0)
        )

    # 스로틀 jerk / 정지 상태 — 모두 파일별로 계산
    throttle_jerk    = throttle_raw.groupby(df[group_col]).diff().fillna(0.0)
    prev_throttle    = throttle_raw.groupby(df[group_col]).shift(1).fillna(0.0)
    is_stopped_state = prev_throttle < THROTTLE_IDLE_MAX

    # 긴급 제동 엣지: 라이다/카메라가 위협을 확정한 상태에서
    # 스로틀을 급격히 떨구는 변곡점.
    # - 즉각 위험(<0.40m): 접근 여부 무관, 거리 자체가 위협
    # - 경계 구간(0.40~0.65m): "접근 중"일 때만 위협 (정지 장애물 스침 오탐 방지)
    #   ※ 기존 코드는 LIDAR_DANGER_M < LIDAR_CLEAR_M라서 이 구간 분리가
    #      안 돼 있었고, is_closing_in이 판정에 전혀 영향을 못 주는
    #      죽은 조건이었음 — 아래에서 분리해 실제로 작동하게 함.
    is_immediate_danger = (
        (dist_smooth < LIDAR_DANGER_M) & (throttle_jerk < THROTTLE_JERK_BRAKE)
    )
    is_approaching_zone = (
        is_closing_in
        & (dist_smooth < LIDAR_CLEAR_M)
        & (throttle_jerk < THROTTLE_JERK_BRAKE)
    )
    is_emergency_braking = (
        is_immediate_danger | is_approaching_zone | camera_obstacle_threat
    )

    # ── [신규] 순간 0값(손 떨림) 필터 ──────────────────────────────────
    # 정지 의도가 아닌데 손이 미끄러져 짧게 0 근처로 찍히는 경우를
    # 실제 정지(장애물/의도적 정차)와 구분한다.
    # [주의] 30 FPS 로깅 가정 → 2초 = 60프레임. 실측 데이터로 "의도적
    # 2초 정지" 케이스가 오탐되지 않는지 검증 필요(손 떨림은 보통 1~5프레임).
    TRANSIENT_ZERO_MAX_FRAMES = 60   # 2초 @ 30 FPS
    # [변경] 0.05 → 0.75. 더 이상 "0에 가까운 값"만이 아니라 "정상 주행
    # 임계 미만 전체(저속 포함)"를 손 떨림 후보로 본다는 뜻이라 이름을
    # LOW_THROTTLE_NOISE_THRESH로 명확히 함. THROTTLE_IDLE_MAX(0.6)보다도
    # 높아서, 이전에 "정지 의도"로 분류되던 0.6~0.75 구간까지 이 필터가
    # 건드리게 됨 — 저속 주행 구간이 통째로 보간될 위험이 커졌다는 뜻.
    LOW_THROTTLE_NOISE_THRESH = 0.75

    def _zero_run_length(s: pd.Series) -> pd.Series:
        is_zero = s < LOW_THROTTLE_NOISE_THRESH
        change = is_zero.ne(is_zero.shift()).cumsum()
        run_len = is_zero.groupby(change).transform('size')
        return run_len.where(is_zero, 0)

    zero_run_len = (
        throttle_raw.groupby(df[group_col], group_keys=False)
                    .apply(_zero_run_length)
    )

    # 물리적 근거(라이다/카메라 위협)가 없는데 짧게(<=2초)만 0이면 → 노이즈
    is_transient_zero = (
        (zero_run_len > 0) & (zero_run_len <= TRANSIENT_ZERO_MAX_FRAMES)
        & (~camera_obstacle_threat)
        & (dist_smooth > LIDAR_CLEAR_M)
    )

    # 노이즈 프레임은 "정지 상태"로 취급하지 않음 → 다음 프레임의 허위 recovery_launch 방지
    is_stopped_state = is_stopped_state & (~is_transient_zero.shift(1, fill_value=False))

    # 노이즈 프레임 자체는 선형 보간으로 채움 (엣지 보호 대상과 무관하게 먼저 정리)
    throttle_raw = throttle_raw.mask(is_transient_zero)
    throttle_raw = (
        throttle_raw.groupby(df[group_col])
                    .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )

    n_transient = int(is_transient_zero.sum())
    print(f"[AUG] 손 떨림성 순간 0값 필터(<= {TRANSIENT_ZERO_MAX_FRAMES}프레임/2초): "
          f"{n_transient}개 프레임 보간 처리")

    # ── [3단계: 카메라 차선 가시성 회복 필터] ────────────────────────────
    # obj{i}_valid는 YOLO '장애물 검출' 플래그이므로 차선 가시성에 쓰지 않는다.
    # 차선 가시성은 lane grid 72셀(각 셀 = 차선 픽셀 비율)의 합으로 판단.
    lane_cols = [f"lane_r{r}c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    if all(c in df.columns for c in lane_cols):
        cam_visibility = (df[lane_cols].sum(axis=1) > LANE_VIS_THRESH).astype(float)
    else:
        cam_visibility = pd.Series(1.0, index=df.index)

    lane_reappearing = cam_visibility.groupby(df[group_col]).diff().fillna(0.0) > 0

    # 재출발(탈출) 엣지: 멈춰 있던 상태에서 (차선 회복 OR 라이다 클리어)와 함께
    # 모터가 실제로 돌기 시작하는 breakaway 이상으로 스로틀을 감아올리는 순간.
    # 카메라가 여전히 근접 장애물을 보고 있으면 재출발 보류 (신규 조건).
    is_accelerating   = throttle_raw > MOTOR_DEAD_ZONE_MAX
    is_recovery_launch = (
        is_stopped_state & is_accelerating
        & (lane_reappearing | (dist_smooth > LIDAR_CLEAR_M))
        & (~camera_obstacle_threat)
    )

    # ── 종합 엣지 보호 마스크 ────────────────────────────────────────────
    # 엣지 이웃 프레임까지 보호를 확장(±(window//2)) — 스무딩 윈도우가
    # 엣지 직전/직후 프레임을 통해 엣지 값을 간접 훼손하는 것을 방지.
    is_edge_intent = (is_emergency_braking | is_recovery_launch)
    pad = max(1, smooth_window // 2)
    is_edge_protected = (
        is_edge_intent.astype(float)
                      .groupby(df[group_col], group_keys=False)
                      .apply(lambda s: s.rolling(2 * pad + 1, center=True,
                                                 min_periods=1).max())
    ) > 0.5

    # 일반 주행 구간의 스로틀을 0.94 "바닥값"으로 보정 (고정 오버라이드 아님)
    # ─────────────────────────────────────────────────────────────────
    # [설계 변경] 이전엔 정상 주행 프레임 전부를 0.94로 덮어써서 코너/직선
    # 간 사람의 실제 속도 변화(=MSE 회귀가 배워야 할 신호)가 소실됐음.
    # 이제는 0.94 미만인 소극적 프레임만 끌어올리고, 0.94 이상인 값(사람이
    # 더 강하게 밟은 구간)은 원본을 그대로 보존해 속도 변화 정보를 유지함.
    NORMAL_DRIVE_BOOST_FLOOR = 0.94
    is_normal_driving = (~is_emergency_braking) & (~is_stopped_state) & (throttle_raw > THROTTLE_IDLE_MAX)
    is_below_floor = is_normal_driving & (throttle_raw < NORMAL_DRIVE_BOOST_FLOOR)

    # np.where 대신 pandas의 mask나 where를 사용하여 Series 형식 보존
    throttle_raw = throttle_raw.mask(is_below_floor, NORMAL_DRIVE_BOOST_FLOOR)

    # ── 조건부 스로틀 스무딩: 엣지는 원본 사수, 일반 구간만 평활화 ────────
    if smooth_window > 1:
        smoothed_throttle = (
            throttle_raw.groupby(df[group_col], group_keys=False)
                        .apply(lambda s: s.rolling(smooth_window, center=True,
                                                   min_periods=1).mean())
        )
        df["target_throttle"] = np.where(is_edge_protected,
                                       throttle_raw, smoothed_throttle)
    else:
        df["target_throttle"] = throttle_raw

    # dead-zone 보간 단계와 공유할 보호 마스크
    df["_edge_protect"] = is_edge_protected.astype(int)

    # ─────────────────────────────────────────────────────────────────
    # [출력 위치] 필터링 결과 및 0.94 오버라이드 통계 프린트
    # ─────────────────────────────────────────────────────────────────
    n_brake    = int(is_emergency_braking.sum())
    n_launch   = int(is_recovery_launch.sum())
    n_prot     = int(is_edge_protected.sum())
    n_override = int(is_below_floor.sum())
    n_cam_threat = int(camera_obstacle_threat.sum())

    print(f"[AUG] 계층형 의도 융합: 긴급제동 {n_brake} / 재출발 {n_launch} "
          f"→ 보호 프레임 {n_prot} (이웃 ±{pad} 포함)")
    print(f"[AUG] 카메라 근접 위협 판정 프레임 수: {n_cam_threat}")
    print(f"[AUG] 일반 주행 0.94 미만 → floor 보정 적용된 프레임 수: {n_override}")
    # ─────────────────────────────────────────────────────────────────
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Column index helpers  (build once from the schema)
# ─────────────────────────────────────────────────────────────────────────────

_COLS = csv_columns()

def _col(name: str) -> int:
    return _COLS.index(name)

def _obj_col(slot: int, feat: str) -> int:
    return _col(f"obj{slot}_{feat}")

OBJ_VALID_COLS   = [_obj_col(i, "valid")        for i in range(N_MAX_OBJECTS)]
OBJ_LAT_COLS     = [_obj_col(i, "lat_offset")   for i in range(N_MAX_OBJECTS)]
OBJ_DIST_COLS    = [_obj_col(i, "dist_norm")    for i in range(N_MAX_OBJECTS)]
OBJ_CONF_COLS    = [_obj_col(i, "conf")         for i in range(N_MAX_OBJECTS)]
OBJ_W_COLS       = [_obj_col(i, "width_norm")   for i in range(N_MAX_OBJECTS)]
OBJ_H_COLS       = [_obj_col(i, "height_norm")  for i in range(N_MAX_OBJECTS)]
OBJ_OVERLAP_COLS = [_obj_col(i, "lane_overlap") for i in range(N_MAX_OBJECTS)]

LANE_GRID_COLS = [
    [_col(f"lane_r{r}c{c}") for c in range(GRID_COLS)]
    for r in range(GRID_ROWS)
]

TARGET_STEER_COL = _col("target_steering")
TARGET_THTL_COL  = _col("target_throttle")
EGO_STEER_COL    = _col("ego_steering")

ALL_OBJ_COLS = []
for i in range(N_MAX_OBJECTS):
    for f in ("valid", "class_norm", "conf", "dist_norm",
              "lat_offset", "width_norm", "height_norm", "lane_overlap"):
        ALL_OBJ_COLS.append(_obj_col(i, f))


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation functions  (all operate on a numpy row vector)
# ─────────────────────────────────────────────────────────────────────────────

def _clone(row: np.ndarray) -> np.ndarray:
    return row.copy()


def aug_identity(row: np.ndarray, rng) -> np.ndarray:
    return _clone(row)


def aug_mirror(row: np.ndarray, rng) -> np.ndarray:
    r = _clone(row)

    # 라이다 좌우 반전: s0 <-> s4, s1 <-> s3
    l_cols = [_col(f"lidar_s{i}") for i in range(5)]
    vals = [r[c] for c in l_cols]
    for i in range(5):
        r[l_cols[i]] = vals[4 - i]

    for c in OBJ_LAT_COLS:
        r[c] = -r[c]
    for gr in range(GRID_ROWS):
        vals = [r[LANE_GRID_COLS[gr][c]] for c in range(GRID_COLS)]
        for c in range(GRID_COLS):
            r[LANE_GRID_COLS[gr][c]] = vals[GRID_COLS - 1 - c]

    r[TARGET_STEER_COL] = -r[TARGET_STEER_COL]
    r[EGO_STEER_COL]    = -r[EGO_STEER_COL]
    return r


def aug_distance_noise(row: np.ndarray, rng, sigma: float = 0.03) -> np.ndarray:
    r = _clone(row)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_DIST_COLS[i]] = float(np.clip(
                r[OBJ_DIST_COLS[i]] + rng.normal(0, sigma), 0.0, 1.0))
    return r


def aug_lateral_jitter(row: np.ndarray, rng, sigma: float = 0.04) -> np.ndarray:
    r = _clone(row)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_LAT_COLS[i]] += rng.normal(0, sigma)
    shift = rng.choice([-1, 0, 0, 1])
    if shift != 0:
        for gr in range(GRID_ROWS):
            vals = [r[LANE_GRID_COLS[gr][c]] for c in range(GRID_COLS)]
            for c in range(GRID_COLS):
                src = c - shift
                r[LANE_GRID_COLS[gr][c]] = vals[src] if 0 <= src < GRID_COLS else 0.0
    return r


def aug_confidence_noise(row: np.ndarray, rng, sigma: float = 0.05) -> np.ndarray:
    r = _clone(row)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_CONF_COLS[i]] = float(np.clip(
                r[OBJ_CONF_COLS[i]] + rng.normal(0, sigma), 0.0, 1.0))
    return r


def aug_object_dropout(row: np.ndarray, rng, drop_prob: float = 0.25) -> np.ndarray:
    r = _clone(row)
    valid_slots = [i for i in range(N_MAX_OBJECTS) if r[OBJ_VALID_COLS[i]] > 0.5]
    if not valid_slots:
        return r
    drop_slot = rng.choice(valid_slots)
    for j in range(OBJ_FEATURES):
        r[ALL_OBJ_COLS[drop_slot * OBJ_FEATURES + j]] = 0.0
    return r


def aug_distance_scale(row: np.ndarray, rng,
                       low: float = 0.85, high: float = 1.15) -> np.ndarray:
    r     = _clone(row)
    scale = rng.uniform(low, high)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_DIST_COLS[i]] = float(np.clip(r[OBJ_DIST_COLS[i]] * scale, 0.0, 1.0))
    return r


def aug_mirror_and_noise(row: np.ndarray, rng) -> np.ndarray:
    return aug_distance_noise(aug_mirror(row, rng), rng)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

AUGMENTATIONS = [
    ("identity",         aug_identity,         1),
    ("mirror",           aug_mirror,           1),
    ("distance_noise",   aug_distance_noise,   1),
    ("lateral_jitter",   aug_lateral_jitter,   1),
    ("confidence_noise", aug_confidence_noise, 1),
    ("object_dropout",   aug_object_dropout,   1),
    ("distance_scale",   aug_distance_scale,   1),
    ("mirror_noise",     aug_mirror_and_noise, 1),
]


def augment(input_files: list, output_csv: Path, seed: int = 42,
            smooth_window: int = 5) -> None:
    print("=" * 65)
    print("  Planner Dataset Load & Merge & Augment")
    print("=" * 65)
    print(f"  Output : {output_csv}")
    print(f"  Seed   : {seed}")
    print()

    # 1. 파일 다중 로드 및 병합
    df_list = []
    total_new_rows = 0

    for i, file_path_str in enumerate(input_files):
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"[WARN] Input CSV not found, skipping: {file_path}")
            continue

        temp_df = pd.read_csv(file_path, on_bad_lines='warn')
        temp_df = temp_df.dropna().reset_index(drop=True)

        # 파일 간 시계열/보간 오염을 막기 위한 고유 인덱스 부여
        temp_df["_src_idx"] = i
        total_new_rows += len(temp_df)
        df_list.append(temp_df)

    if not df_list:
        print("\n[ERROR] No valid data found to augment.")
        return

    df = pd.concat(df_list, ignore_index=True)
    print(f"\n[MERGE] Total combined rows: {total_new_rows}")
    print("-" * 65)

    # 2. Schema 검증
    expected = set(csv_columns())
    missing  = expected - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in combined CSV: {missing}")
        return

    # ── 3단계 계층형 의도 융합 라벨 전처리 ───────────────────────────────
    # 조향 스무딩·조건부 스로틀 스무딩·엣지 보호 마스크를 한 번에 처리.
    # (기존의 별도 steering smoothing 블록은 이중 스무딩이라 제거됨)
    if smooth_window > 1:
        steer_raw_std = df["target_steering"].std()
        thtl_raw_std  = df["target_throttle"].std()

    df = apply_hierarchical_intent_filter(df, smooth_window, group_col="_src_idx")

    if smooth_window > 1:
        print(f"[AUG] Label smoothing (window={smooth_window}):")
        print(f"      Steer std {steer_raw_std:.4f} → {df['target_steering'].std():.4f}")
        print(f"      Thtl  std {thtl_raw_std:.4f} → {df['target_throttle'].std():.4f}")
        print()

    # ── 모터 죽은 구간(dead zone) 스무딩 ──
    # 주행 의도는 있지만(throttle > 0.05) ESC breakaway(MOTOR_DEAD_ZONE_MAX)에
    # 못 미쳐 차가 사실상 멈춰 있는 구간을 선형 보간으로 램프 처리.
    # 단, 계층형 필터가 보호한 엣지 프레임(긴급제동/재출발의 과도 구간)은
    # 인간의 원본 의도이므로 보간 대상에서 제외한다.
    dead_zone_mask = (
        (df["target_throttle"] > THROTTLE_IDLE_MAX)
        & (df["target_throttle"] < MOTOR_DEAD_ZONE_MAX)
        & (df["_edge_protect"] < 0.5)          # ← 엣지 보호 프레임 제외
    )
    n_dead = int(dead_zone_mask.sum())

    if n_dead > 0:
        masked = df["target_throttle"].mask(dead_zone_mask)
        df["target_throttle"] = (
            masked.groupby(df["_src_idx"])
                  .transform(lambda s: s.interpolate(method="linear",
                                                     limit_direction="both"))
        )

    print(f"[AUG] 모터 죽은 구간({THROTTLE_IDLE_MAX} < throttle < {MOTOR_DEAD_ZONE_MAX}) "
          f"스무딩: {n_dead}개 행 보간 처리 (엣지 보호 프레임 제외)")
    print("-" * 65)

    # 내부 작업 컬럼 제거 (스키마 밖 컬럼이 증강/저장으로 새지 않게)
    df = df.drop(columns=["_src_idx", "_edge_protect"])

    # [증강 전 원본 데이터 병합본 별도 저장]
    merged_csv_path = output_csv.parent / "planner_data.csv"
    df["frame_id"] = np.arange(1, len(df) + 1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(merged_csv_path, index=False)
    print(f"[SAVE] Merged original data saved to → {merged_csv_path}\n")

    # 4. 증강 (Augmentation)
    rng = np.random.default_rng(seed)
    cols_ordered = csv_columns()
    data_np      = df[cols_ordered].to_numpy(dtype=float)

    aug_rows = []
    for orig_row in data_np:
        for name, fn, weight in AUGMENTATIONS:
            for _ in range(weight):
                aug_rows.append(fn(orig_row, rng))

    aug_np = np.stack(aug_rows, axis=0)
    aug_df = pd.DataFrame(aug_np, columns=cols_ordered)

    # 후처리 및 클리핑
    aug_df["frame_id"] = np.arange(1, len(aug_df) + 1)

    for i in range(N_MAX_OBJECTS):
        aug_df[f"obj{i}_valid"]        = aug_df[f"obj{i}_valid"].clip(0, 1).round()
        aug_df[f"obj{i}_conf"]         = aug_df[f"obj{i}_conf"].clip(0, 1)
        aug_df[f"obj{i}_dist_norm"]    = aug_df[f"obj{i}_dist_norm"].clip(0, 1)
        aug_df[f"obj{i}_lane_overlap"] = aug_df[f"obj{i}_lane_overlap"].clip(0, 1)
        aug_df[f"obj{i}_width_norm"]   = aug_df[f"obj{i}_width_norm"].clip(0, 1)
        aug_df[f"obj{i}_height_norm"]  = aug_df[f"obj{i}_height_norm"].clip(0, 1)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            aug_df[f"lane_r{r}c{c}"] = aug_df[f"lane_r{r}c{c}"].clip(0, 1)
    aug_df["target_steering"] = aug_df["target_steering"].clip(-1, 1)
    aug_df["target_throttle"] = aug_df["target_throttle"].clip(-1, 1)
    aug_df["scenario"] = aug_df["scenario"].round().astype(int)

    # 파일 저장
    aug_df.to_csv(output_csv, index=False)

    n_aug  = len(aug_df)
    n_orig = len(df)
    print(f"[RESULT] Augmented: {n_orig} → {n_aug} rows  (×{n_aug/n_orig:.1f})")
    print(f"[RESULT] Saved to → {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir / "data"

    parser = argparse.ArgumentParser(
        description="Load, merge, and augment the structured planner dataset")
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='Input CSV files to merge and augment')
    parser.add_argument('--output', type=Path,
                        default=data_dir / "augmented_data.csv",
                        help='Output CSV (default: data/augmented_data.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Rolling window for label smoothing (1 = disable)')
    args = parser.parse_args()

    augment(args.inputs, args.output, args.seed, args.smooth)
