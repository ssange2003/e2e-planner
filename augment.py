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

# [실측 확정] 정확히 0.0(진짜 정지/노이즈 후보)과 0.75+(정상 주행) 사이가
# 이분법적으로 갈리고, 중간값은 거의 없음(course1~3 실측 확인).
THROTTLE_IDLE_MAX   = 0.73   # 이 이하 = 정지 상태로 취급 (baseline 계산 등에 사용)

# [주의] 라이다 마운트 위치(센서 원점) 기준 거리인지, 범퍼 앞단 기준으로
# 오프셋 보정된 거리인지 미확인. 오프셋이 있다면 실제 위협 거리가 이 값과
# 다를 수 있음 — 마운트 위치 실측 후 검증 필요.
LIDAR_DANGER_M      = 0.30   # 절대 위협 거리 [m] — 이보다 가까우면 코스 불문 위협
LIDAR_CLEAR_M       = 0.50   # 절대 상한 [m] — 상대판정도 이 안에서만 유효(과도 오탐 방지)

# [실측 확정] planner_model.py: dist_norm = 실제거리(m) / MAX_DIST_M(5.0)
# 라이다와 동일한 물리 거리 기준(LIDAR_DANGER_M=0.30m)으로 정확히 환산.
CAMERA_OBJ_DANGER_NORM = 0.06  # = 0.30m / 5.0 (LIDAR_DANGER_M과 통일)

# ── [신규] Zero-event 기반 판정 상수 ────────────────────────────────
# [실측 확정] course1(전량 노이즈 확정)/course3(회피기동) 대조 검증 완료.
# 코스마다 라이다 baseline 거리가 완전히 다름(개활 코스 4m대 vs 협로 0.5m대)
# 이라 절대값 하나로는 위협을 못 가림 — 그룹별 상대 baseline 비율로 보정.
ZERO_EVENT_THRESH   = 0.10   # 이 미만만 "zero event" 후보 (0.75는 노이즈 판정선으로 부적합했음)
LIDAR_RATIO         = 0.6    # baseline 대비 이 비율 미만 & LIDAR_CLEAR_M 이내면 위협으로 인정
CONTEXT_PAD         = 5      # zero event 전후 컨텍스트 프레임 수 (라이다/카메라 최소거리 탐색 범위)

# [신규][논리 추론, 미검증] 접근 변화율 — "느리지만 꾸준히 다가오는" 장애물은
# 절대/상대 거리 조건에 안 걸릴 수 있음(예: 0.80→0.45m로 좁아져도 0.45>
# LIDAR_CLEAR_M(0.50)에 안 걸림 — 정지 직전에야 겨우 걸림). 더 넓은 윈도우로
# 추세 자체를 봐서 조기 포착. 실측 데이터로 APPROACH_DROP_M 재검증 필요.
APPROACH_TREND_WINDOW = 15   # 0.5초 @ 30fps — 최소거리 탐색(±5)보다 넓게
APPROACH_DROP_M       = 0.15 # 이 윈도우 안에서 이만큼 좁아지면 "접근 중"으로 판정

# [신규][미검증] lane visibility 보조 증거 — 문서 지적대로 이전 버전에서
# 실제 미구현이었던 부분. 정지 원인 증거로는 라이다/카메라보다 약한 신호라
# ratio를 낮게(엄격하게) 잡아 과탐 방지.
LANE_VIS_DROP_RATIO  = 0.4   # baseline 대비 이 비율 미만으로 차선 신호가 줄면 보조 증거로 인정

# [실측 확정] steering은 게임패드 디지털 입력이라 {0, +0.9, -0.9}에 93%가
# 몰려 있음(course1/3 실측) — "코너링"과 "회피조향"이 값으로 구분 안 됨.
# 그래서 조향은 증거 판정에서 제외하고, 라이다·카메라 증거만으로 판정한다.


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 계층형 의도 융합 필터  (라벨 전처리 — 병합 직후, _src_idx 살아있을 때 호출)
# ─────────────────────────────────────────────────────────────────────────────

def apply_hierarchical_intent_filter(df: pd.DataFrame,
                                     smooth_window: int = 5,
                                     group_col: str = "_src_idx") -> pd.DataFrame:
    """
    [Zero-event 기반 계층형 의도 필터]

    기존의 "throttle < 0.75 = 노이즈 후보 / 2초 캡 / steering excursion" 방식은
    실측 검증 결과 다음 문제가 확인되어 폐기됨:
      - 0.75는 실제 노이즈 판정선이 아니라 정지/주행 이분법 경계일 뿐이었음
        (0~0.75 사이 매끄러운 전환값을 노이즈로 오인해 삭제할 위험)
      - 고정 프레임 캡(60)은 물리적 근거와 무관해 진짜 정지도 지울 수 있었음
      - steering excursion은 게임패드가 디지털 입력({0,+0.9,-0.9})이라
        코너링과 회피조향을 구분 못 함 (course1 확정 노이즈에서도 0.9 관측됨)

    새 방식: throttle이 거의 0(< ZERO_EVENT_THRESH)인 연속 구간을
    "zero event"로 묶고, 이벤트 단위로 라이다/카메라 물리적 증거를 검사한다.
      - 증거 있음(라이다 절대/상대 위협 또는 카메라 위협) → 원본 보존
      - 파일 시작/끝에 걸침(보간 불가) → 원본 보존
      - 증거 없음 & 경계 아님 → 노이즈로 판정, 선형보간

    course1(전량 노이즈로 확정된 실측 데이터)과 course3(장애물 회피 실측
    데이터)로 교차검증 완료 — course1의 비경계 이벤트 전부가 정확히
    노이즈로, course3는 라이다 증거 유무로 내부 일관되게 분류됨.

    라이다 baseline은 그룹(파일)별로 다르므로(개활 코스 4m대 vs 협로
    0.5m대) 상대 비율(LIDAR_RATIO)과 절대 상한(LIDAR_CLEAR_M)을 함께
    적용해 코스 스케일에 관계없이 동작하도록 함.

    모든 시계열 연산은 group_col(파일별) 안에서만 수행되어 파일 경계
    오염이 없다.
    """
    if group_col not in df.columns:
        df = df.copy()
        df[group_col] = 0
    grp = df[group_col]

    # ── 라이다 시계열 (rolling-min으로 센서 노이즈 스무딩) ────────────
    lidar_cols = ["lidar_s1", "lidar_s2", "lidar_s3"]
    if all(c in df.columns for c in lidar_cols):
        dist_min = df[lidar_cols].min(axis=1)
        dist_smooth = (
            dist_min.groupby(grp, group_keys=False)
                    .apply(lambda s: s.rolling(smooth_window, center=True,
                                               min_periods=1).min())
        )
    else:
        dist_smooth = pd.Series(999.0, index=df.index)
    # ── 차선 가시성 (lane grid 72셀 합) — 보조 증거 및 recovery 신호용 ────
    # [주의] 문서 지적대로 이전 버전에서 실제 미구현이었던 부분. 라이다/
    # 카메라보다 약한 신호로 취급(LANE_VIS_DROP_RATIO를 낮게 잡음).
    lane_cols = [f"lane_r{r}c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    if all(c in df.columns for c in lane_cols):
        lane_sum = df[lane_cols].sum(axis=1)
    else:
        lane_sum = pd.Series(np.nan, index=df.index)
    # ── 카메라 장애물(YOLO) 근접 위협 ─────────────────────────────────
    # [주의] 실측 4개 코스 전부에서 obj_valid가 거의(0~5/4190) 발생하지
    # 않아 이 신호는 현재 실질적으로 비활성 상태. 향후 카메라 파이프라인이
    # 개선되면 자동으로 다시 유효해지도록 로직은 유지.
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

    # ── 조향: 잔떨림만 스무딩 (증거 판정에는 미사용 — 위 상수 설명 참고) ──
    steering_raw = df["target_steering"].copy()
    throttle_raw = df["target_throttle"].copy()
    if smooth_window > 1:
        df["target_steering"] = (
            steering_raw.groupby(grp, group_keys=False)
                        .apply(lambda s: s.rolling(smooth_window, center=True,
                                                   min_periods=1).mean())
                        .clip(-1.0, 1.0)
        )

    # ── [1단계] Zero event 검출 및 그룹별 baseline 계산 ────────────────
    is_near_zero = throttle_raw < ZERO_EVENT_THRESH
    run_change = (
        is_near_zero.groupby(grp, group_keys=False)
                    .apply(lambda s: s.ne(s.shift()).cumsum())
    )
    event_key = grp.astype(str) + "_" + run_change.astype(str)
    event_key = event_key.where(is_near_zero)

    # 그룹(파일)별 "정상 주행" 라이다/차선 baseline — 코스 스케일 보정용
    driving_mask = throttle_raw > THROTTLE_IDLE_MAX
    baseline_by_group = (
        dist_smooth[driving_mask].groupby(grp[driving_mask]).median()
    )
    lane_baseline_by_group = (
        lane_sum[driving_mask].groupby(grp[driving_mask]).median()
    )

    is_transient_noise = pd.Series(False, index=df.index)
    is_event_evidence  = pd.Series(False, index=df.index)
    is_event_boundary  = pd.Series(False, index=df.index)

    # 이벤트 단위 판정 (그룹별 순회 — 데이터 규모상 파이썬 루프로 충분히 빠름)
    for gid, sub in df.groupby(grp):
        idx = sub.index
        gsize = len(idx)
        base_d = baseline_by_group.get(gid, LIDAR_CLEAR_M)
        base_lane = lane_baseline_by_group.get(gid, np.nan)
        ev_ids = event_key.loc[idx].dropna().unique()
        for eid in ev_ids:
            ev_idx = idx[event_key.loc[idx] == eid]
            local_s = idx.get_loc(ev_idx.min())
            local_e = idx.get_loc(ev_idx.max())
            is_boundary = (local_s == 0) or (local_e == gsize - 1)

            ps = idx[max(0, local_s - CONTEXT_PAD)]
            pe = idx[min(gsize - 1, local_e + CONTEXT_PAD)]
            dmin = dist_smooth.loc[ps:pe].min()
            cam_hit = bool(camera_obstacle_threat.loc[ps:pe].any())

            # 접근 변화율: 이벤트 시작 이전 더 넓은 윈도우(±더 큼)에서
            # 거리가 꾸준히 좁혀졌는지 확인 — 절대/상대 거리에 안 걸려도
            # "느리지만 다가오는" 장애물을 조기에 잡기 위함.
            trend_s = idx[max(0, local_s - APPROACH_TREND_WINDOW)]
            pre_window_start_dist = dist_smooth.loc[trend_s]
            approach_drop = pre_window_start_dist - dmin
            is_approaching = approach_drop > APPROACH_DROP_M

            # 차선 가시성 보조 증거 (약한 신호 — 단독으로는 과탐 방지 위해
            # 엄격한 비율만 인정). [실측 확정] ±CONTEXT_PAD 윈도우+최소값
            # 방식은 이벤트와 무관한 인접 프레임의 일시적 카메라 글리치를
            # 잡아내 오탐을 냄(course1 frame 25-26 검증에서 확인) — 반드시
            # 이벤트 구간 자체의 평균으로 좁혀야 함.
            lane_hit = False
            if not np.isnan(base_lane) and base_lane > 0:
                lane_event_mean = lane_sum.loc[ev_idx].mean()
                lane_hit = lane_event_mean < LANE_VIS_DROP_RATIO * base_lane

            evidence = (
                (dmin < LIDAR_DANGER_M)
                or (dmin < LIDAR_RATIO * base_d and dmin < LIDAR_CLEAR_M)
                or (is_approaching and dmin < LIDAR_CLEAR_M)
                or cam_hit
                or lane_hit
            )

            if is_boundary:
                is_event_boundary.loc[ev_idx] = True
            elif evidence:
                is_event_evidence.loc[ev_idx] = True
            else:
                is_transient_noise.loc[ev_idx] = True

    # ── 노이즈로 판정된 이벤트만 선형보간 (증거/경계 이벤트는 원본 보존) ──
    throttle_raw = throttle_raw.mask(is_transient_noise)
    throttle_raw = (
        throttle_raw.groupby(grp)
                    .transform(lambda s: s.interpolate(method="linear",
                                                       limit_direction="both"))
    )

    # ── [2단계] 재출발(recovery) 엣지 ──────────────────────────────────
    # 증거 있는 정지(혹은 경계) 이벤트가 끝난 직후, 스로틀이 breakaway를
    # 다시 넘기는 시점을 재출발로 표시 — 그룹별 shift로 파일 경계 보호.
    was_real_stop = is_event_evidence | is_event_boundary
    was_real_stop_prev = was_real_stop.groupby(grp).shift(1, fill_value=False)
    is_recovery_launch = (
        was_real_stop_prev & (throttle_raw > MOTOR_DEAD_ZONE_MAX) & (~was_real_stop)
    )

    # ── [신규] 비정지 회피(non-zero avoidance) 프레임 보호 ────────────────
    # [문서 지적사항 반영] zero-event만 보면 "스로틀은 유지한 채 장애물을
    # 피해간" 회피기동은 아예 감지 대상이 아니라서, 그냥 일반 스무딩을
    # 그대로 맞아 궤적이 흐려질 수 있었음. 프레임 단위로 라이다/카메라
    # 위협이 있으면서 스로틀이 살아있는(정지 이벤트가 아닌) 프레임을
    # 별도로 보호한다. steering은 게임패드 디지털 입력이라 판정에 안 씀.
    frame_baseline = grp.map(baseline_by_group).fillna(LIDAR_CLEAR_M)
    frame_lidar_threat = (
        (dist_smooth < LIDAR_DANGER_M)
        | ((dist_smooth < LIDAR_RATIO * frame_baseline) & (dist_smooth < LIDAR_CLEAR_M))
    )
    is_avoidance_frame = (
        (frame_lidar_threat | camera_obstacle_threat)
        & (throttle_raw >= ZERO_EVENT_THRESH)
    )
    # ==========================================================
    # 코너링 deadzone 보정
    # ==========================================================

    is_safe_front = (
        (~frame_lidar_threat)
        & (~camera_obstacle_threat)
    )

    is_straight = (
        df["target_steering"].abs() < 0.15
    )

    mask_corner_intent = (
        is_safe_front
        & (~is_straight)
        & (throttle_raw >= ZERO_EVENT_THRESH)
        & (throttle_raw < MOTOR_DEAD_ZONE_MAX)
    )

    corner_orig = throttle_raw.loc[mask_corner_intent]

    throttle_raw.loc[mask_corner_intent] = (
        MOTOR_DEAD_ZONE_MAX
        + (
            corner_orig - ZERO_EVENT_THRESH
        ) * (
            (0.92 - MOTOR_DEAD_ZONE_MAX)
            / (
                MOTOR_DEAD_ZONE_MAX
                - ZERO_EVENT_THRESH
            )
        )
    )

# ==========================================================

    # ── [3단계] 스무딩 보호 마스크 ──────────────────────────────────────
    is_edge_intent = was_real_stop | is_recovery_launch | is_avoidance_frame
    pad = max(1, smooth_window // 2)
    is_edge_protected = (
        is_edge_intent.astype(float)
                      .groupby(grp, group_keys=False)
                      .apply(lambda s: s.rolling(2 * pad + 1, center=True,
                                                 min_periods=1).max())
    ) > 0.5

    # [설계 변경] NORMAL_DRIVE_BOOST_FLOOR(구 0.94 고정/floor 보정) 제거.
    # 실측 검증 결과 이 보정은 "noise 제거"가 아니라 정상 주행의 자연스런
    # 속도 변화(코너 감속/직선 가속) 정보를 삭제하는 별개의 작업이었음
    # (문서 §13). MSE 회귀가 배워야 할 신호이므로 원본을 그대로 둔다.
    if smooth_window > 1:
        smoothed_throttle = (
            throttle_raw.groupby(grp, group_keys=False)
                        .apply(lambda s: s.rolling(smooth_window, center=True,
                                                   min_periods=1).mean())
        )
        df["target_throttle"] = np.where(is_edge_protected,
                                       throttle_raw, smoothed_throttle)
    else:
        df["target_throttle"] = throttle_raw

    df["_edge_protect"] = is_edge_protected.astype(int)

    n_noise    = int(is_transient_noise.sum())
    n_evidence = int(is_event_evidence.sum())
    n_boundary = int(is_event_boundary.sum())
    n_launch   = int(is_recovery_launch.sum())
    n_cam_hit  = int(camera_obstacle_threat.sum())
    n_avoid    = int(is_avoidance_frame.sum())

    print(f"[AUG] Zero-event 분류: 노이즈(보간) {n_noise} / "
          f"증거있음(보존) {n_evidence} / 파일경계(보존) {n_boundary} 프레임")
    print(f"[AUG] 재출발(recovery) 프레임: {n_launch}")
    print(f"[AUG] 비정지 회피(non-zero avoidance) 보호 프레임: {n_avoid}")
    print(f"[AUG] 카메라 근접 위협 판정 프레임 수: {n_cam_hit} "
          f"(실측상 obj_valid 거의 미발생 — 사실상 비활성 상태)")
    print(f"[AUG] NORMAL_DRIVE_BOOST_FLOOR 비활성화 — 정상 주행 throttle 변동폭 원본 보존")

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
        # --------------------------------------------------
        # source 정보
        # --------------------------------------------------

        temp_df["_src_idx"] = i

        name = file_path.stem.lower()

        if "avoid" in name:
            temp_df["scenario"] = 1

        elif "stop" in name:
            temp_df["scenario"] = 2

        elif "recovery" in name:
            temp_df["scenario"] = 3

        else:
            temp_df["scenario"] = 0

    # --------------------------------------------------


        # 파일 간 시계열/보간 오염을 막기 위한 고유 인덱스 부여
        temp_df["_src_idx"] = i
        #파일 출신을 파악하기 위해 추가
        temp_df["source_name"] = file_path.stem
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

    # [설계 변경] 기존 "0.75~0.85 구간 무조건 보간" 블록 제거.
    # 문서 §14 지적대로, 이 구간(ESC breakaway 전이 구간)은 노이즈가
    # 아니라 정지→재출발 궤적의 일부일 수 있어 무조건 지우면 안 됨.
    # apply_hierarchical_intent_filter의 zero-event 판정(ZERO_EVENT_THRESH
    # 미만만 후보로 삼고 증거 유무로 분류)이 이미 이 역할을 대신한다.
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