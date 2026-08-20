"""
sensor_evidence.py — 센서 "증거"만 계산한다. 상황 판단은 하지 않는다.
=====================================================================

이 모듈의 유일한 책임은 원시 센서값을 스무딩·정규화해서
"물리적으로 무엇이 관측되었는가"를 내놓는 것이다.

STOP인지 AVOIDANCE인지 S_CURVE인지는 여기서 정하지 않는다 —
그건 scenario.py의 몫이다. 이 분리가 중요한 이유: 같은 "측면 0.25m"라는
증거가 S구간에서는 정상이고 다른 곳에서는 위협일 수 있는데, 센서 계층이
미리 결론을 내려버리면 그 맥락 판단이 불가능해진다.

모든 시계열 연산은 grp(_src_idx) 그룹 안에서만 수행된다.
"""

import numpy as np
import pandas as pd

from config import (
    FRONT_LIDAR_COL, SIDE_LIDAR_COLS, OBLIQUE_LIDAR_COLS,
    OBJ_VALID_COLS, OBJ_DIST_COLS, CAMERA_OBJ_DANGER_NORM,
    CAMERA_EVIDENCE_ENABLED,
    GRID_ROWS, GRID_COLS,
    N_MAX_OBJECTS,
)


def _grouped_rolling_min(series: pd.Series, grp: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(grp, group_keys=False)
              .transform(lambda s: s.rolling(window, center=True, min_periods=1).min())
    )


def compute_sensor_evidence(df: pd.DataFrame,
                            grp: pd.Series,
                            smooth_window: int = 5) -> pd.DataFrame:
    """
    센서 증거 테이블을 반환한다.

    컬럼:
      front_dist   : 정면 라이다 s2 (±9.9°, rolling-min). STOP 판정의 유일한 거리 근거.
      side_left    : 좌측 라이다 s0 (29.9~59.8°, rolling-min)
      side_right   : 우측 라이다 s4 (29.9~59.8°, rolling-min)
      side_dist    : min(좌, 우) — 회피/S구간 판정용
      oblique_dist : 전방 사선 s1/s3 (9.7~29.9°) 중 최솟값 — 회피 보조
      cam_threat   : 카메라 근접 장애물 유무 (bool)
      lane_sum     : 차선 grid 72셀 합 (가시성 척도)

    [섹터 선정 근거] 라이다 1000점/360° = 0.36°/idx 로 환산하면
    s1/s3 는 전방 사선(9.7~29.9°)이지 측면이 아니다. 좌우 d[m] 벽에
    대한 최소 관측거리는 d/sin(θmax) 이므로 s1 은 s0 보다 항상 1.7배
    멀게 읽히고, 그 결과 S_CURVE_SIDE_CLOSE_M(0.45) 에 도달하지 못한다.
    실측 s_curve_course1: s0/s4 = 0.382/0.351 vs s1/s3 = 0.636/0.697.

    [중요] side_left/side_right를 따로 유지하는 이유:
    S구간의 정의는 "양쪽이 동시에 가까움"이다. min() 하나로 뭉개면
    한쪽 벽에만 붙어 도는 일반 코너와 구분할 수 없게 된다.
    """
    ev = pd.DataFrame(index=df.index)

    # ── 정면 (STOP 판정 전용) ────────────────────────────────────────
    if FRONT_LIDAR_COL in df.columns:
        ev["front_dist"] = _grouped_rolling_min(
            df[FRONT_LIDAR_COL].astype(float), grp, smooth_window
        )
    else:
        ev["front_dist"] = 999.0

    # ── 측면 (좌/우 개별 유지 — S구간 판정에 필수) ───────────────────
    left_col, right_col = SIDE_LIDAR_COLS[0], SIDE_LIDAR_COLS[1]
    if left_col in df.columns and right_col in df.columns:
        ev["side_left"] = _grouped_rolling_min(
            df[left_col].astype(float), grp, smooth_window
        )
        ev["side_right"] = _grouped_rolling_min(
            df[right_col].astype(float), grp, smooth_window
        )
        ev["side_dist"] = ev[["side_left", "side_right"]].min(axis=1)
    else:
        ev["side_left"] = 999.0
        ev["side_right"] = 999.0
        ev["side_dist"] = 999.0

    # ── 전방 사선 (s1/s3) — 회피 판정 보조. STOP 증거로는 쓰지 않는다 ──
    if all(c in df.columns for c in OBLIQUE_LIDAR_COLS):
        ev["oblique_dist"] = _grouped_rolling_min(
            df[OBLIQUE_LIDAR_COLS].min(axis=1).astype(float), grp, smooth_window
        )
    else:
        ev["oblique_dist"] = 999.0

    # ── 카메라 장애물 (YOLO) ─────────────────────────────────────────
    # [주의] 실측 4개 코스에서 obj_valid가 거의 발생하지 않아(0~5/4190)
    # 이 신호는 현재 사실상 비활성이다. 카메라 파이프라인이 개선되면
    # 코드 변경 없이 자동으로 다시 유효해진다.
    obj_valid_names = [f"obj{i}_valid" for i in range(N_MAX_OBJECTS)]
    obj_dist_names = [f"obj{i}_dist_norm" for i in range(N_MAX_OBJECTS)]
    if not CAMERA_EVIDENCE_ENABLED:
        # yolo_best.pt 의 13클래스(BFMC 표지판/차량/보행자)에 종이컵이 없어
        # 이 신호는 구조적으로 항상 False 다. 실측 obj_valid: 5/11200 슬롯.
        # 죽은 신호를 살아있는 것처럼 두면 판정 로직을 오독하게 되므로
        # 명시적으로 끈다. YOLO 재학습 시 config 플래그만 되돌리면 된다.
        ev["cam_threat"] = False
    elif all(c in df.columns for c in obj_valid_names + obj_dist_names):
        valid_mat = df[obj_valid_names].to_numpy() > 0.5
        dist_mat = df[obj_dist_names].to_numpy()
        ev["cam_threat"] = (valid_mat & (dist_mat < CAMERA_OBJ_DANGER_NORM)).any(axis=1)
    else:
        ev["cam_threat"] = False

    # ── 차선 가시성 ──────────────────────────────────────────────────
    lane_names = [f"lane_r{r}c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    if all(c in df.columns for c in lane_names):
        ev["lane_sum"] = df[lane_names].sum(axis=1)
    else:
        ev["lane_sum"] = np.nan

    return ev