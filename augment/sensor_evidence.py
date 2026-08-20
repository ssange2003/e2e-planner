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
    FRONT_LIDAR_COL, SIDE_LIDAR_COLS,
    OBJ_VALID_COLS, OBJ_DIST_COLS, CAMERA_OBJ_DANGER_NORM,
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
      front_dist  : 정면 라이다 (rolling-min). STOP 판정의 유일한 거리 근거.
      side_left   : 좌측 라이다 (rolling-min)
      side_right  : 우측 라이다 (rolling-min)
      side_dist   : min(좌, 우) — 회피 판정용
      cam_threat  : 카메라 근접 장애물 유무 (bool)
      lane_sum    : 차선 grid 72셀 합 (가시성 척도)

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

    # ── 카메라 장애물 (YOLO) ─────────────────────────────────────────
    # [주의] 실측 4개 코스에서 obj_valid가 거의 발생하지 않아(0~5/4190)
    # 이 신호는 현재 사실상 비활성이다. 카메라 파이프라인이 개선되면
    # 코드 변경 없이 자동으로 다시 유효해진다.
    obj_valid_names = [f"obj{i}_valid" for i in range(N_MAX_OBJECTS)]
    obj_dist_names = [f"obj{i}_dist_norm" for i in range(N_MAX_OBJECTS)]
    if all(c in df.columns for c in obj_valid_names + obj_dist_names):
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