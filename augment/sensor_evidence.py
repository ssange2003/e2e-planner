# sensor_evidence.py
import pandas as pd
import numpy as np
from config import *

def compute_sensor_evidence(df: pd.DataFrame, smooth_window: int, grp: pd.Series) -> pd.DataFrame:
    """센서 원시 데이터를 스무딩하여 물리적 '증거(거리, 위협 유무)'만 추출"""
    ev = pd.DataFrame(index=df.index)
    
    # 정면 라이다 (STOP 판단용)
    if FRONT_LIDAR_COL in df.columns:
        ev['front_dist'] = df[FRONT_LIDAR_COL].groupby(grp, group_keys=False).apply(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).min()
        )
    else:
        ev['front_dist'] = 999.0

    # 측면 라이다 (회피 판단용)
    if all(c in df.columns for c in SIDE_LIDAR_COLS):
        ev['side_dist'] = df[SIDE_LIDAR_COLS].min(axis=1).groupby(grp, group_keys=False).apply(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).min()
        )
    else:
        ev['side_dist'] = 999.0

    # 카메라 장애물
    if all(c in df.columns for c in OBJ_VALID_COLS + OBJ_DIST_COLS):
        valid_mat = df[OBJ_VALID_COLS].to_numpy() > 0.5
        dist_mat  = df[OBJ_DIST_COLS].to_numpy()
        ev['cam_threat'] = (valid_mat & (dist_mat < CAMERA_OBJ_DANGER_NORM)).any(axis=1)
    else:
        ev['cam_threat'] = False

    # 차선 가시성
    lane_cols = [f"lane_r{r}c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    if all(c in df.columns for c in lane_cols):
        ev['lane_sum'] = df[lane_cols].sum(axis=1)
    else:
        ev['lane_sum'] = np.nan

    return ev