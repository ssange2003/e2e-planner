# config.py
from planner_model import (
    N_MAX_OBJECTS, OBJ_FEATURES, LANE_FEATURES, EGO_FEATURES,
    GRID_ROWS, GRID_COLS, csv_columns
)

# ── [Thresholds] ─────────────────────────────────────────
MOTOR_DEAD_ZONE_MAX = 0.85
THROTTLE_IDLE_MAX   = 0.75
ZERO_EVENT_THRESH   = 0.10

LIDAR_DANGER_M      = 0.30
LIDAR_CLEAR_M       = 0.50
LIDAR_RATIO         = 0.6
CONTEXT_PAD         = 5

CAMERA_OBJ_DANGER_NORM = 0.06

APPROACH_TREND_WINDOW = 15
APPROACH_DROP_M       = 0.15
LANE_VIS_DROP_RATIO   = 0.4

FRONT_LIDAR_COL = "lidar_s2"
SIDE_LIDAR_COLS = ["lidar_s1", "lidar_s3"]
MIN_STOP_FRAMES = 5

# ── [Column Helpers] ─────────────────────────────────────
_COLS = csv_columns()
def _col(name: str) -> int: return _COLS.index(name)
def _obj_col(slot: int, feat: str) -> int: return _col(f"obj{slot}_{feat}")

OBJ_VALID_COLS   = [_obj_col(i, "valid") for i in range(N_MAX_OBJECTS)]
OBJ_LAT_COLS     = [_obj_col(i, "lat_offset") for i in range(N_MAX_OBJECTS)]
OBJ_DIST_COLS    = [_obj_col(i, "dist_norm") for i in range(N_MAX_OBJECTS)]
OBJ_CONF_COLS    = [_obj_col(i, "conf") for i in range(N_MAX_OBJECTS)]
LANE_GRID_COLS   = [[_col(f"lane_r{r}c{c}") for c in range(GRID_COLS)] for r in range(GRID_ROWS)]

TARGET_STEER_COL = _col("target_steering")
TARGET_THTL_COL  = _col("target_throttle")
EGO_STEER_COL    = _col("ego_steering")

ALL_OBJ_COLS = []
for i in range(N_MAX_OBJECTS):
    for f in ("valid", "class_norm", "conf", "dist_norm", "lat_offset", "width_norm", "height_norm", "lane_overlap"):
        ALL_OBJ_COLS.append(_obj_col(i, f))