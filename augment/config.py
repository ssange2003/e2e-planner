"""
config.py — 모든 threshold와 스키마 헬퍼를 한 곳에서 관리
=========================================================

각 값 옆의 태그 의미:
  [실측]   — 실제 주행 로그(course1~3)로 검증된 값. 함부로 바꾸지 말 것.
  [추론]   — 물리/논리적 근거는 있으나 실측 캘리브레이션 전. 튜닝 대상.
  [미검증] — 근거가 약함. 실측 데이터 확보 시 최우선 재검토 대상.
"""

from planner_model import (
    N_MAX_OBJECTS, OBJ_FEATURES, LANE_FEATURES, EGO_FEATURES,
    GRID_ROWS, GRID_COLS, csv_columns,
)

# ─────────────────────────────────────────────────────────────────────────────
# Throttle / Motor
# ─────────────────────────────────────────────────────────────────────────────

# [실측] ESC breakaway. 이 미만은 실제로 모터가 돌지 않음.
MOTOR_DEAD_ZONE_MAX = 0.85

# [실측] 실측 로그에서 0.75 미만 값이 거의 관찰되지 않음 —
# 정지/주행이 이분법적으로 갈리는 경계.
THROTTLE_IDLE_MAX = 0.75

# [실측] "거의 0"인 프레임만 zero-event 후보로 삼는다.
# (과거 0.75를 노이즈 판정선으로 썼다가 정상 저속 주행을 삭제하는
#  문제가 실측으로 확인되어 폐기됨)
ZERO_EVENT_THRESH = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# LiDAR
# ─────────────────────────────────────────────────────────────────────────────

# [추론] 절대 위협 거리. 라이다 마운트가 센서 원점 기준인지 범퍼
# 오프셋 보정된 값인지 미확인 — 마운트 실측 후 재검증 필요.
LIDAR_DANGER_M = 0.30

# [추론] 위협 해제 거리. DANGER와 다른 값을 써서 hysteresis를 만든다
# (경계선에서 STOP↔RECOVERY가 프레임마다 진동하는 것 방지).
LIDAR_CLEAR_M = 0.50

# [실측] 코스마다 라이다 baseline이 완전히 다름(개활 4m대 vs 협로 0.5m대)이라
# 절대값만으로는 위협을 못 가림. 단, 상대비율만 쓰면 baseline이 넓은 코스에서
# 오탐이 나므로(course1 frame 350 검증) 반드시 LIDAR_CLEAR_M과 AND로 묶어 쓸 것.
LIDAR_RATIO = 0.60

# [실측] STOP 판정은 반드시 정면만 사용. 측면(s1/s3)은 S구간에서 상시
# 근접이라 STOP 증거로 쓰면 S구간 전체가 STOP으로 오분류됨.
FRONT_LIDAR_COL = "lidar_s2"
SIDE_LIDAR_COLS = ["lidar_s1", "lidar_s3"]

# ─────────────────────────────────────────────────────────────────────────────
# Camera (YOLO objects)
# ─────────────────────────────────────────────────────────────────────────────

# [실측] planner_model.py의 정규화 공식: dist_norm = 실제거리(m) / 5.0
# 따라서 LIDAR_DANGER_M과 동일한 물리 거리 기준으로 환산한 값.
CAMERA_OBJ_DANGER_NORM = LIDAR_DANGER_M / 5.0   # = 0.06

# ─────────────────────────────────────────────────────────────────────────────
# Event / context
# ─────────────────────────────────────────────────────────────────────────────

# [실측] zero-event 전후 증거 탐색 범위. course1/3 교차검증 통과.
CONTEXT_PAD = 5

# [추론] 정식 STOP으로 승격할 최소 지속 프레임(30fps 기준 0.17초).
# 짧은 정면 튐 하나로 recovery 상태머신이 열리는 것을 방지.
MIN_STOP_FRAMES = 5

# [추론] 접근 추세 — 절대/상대 거리 조건에 안 걸리는 "느리지만 꾸준히
# 다가오는" 장애물을 조기 포착. 반드시 LIDAR_CLEAR_M 게이트와 함께 쓸 것
# (게이트 없이 낙폭만 보면 course1 정상 주행이 오탐됨).
APPROACH_TREND_WINDOW = 15
APPROACH_DROP_M = 0.15

# [미검증] 차선 가시성 보조 증거. 라이다/카메라보다 약한 신호라 엄격하게.
# 반드시 "이벤트 구간 자체의 평균"으로 판정할 것 — 넓은 윈도우+최소값
# 방식은 인접 프레임의 일시적 카메라 글리치를 오탐함(course1 frame 25-26).
LANE_VIS_DROP_RATIO = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# S-CURVE (종이컵 구간)
# ─────────────────────────────────────────────────────────────────────────────

# [추론] 차선 폭 약 40cm이므로 좌우 라이다가 0.2~0.4m로 지속되는 것은 정상.
# "양쪽이 동시에 가까움 + 정면은 열림"이 S구간의 정의.
S_CURVE_SIDE_CLOSE_M = 0.45     # 한쪽 측면이 이 미만이면 "가깝다"
S_CURVE_FRONT_OPEN_M = 0.35     # 정면이 이 이상이면 "진행 공간 있다"
S_CURVE_REQUIRED_RATIO = 0.60   # 윈도우 내 이 비율 이상 지속되어야 S구간 인정
S_CURVE_WINDOW = 5              # S구간 판정 스무딩 윈도우

# ─────────────────────────────────────────────────────────────────────────────
# Scenario types (파일명 prior)
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_NORMAL    = "normal"
SCENARIO_NOISE     = "noise"
SCENARIO_AVOIDANCE = "avoidance"
SCENARIO_S_CURVE   = "s_curve"
SCENARIO_STOP      = "stop"
SCENARIO_RECOVERY  = "recovery"

KNOWN_SCENARIOS = (
    SCENARIO_NORMAL,
    SCENARIO_NOISE,
    SCENARIO_AVOIDANCE,
    SCENARIO_S_CURVE,
    SCENARIO_STOP,
    SCENARIO_RECOVERY,
)

# 파일명을 파싱하지 못했을 때의 기본값. NORMAL로 두는 이유:
# prior 없이 순수 센서 판정만 받게 되어(=구버전 동작) 가장 보수적.
SCENARIO_FALLBACK = SCENARIO_NORMAL

# 이 시나리오의 파일은 학습 데이터에서 제외한다(증강도 하지 않음).
# 사람이 "이건 실수로 찍힌 데이터"라고 직접 라벨링한 것이므로,
# 8배 증강하면 잘못된 행동이 8배로 복제된다.
SCENARIOS_EXCLUDED_FROM_TRAINING = (SCENARIO_NOISE,)

# 좌우 반전(mirror) 증강을 적용하지 않을 시나리오.
# [근거] mirror는 물리적으로 타당한 변환이지만, AVOIDANCE는 "어느 쪽으로
# 피했는가"가 행동의 핵심이라 반전 시 원본과 반대 방향 회피가 생성된다.
# 좌우 비대칭 장애물 배치를 학습해야 하는 경우 노이즈가 될 수 있어 제외.
SCENARIOS_NO_MIRROR = (SCENARIO_AVOIDANCE,)

# ─────────────────────────────────────────────────────────────────────────────
# Column index helpers (numpy 행 벡터 조작용 — augmentation에서 사용)
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

LIDAR_COLS_IDX = [_col(f"lidar_s{i}") for i in range(5)]

TARGET_STEER_COL = _col("target_steering")
TARGET_THTL_COL  = _col("target_throttle")
EGO_STEER_COL    = _col("ego_steering")

ALL_OBJ_COLS = []
for _i in range(N_MAX_OBJECTS):
    for _f in ("valid", "class_norm", "conf", "dist_norm",
               "lat_offset", "width_norm", "height_norm", "lane_overlap"):
        ALL_OBJ_COLS.append(_obj_col(_i, _f))

# 내부 작업용 컬럼 (최종 CSV 저장 전 제거 대상)
INTERNAL_COLS = [
    "_src_idx", "_scenario_type", "_course_id", "_source_file",
    "_edge_protect", "_is_s_curve", "_is_stop", "_is_recovery",
    "_is_avoidance", "_is_noise_event",
]