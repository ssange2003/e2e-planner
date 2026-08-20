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

# ─────────────────────────────────────────────────────────────────
# [적용됨] 측면 섹터 재지정  —  s1/s3  →  s0/s4
# ─────────────────────────────────────────────────────────────────
# ■ 제어주기 영향 : 0
#   근거1) planner_inference.py 의 import 목록에 augment/ 모듈이 없다.
#          → 이 파일의 코드는 주행 중 한 줄도 실행되지 않는다.
#   근거2) planner_inference.py:468 process_lidar_to_5_sectors() 가
#          이미 매 프레임 s0~s4 를 전부 계산하고, :475 에서 5개를
#          통째로 모델에 넘긴다. 센서를 더 읽지 않는다.
#   → 바뀌는 것은 "오프라인 라벨 판정이 어느 컬럼을 보느냐"뿐이다.
#
# ■ 왜 바꿔야 하는가 (기하)
#   라이다 1000점 / 360° = 0.36°/idx. collect_data_planner.py 의
#   섹터 인덱스를 각도로 환산하면:
#       s2 = 972~27  →  ±9.9°        (정면)
#       s1 = 27~83   →  9.7°~29.9°   (전방 사선)
#       s0 = 83~166  →  29.9°~59.8°  (측면)
#   좌우 d[m] 벽에 대한 각 섹터의 최소 관측거리 = d / sin(θ_max)
#       s1 → d/sin(29.9°) = 2.01·d      s0 → d/sin(59.8°) = 1.16·d
#   즉 s1 은 벽에 아무리 붙어도 s0 보다 항상 1.7배 멀게 읽힌다.
#
# ■ 실측 (s_curve_course1.csv, 371 프레임)
#       s0 중앙값 0.382 / s1 0.636 / s2 0.921 / s3 0.697 / s4 0.351
#   S_CURVE_SIDE_CLOSE_M(0.45) 기준 기하조건 성립률:
#       s1&s3 →  0.0%  (normal 2파일도 0.0% — 판별력 자체가 없음)
#       s0&s4 → 59.0%  (normal_course2 0.0% / normal_course1 2.5%)
#   현재 61.7% 가 s_curve 로 잡히는 것은 전부 파일명 prior 분기의
#   느슨한 min(s1,s3) 조건이 처리한 결과다. 파일명을 떼면 0% 가 된다.
#
# ■ sensor_evidence.py 는 SIDE_LIDAR_COLS[0], [1] 을 인덱스로 읽으므로
#   소비 측 코드 수정 없이 이 한 곳으로 전환이 끝난다.
SIDE_LIDAR_COLS    = ["lidar_s0", "lidar_s4"]   # 진짜 측면 (29.9~59.8°)
OBLIQUE_LIDAR_COLS = ["lidar_s1", "lidar_s3"]   # 전방 사선 — 회피 보조용

# ─────────────────────────────────────────────────────────────────────────────
# Camera (YOLO objects)
# ─────────────────────────────────────────────────────────────────────────────

# [실측] planner_model.py의 정규화 공식: dist_norm = 실제거리(m) / 5.0
# 따라서 LIDAR_DANGER_M과 동일한 물리 거리 기준으로 환산한 값.
CAMERA_OBJ_DANGER_NORM = LIDAR_DANGER_M / 5.0   # = 0.06

# [적용됨]  제어주기 영향 : 0  (augment/ 전용 상수)
#   이 임계값은 구조적으로 한 번도 통과된 적이 없다.
#   yolo_config.py 의 13개 클래스는 BFMC 표지판/차량/보행자이며
#   종이컵에 해당하는 클래스가 존재하지 않는다.
#   실측 obj_valid 발생: normal_c1 5/4190, 나머지 3파일 0/전체.
#   "카메라 파이프라인이 개선되면 자동으로 유효해진다"는 기존 주석은
#   사실이 아니다 — YOLO 를 재학습하기 전까지 영원히 False 다.
#   죽은 신호를 살아있는 것처럼 두면 판정 로직을 읽을 때 오해를 부른다.
CAMERA_EVIDENCE_ENABLED = False   # sensor_evidence.py 에서 게이트로 사용

# ─────────────────────────────────────────────────────────────────────────────
# Event / context
# ─────────────────────────────────────────────────────────────────────────────

# [실측] zero-event 전후 증거 탐색 범위. course1/3 교차검증 통과.
CONTEXT_PAD = 5

# [추론] 정식 STOP으로 승격할 최소 지속 프레임(30fps 기준 0.17초).
# 짧은 정면 튐 하나로 recovery 상태머신이 열리는 것을 방지.
# MIN_STOP_FRAMES = 5   ← 구값. 30fps 오해석이었음(실제 0.5초)

# [적용됨]  제어주기 영향 : 0  (augment/ 전용 상수)
#   위 주석의 "30fps 기준 0.17초" 가 사실과 다르다.
#   collect_data_planner.py: SAVE_FPS = 10  →  1프레임 = 0.1초
#   따라서 현재 값 5 는 0.17초가 아니라 0.5초다.
#   0.5초 미만 정지는 is_stop=False 가 되어 stop_state 에 들어가지
#   못하고, 그 결과 recovery 상태머신이 아예 열리지 않는다.
MIN_STOP_FRAMES = 2          # 0.2초 @ 10fps

# [추론] 접근 추세 — 절대/상대 거리 조건에 안 걸리는 "느리지만 꾸준히
# 다가오는" 장애물을 조기 포착. 반드시 LIDAR_CLEAR_M 게이트와 함께 쓸 것
# (게이트 없이 낙폭만 보면 course1 정상 주행이 오탐됨).
# APPROACH_TREND_WINDOW = 15   ← 구값(실제 1.5초)

# [적용됨]  제어주기 영향 : 0  (augment/ 전용 상수)
#   같은 이유로 "0.5초 @30fps" 가 아니라 실제로는 1.5초를 본다.
APPROACH_TREND_WINDOW = 5    # 0.5초 @ 10fps
APPROACH_DROP_M = 0.15

# [미검증] 차선 가시성 보조 증거. 라이다/카메라보다 약한 신호라 엄격하게.
# 반드시 "이벤트 구간 자체의 평균"으로 판정할 것 — 넓은 윈도우+최소값
# 방식은 인접 프레임의 일시적 카메라 글리치를 오탐함(course1 frame 25-26).
LANE_VIS_DROP_RATIO = 0.40

# [적용됨]  제어주기 영향 : 0  (augment/ 전용 상수)
#   lane 증거는 "baseline 대비 비율" 로 판정하는데, S구간에서는
#   baseline 자체가 노이즈 수준이라 판정식이 무의미해진다.
#   실측 driving 중 lane_sum 중앙값:
#       normal_course1 0.857 / normal_course2 0.923 / avoidance 0.966
#       s_curve_course1 0.003   ← 0.4 를 곱하면 임계값이 0.0012
#   BiSeNet 이 S구간에서 차선을 거의 못 잡는다(완전 0 인 프레임 31%).
#   이 상태의 상대비율 판정은 우연히 STOP 증거를 만들어낼 수 있다.
#   → baseline 이 이 값 미만이면 lane 증거를 아예 쓰지 않는다.
LANE_BASELINE_MIN = 0.05     # scenario.py classify() 에서 게이트로 사용

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

# [적용됨] 거리 스케일링(aug_distance_scale)을 적용하지 않을 시나리오.
# [근거] STOP/RECOVERY 는 "얼마나 가까워서 멈췄는가 / 언제 다시 출발했는가"
# 가 행동의 전부다. 거리를 U(0.85,1.15) 로 흔들면 그 인과관계 자체가
# 훼손된다. 반면 NORMAL/S_CURVE 에서는 유효한 다양성 확보 수단이다.
SCENARIOS_NO_DIST_SCALE = (SCENARIO_STOP, SCENARIO_RECOVERY)

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