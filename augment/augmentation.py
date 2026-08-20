"""
augmentation.py — 데이터 증강. 시나리오별로 다르게 적용한다.
=============================================================

핵심: 모든 프레임에 동일한 8배 증강을 적용하면
"잘못된 행동도 8배로 복제"된다. 시나리오별로 다음을 구분한다.

  NOISE     → 학습 데이터에서 제외 (증강 안 함)
  AVOIDANCE → mirror 제외 (어느 쪽으로 피했는지가 행동의 핵심)
  그 외      → 전체 증강 적용

모든 증강은 numpy 행 벡터 하나에 대해 동작하며 순수 함수다
(입력을 변경하지 않고 복사본을 반환) — 단위 테스트가 쉽도록.
"""

import numpy as np
import pandas as pd

from config import (
    N_MAX_OBJECTS, OBJ_FEATURES, GRID_ROWS, GRID_COLS,
    OBJ_VALID_COLS, OBJ_LAT_COLS, OBJ_DIST_COLS, OBJ_CONF_COLS,
    ALL_OBJ_COLS, LANE_GRID_COLS, LIDAR_COLS_IDX,
    TARGET_STEER_COL, EGO_STEER_COL,
    SCENARIO_NOISE, SCENARIOS_EXCLUDED_FROM_TRAINING, SCENARIOS_NO_MIRROR,
    _COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 개별 증강 함수 (모두 순수 함수)
# ─────────────────────────────────────────────────────────────────────────────

def _clone(row: np.ndarray) -> np.ndarray:
    return row.copy()


def aug_identity(row: np.ndarray, rng) -> np.ndarray:
    return _clone(row)


def aug_mirror(row: np.ndarray, rng) -> np.ndarray:
    """
    좌우 반전. 물리적으로 타당한 변환이다.

    LiDAR : s0↔s4, s1↔s3, s2 유지 (정면은 반전해도 정면)
    object: lat_offset 부호 반전
    lane  : 열 순서 뒤집기
    steer : 부호 반전
    throttle은 건드리지 않는다 — 좌우 반전은 전후 방향(속도)에
    영향을 주지 않기 때문.
    """
    r = _clone(row)

    vals = [r[c] for c in LIDAR_COLS_IDX]
    for i in range(5):
        r[LIDAR_COLS_IDX[i]] = vals[4 - i]

    for c in OBJ_LAT_COLS:
        r[c] = -r[c]

    for gr in range(GRID_ROWS):
        row_vals = [r[LANE_GRID_COLS[gr][c]] for c in range(GRID_COLS)]
        for c in range(GRID_COLS):
            r[LANE_GRID_COLS[gr][c]] = row_vals[GRID_COLS - 1 - c]

    r[TARGET_STEER_COL] = -r[TARGET_STEER_COL]
    r[EGO_STEER_COL] = -r[EGO_STEER_COL]
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
            row_vals = [r[LANE_GRID_COLS[gr][c]] for c in range(GRID_COLS)]
            for c in range(GRID_COLS):
                src = c - shift
                r[LANE_GRID_COLS[gr][c]] = (
                    row_vals[src] if 0 <= src < GRID_COLS else 0.0
                )
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
    if rng.random() > drop_prob:
        return r
    drop_slot = int(rng.choice(valid_slots))
    start = drop_slot * OBJ_FEATURES
    for j in range(OBJ_FEATURES):
        r[ALL_OBJ_COLS[start + j]] = 0.0
    return r


def aug_distance_scale(row: np.ndarray, rng,
                       low: float = 0.85, high: float = 1.15) -> np.ndarray:
    r = _clone(row)
    scale = rng.uniform(low, high)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_DIST_COLS[i]] = float(np.clip(
                r[OBJ_DIST_COLS[i]] * scale, 0.0, 1.0))
    return r


def aug_mirror_and_noise(row: np.ndarray, rng) -> np.ndarray:
    return aug_distance_noise(aug_mirror(row, rng), rng)


# (이름, 함수, 반복횟수, 좌우반전 포함 여부)
AUGMENTATIONS = [
    ("identity",         aug_identity,         1, False),
    ("mirror",           aug_mirror,           1, True),
    ("distance_noise",   aug_distance_noise,   1, False),
    ("lateral_jitter",   aug_lateral_jitter,   1, False),
    ("confidence_noise", aug_confidence_noise, 1, False),
    ("object_dropout",   aug_object_dropout,   1, False),
    ("distance_scale",   aug_distance_scale,   1, False),
    ("mirror_noise",     aug_mirror_and_noise, 1, True),
]


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

class Augmentor:
    """시나리오를 존중하며 증강한다."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.last_stats = {}

    def augment(self, df: pd.DataFrame, scenario_labels: pd.Series) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)

        # ── 1. 학습에서 제외할 시나리오 필터링 ───────────────────────
        # 파일명이 noise이거나, 센서 판정으로 노이즈 zero-event인 프레임.
        file_scenario = df["_scenario_type"]
        excluded = file_scenario.isin(SCENARIOS_EXCLUDED_FROM_TRAINING)
        n_excluded_files = int(excluded.sum())

        keep = ~excluded
        df_keep = df.loc[keep].reset_index(drop=True)
        labels_keep = scenario_labels.loc[keep].reset_index(drop=True)
        no_mirror = df_keep["_scenario_type"].isin(SCENARIOS_NO_MIRROR).to_numpy()

        if df_keep.empty:
            self.last_stats = {
                "excluded_noise_rows": n_excluded_files,
                "input_rows": 0, "output_rows": 0, "mirror_skipped": 0,
            }
            return pd.DataFrame(columns=_COLS)

        data_np = df_keep[_COLS].to_numpy(dtype=float)

        # ── 2. 행별 증강 ─────────────────────────────────────────────
        aug_rows = []
        n_mirror_skipped = 0

        for row_i, orig_row in enumerate(data_np):
            skip_mirror = bool(no_mirror[row_i])
            for name, fn, weight, is_mirror in AUGMENTATIONS:
                if is_mirror and skip_mirror:
                    n_mirror_skipped += 1
                    continue
                for _ in range(weight):
                    aug_rows.append(fn(orig_row, rng))

        aug_df = pd.DataFrame(np.stack(aug_rows, axis=0), columns=_COLS)

        # ── 3. 후처리 클리핑 ─────────────────────────────────────────
        aug_df["frame_id"] = np.arange(1, len(aug_df) + 1)

        for i in range(N_MAX_OBJECTS):
            aug_df[f"obj{i}_valid"] = aug_df[f"obj{i}_valid"].clip(0, 1).round()
            for feat in ("conf", "dist_norm", "lane_overlap",
                         "width_norm", "height_norm"):
                col = f"obj{i}_{feat}"
                aug_df[col] = aug_df[col].clip(0, 1)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                col = f"lane_r{r}c{c}"
                aug_df[col] = aug_df[col].clip(0, 1)

        aug_df["target_steering"] = aug_df["target_steering"].clip(-1, 1)
        aug_df["target_throttle"] = aug_df["target_throttle"].clip(-1, 1)
        aug_df["scenario"] = aug_df["scenario"].round().astype(int)

        self.last_stats = {
            "excluded_noise_rows": n_excluded_files,
            "input_rows": len(df_keep),
            "output_rows": len(aug_df),
            "mirror_skipped": n_mirror_skipped,
        }

        return aug_df