"""
label_processor.py — 라벨 처리(보간/스무딩). 상황 판단은 하지 않는다.
======================================================================

ScenarioClassifier가 "이 프레임은 RECOVERY다"라고 정하면,
LabelProcessor는 "RECOVERY니까 스무딩하지 않는다"를 실행할 뿐이다.
판정과 처리를 분리해야 각각을 독립적으로 테스트/수정할 수 있다.

모든 시계열 연산은 _src_idx 그룹 안에서만 수행된다.
"""

import numpy as np
import pandas as pd


def _grouped_rolling_mean(series: pd.Series, grp: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(grp, group_keys=False)
              .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    )


class LabelProcessor:
    """노이즈 보간 + 조건부 스무딩을 수행한다."""

    def __init__(self, smooth_window: int = 5):
        self.smooth_window = smooth_window

    def process(self, df: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grp = df["_src_idx"]

        throttle = df["target_throttle"].copy()
        steering = df["target_steering"].copy()

        # ── 1. 노이즈로 판정된 zero-event만 선형 보간 ────────────────
        # 증거 있는 정지/파일경계 이벤트는 손대지 않는다.
        # 파일 경계 이벤트를 보간에서 제외하는 이유: 앞뒤 컨텍스트가
        # 부족해 실제 정지를 잘못된 값으로 채울 위험이 있다.
        noise_mask = decisions["is_noise_event"]
        n_interpolated = int(noise_mask.sum())

        throttle = throttle.mask(noise_mask)
        throttle = (
            throttle.groupby(grp)
                    .transform(lambda s: s.interpolate(method="linear",
                                                       limit_direction="both"))
        )

        # ── 2. 조건부 스무딩 ─────────────────────────────────────────
        # 보호 프레임(STOP/RECOVERY/AVOIDANCE/S_CURVE 및 그 이웃)은
        # 원본을 그대로 두고, 나머지 일반 주행 구간만 평활화한다.
        protected = decisions["is_protected"]

        if self.smooth_window > 1:
            steering_sm = _grouped_rolling_mean(
                steering, grp, self.smooth_window
            ).clip(-1.0, 1.0)
            df["target_steering"] = np.where(protected, steering, steering_sm)

            throttle_sm = _grouped_rolling_mean(throttle, grp, self.smooth_window)
            df["target_throttle"] = np.where(protected, throttle, throttle_sm)
        else:
            df["target_steering"] = steering
            df["target_throttle"] = throttle

        # ── 3. 판정 결과를 디버깅용 컬럼으로 부착 ────────────────────
        # 이 컬럼들은 최종 CSV 저장 직전에 제거되지만,
        # 증강 단계에서 시나리오별 차등 처리를 하려면 반드시 필요하다.
        df["_edge_protect"] = protected.astype(int)
        df["_is_s_curve"] = decisions["is_s_curve"].astype(int)
        df["_is_stop"] = decisions["is_stop"].astype(int)
        df["_is_recovery"] = decisions["is_recovery"].astype(int)
        df["_is_avoidance"] = decisions["is_avoidance"].astype(int)
        df["_is_noise_event"] = decisions["is_noise_event"].astype(int)

        self.last_stats = {
            "interpolated": n_interpolated,
            "protected": int(protected.sum()),
        }

        return df