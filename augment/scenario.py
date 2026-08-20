"""
scenario.py — 파일명 prior + 센서 증거 → 최종 시나리오 분류
============================================================

설계 원칙: prior는 비대칭으로 적용한다.
------------------------------------------------
  파일명이 stop/recovery인데 센서 증거가 없다  → 보존 쪽으로 승격 (prior 채택)
  파일명이 normal인데 센서 증거가 강하다        → 증거를 존중해 보존 (prior 무시)

즉 prior는 "증거가 없어도 보존"하는 방향으로만 작용하고,
"증거가 있어도 무시"하는 방향으로는 절대 작용하지 않는다.
사람이 파일명을 잘못 붙였을 가능성이 항상 있으므로, 안전한 쪽
(=라벨을 지우지 않는 쪽)으로만 prior를 신뢰한다.

프레임 라벨(우선순위 순, 뒤가 앞을 덮어씀):
    NORMAL < S_CURVE < AVOIDANCE < RECOVERY < STOP < NOISE
STOP/RECOVERY가 가장 강한 이유는 이 둘이 behavior cloning에서
가장 잃기 쉬우면서 가장 중요한 transition이기 때문이다.
"""

import numpy as np
import pandas as pd

from config import (
    ZERO_EVENT_THRESH, THROTTLE_IDLE_MAX, MOTOR_DEAD_ZONE_MAX,
    LIDAR_DANGER_M, LIDAR_CLEAR_M, LIDAR_RATIO,
    CONTEXT_PAD, MIN_STOP_FRAMES,
    APPROACH_TREND_WINDOW, APPROACH_DROP_M,
    LANE_VIS_DROP_RATIO, LANE_BASELINE_MIN,
    S_CURVE_SIDE_CLOSE_M, S_CURVE_FRONT_OPEN_M,
    S_CURVE_REQUIRED_RATIO, S_CURVE_WINDOW,
    SCENARIO_NORMAL, SCENARIO_NOISE, SCENARIO_AVOIDANCE,
    SCENARIO_S_CURVE, SCENARIO_STOP, SCENARIO_RECOVERY,
)


# 라벨 우선순위 — 숫자가 클수록 강하다.
# 단순 if/else 덩어리 대신 이 표로 관리해서, 새 시나리오 추가 시
# 이 딕셔너리에 한 줄만 넣으면 되도록 한다.
LABEL_PRIORITY = {
    SCENARIO_NORMAL:    0,
    SCENARIO_S_CURVE:   1,
    SCENARIO_AVOIDANCE: 2,
    SCENARIO_RECOVERY:  3,
    SCENARIO_STOP:      4,
    SCENARIO_NOISE:     5,
}


def detect_s_curve(ev: pd.DataFrame, grp: pd.Series) -> pd.Series:
    """
    S구간 검출: 양쪽이 동시에 좁고 + 정면은 열려 있음.

    [핵심] side_left와 side_right를 각각 봐야 한다. min() 하나로는
    한쪽 벽에 붙어 도는 일반 코너와 구분이 안 된다.

    순간적인 값 튐을 배제하기 위해 "윈도우 내 지속 비율"로 판정한다.
    """
    both_close_raw = (
        (ev["side_left"] < S_CURVE_SIDE_CLOSE_M)
        & (ev["side_right"] < S_CURVE_SIDE_CLOSE_M)
    )
    both_close_ratio = (
        both_close_raw.astype(float)
        .groupby(grp, group_keys=False)
        .transform(lambda s: s.rolling(S_CURVE_WINDOW, center=True, min_periods=1).mean())
    )
    both_close = both_close_ratio >= S_CURVE_REQUIRED_RATIO
    front_open = ev["front_dist"] >= S_CURVE_FRONT_OPEN_M
    return both_close & front_open


def compute_front_threat(ev: pd.DataFrame, grp: pd.Series,
                         throttle: pd.Series) -> pd.Series:
    """
    프레임 단위 정면 위협. 절대거리 또는 (상대비율 AND 절대상한).

    [실측] 상대비율만 쓰면 baseline이 넓은 코스(course1, 4.5m)에서
    1.87m를 "가까움"으로 오판정한다 — 반드시 LIDAR_CLEAR_M과 AND로 묶는다.
    """
    driving = throttle > THROTTLE_IDLE_MAX
    baseline = ev["front_dist"][driving].groupby(grp[driving]).median()
    frame_baseline = grp.map(baseline).fillna(LIDAR_CLEAR_M)
    return (
        (ev["front_dist"] < LIDAR_DANGER_M)
        | ((ev["front_dist"] < LIDAR_RATIO * frame_baseline)
           & (ev["front_dist"] < LIDAR_CLEAR_M))
    )


class ScenarioClassifier:
    """센서 증거 + 파일명 prior → 프레임별 시나리오 라벨과 보호 마스크."""

    def __init__(self, smooth_window: int = 5):
        self.smooth_window = smooth_window

    def classify(self, df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
        """
        반환: 프레임별 판정 결과 DataFrame
          scenario_label   : 최종 시나리오 문자열
          is_stop          : 정식 STOP (MIN_STOP_FRAMES 충족)
          is_recovery      : 재출발 edge
          is_avoidance     : 비정지 회피
          is_s_curve       : S구간 통과
          is_noise_event   : 노이즈로 판정된 zero-event (보간 대상)
          is_protected     : 원본 라벨 보존(스무딩 금지) 대상
        """
        grp = df["_src_idx"]
        throttle = df["target_throttle"]
        scenario_type = df["_scenario_type"]

        out = pd.DataFrame(index=df.index)
        out["is_noise_event"] = False
        out["is_evidence_event"] = False
        out["is_boundary_event"] = False
        out["is_stop"] = False

        # ── S구간 (프레임 단위, zero-event와 무관하게 성립) ──────────
        is_s_curve = detect_s_curve(ev, grp)
        # 파일명 prior: s_curve 파일이면 측면 근접 조건을 만족하는 구간을
        # 더 관대하게 인정한다(정면만 열려 있으면 S구간으로 본다).
        s_curve_prior = scenario_type == SCENARIO_S_CURVE
        is_s_curve = is_s_curve | (
            s_curve_prior
            & (ev["front_dist"] >= S_CURVE_FRONT_OPEN_M)
            & (ev["side_dist"] < S_CURVE_SIDE_CLOSE_M)
        )
        out["is_s_curve"] = is_s_curve

        # ── zero-event 그룹화 ────────────────────────────────────────
        is_near_zero = throttle < ZERO_EVENT_THRESH
        run_change = (
            is_near_zero.groupby(grp, group_keys=False)
                        .transform(lambda s: s.ne(s.shift()).cumsum())
        )
        event_key = (grp.astype(str) + "_" + run_change.astype(str)).where(is_near_zero)

        # ── 그룹별 baseline ──────────────────────────────────────────
        driving = throttle > THROTTLE_IDLE_MAX
        front_baseline = ev["front_dist"][driving].groupby(grp[driving]).median()
        lane_baseline = ev["lane_sum"][driving].groupby(grp[driving]).median()

        # ── 이벤트 단위 판정 ─────────────────────────────────────────
        for gid, sub in df.groupby(grp):
            idx = sub.index
            if len(idx) == 0:
                continue
            gsize = len(idx)

            base_front = front_baseline.get(gid, LIDAR_CLEAR_M)
            if pd.isna(base_front):
                base_front = LIDAR_CLEAR_M
            base_lane = lane_baseline.get(gid, np.nan)

            stype = sub["_scenario_type"].iloc[0]
            # prior: stop/recovery 파일의 zero-event는 사람이 의도한
            # 정지라고 보증된 것이므로 증거가 없어도 보존한다.
            prior_preserve = stype in (SCENARIO_STOP, SCENARIO_RECOVERY)
            # prior: noise 파일의 zero-event는 사람이 "실수"라고 라벨링한
            # 것이므로 노이즈 판정을 강화한다(단, 물리적 증거가 명확하면
            # 아래 evidence 계산에서 살아남는다 — 비대칭 원칙).
            prior_noise = stype == SCENARIO_NOISE

            for eid in event_key.loc[idx].dropna().unique():
                ev_idx = idx[event_key.loc[idx] == eid]
                if len(ev_idx) == 0:
                    continue

                local_s = idx.get_loc(ev_idx.min())
                local_e = idx.get_loc(ev_idx.max())
                is_boundary = (local_s == 0) or (local_e == gsize - 1)
                ev_len = len(ev_idx)

                ctx_idx = idx[max(0, local_s - CONTEXT_PAD):
                              min(gsize - 1, local_e + CONTEXT_PAD) + 1]

                # 정면 거리만 STOP 증거로 사용 (측면은 S구간에서 상시 근접)
                dmin = ev["front_dist"].loc[ctx_idx].min()
                cam_hit = bool(ev["cam_threat"].loc[ctx_idx].any())

                # 윈도우 시작점 "한 프레임" 만 보면 그 프레임이 튀었을 때
                # 판정 전체가 흔들린다. 구간 중앙값으로 기준선을 잡는다.
                trend_slice = idx[max(0, local_s - APPROACH_TREND_WINDOW):max(1, local_s)]
                pre_level = (ev["front_dist"].loc[trend_slice].median()
                             if len(trend_slice) else dmin)
                approach_drop = pre_level - dmin
                is_approaching = approach_drop > APPROACH_DROP_M

                # 차선 증거는 반드시 이벤트 구간 자체의 평균으로 판정
                # lane 증거는 baseline 대비 상대비율로 판정하는데, S구간에서는
                # BiSeNet 이 차선을 거의 못 잡아 baseline 자체가 0.003 수준이다
                # (normal 은 0.86~0.97). 그 상태의 상대비율은 무의미하므로
                # 최소 신뢰 하한을 넘는 경우에만 증거로 인정한다.
                lane_hit = False
                if not pd.isna(base_lane) and base_lane >= LANE_BASELINE_MIN:
                    lane_mean = ev["lane_sum"].loc[ev_idx].mean()
                    lane_hit = lane_mean < LANE_VIS_DROP_RATIO * base_lane

                front_danger = (
                    (dmin < LIDAR_DANGER_M)
                    or (dmin < LIDAR_RATIO * base_front and dmin < LIDAR_CLEAR_M)
                )

                # S구간이면 정면 위협 증거를 STOP으로 승격하지 않는다
                event_is_s_curve = bool(is_s_curve.loc[ctx_idx].mean() >= 0.5)

                sensor_evidence = (
                    (front_danger and not event_is_s_curve)
                    or (is_approaching and not event_is_s_curve and dmin < LIDAR_CLEAR_M)
                    or cam_hit
                    or lane_hit
                )

                # 비대칭 prior 적용:
                #  - stop/recovery 파일 → 증거 없어도 보존
                #  - noise 파일 → 증거 있으면 여전히 보존(무시하지 않음)
                evidence = sensor_evidence or (prior_preserve and not prior_noise)

                if is_boundary:
                    out.loc[ev_idx, "is_boundary_event"] = True
                elif evidence:
                    out.loc[ev_idx, "is_evidence_event"] = True
                    if ev_len >= MIN_STOP_FRAMES:
                        out.loc[ev_idx, "is_stop"] = True
                else:
                    out.loc[ev_idx, "is_noise_event"] = True

        # ── 프레임 단위 위협 (회피/재출발 판정용) ────────────────────
        front_threat = compute_front_threat(ev, grp, throttle)
        # 측면(s0/s4)은 절대거리만 사용한다. S구간에서 측면이 상시 근접인
        # 것은 "좁은 구간을 통과 중"이라는 뜻이지 위협이 아니므로,
        # 상대비율(LIDAR_RATIO)은 적용하지 않는다.
        side_threat = ev["side_dist"] < LIDAR_DANGER_M
        # 전방 사선(s1/s3)은 정면과 측면 사이의 사각을 메운다. 비스듬히
        # 접근하는 장애물은 s2 에도 s0/s4 에도 안 걸릴 수 있다.
        oblique_threat = ev["oblique_dist"] < LIDAR_DANGER_M
        obstacle_cleared = (~front_threat) & (~ev["cam_threat"])

        # ── 재출발 상태머신 ──────────────────────────────────────────
        # [실측 근거] shift(1) 단일 프레임 방식은 정지 종료 후 스로틀이
        # 여러 프레임에 걸쳐 서서히 breakaway를 넘는 실측 패턴
        # (0.45→0.70→0.82→0.95)에서 재출발을 통째로 놓친다.
        # "정지 종료 후 breakaway를 처음 넘는 순간까지" 대기한다.
        stop_state = out["is_stop"] | out["is_boundary_event"]
        is_recovery = pd.Series(False, index=df.index)

        for gid, sub in df.groupby(grp):
            idx = sub.index
            ss = stop_state.loc[idx].to_numpy()
            thr = throttle.loc[idx].to_numpy()
            clr = obstacle_cleared.loc[idx].to_numpy()
            stype = sub["_scenario_type"].iloc[0]
            # recovery 파일이면 장애물 해소 조건을 요구하지 않는다
            # (사람이 "여기서 재출발했다"고 보증했으므로).
            relax_clear = (stype == SCENARIO_RECOVERY)

            # [구간화] breakaway 를 넘은 "한 프레임" 이 아니라, 정지가 끝난
            # 지점부터 breakaway 를 처음 넘는 지점까지 전체를 RECOVERY 로 본다.
            # 0 → 0.4 → 0.7 → 0.85 → 0.95 전 구간이 재출발 행동이기 때문이다.
            awaiting, span_start = False, None
            for i in range(len(idx)):
                if ss[i]:
                    awaiting, span_start = False, None
                    continue
                if i > 0 and ss[i - 1] and not ss[i]:
                    awaiting, span_start = True, i
                if awaiting and (clr[i] or relax_clear):
                    if thr[i] > MOTOR_DEAD_ZONE_MAX:
                        is_recovery.loc[idx[span_start:i + 1]] = True
                        awaiting, span_start = False, None
                    elif i - span_start > 30:      # 3초 내 재출발 없으면 포기
                        awaiting, span_start = False, None

        out["is_recovery"] = is_recovery


        # ── 비정지 회피 ──────────────────────────────────────────────
        # 스로틀이 살아있는 상태에서 위협을 지나가는 프레임.
        # S구간은 별도 라벨이므로 회피에서 제외한다(둘 다 보호되지만
        # 라벨이 섞이면 behavior cloning이 구분을 못 배운다).
        is_avoidance = (
            (front_threat | side_threat | oblique_threat | ev["cam_threat"])
            & (throttle >= ZERO_EVENT_THRESH)
            & (~out["is_s_curve"])
        )
        # 파일명 prior: avoidance 파일에서 위협이 관측되면 회피로 인정
        avoid_prior = scenario_type == SCENARIO_AVOIDANCE
        is_avoidance = is_avoidance | (
            avoid_prior
            & (throttle >= ZERO_EVENT_THRESH)
            & (front_threat | side_threat | oblique_threat | ev["cam_threat"])
        )
        out["is_avoidance"] = is_avoidance

        # ── 최종 라벨 결정 (우선순위 표 기반) ────────────────────────
        label = pd.Series(SCENARIO_NORMAL, index=df.index)
        for flag_col, name in (
            ("is_s_curve",   SCENARIO_S_CURVE),
            ("is_avoidance", SCENARIO_AVOIDANCE),
            ("is_recovery",  SCENARIO_RECOVERY),
            ("is_stop",      SCENARIO_STOP),
            ("is_noise_event", SCENARIO_NOISE),
        ):
            mask = out[flag_col]
            # 더 높은 우선순위만 덮어쓴다
            higher = mask & (
                label.map(LABEL_PRIORITY) < LABEL_PRIORITY[name]
            )
            label.loc[higher] = name
        out["scenario_label"] = label

        # ── 보호 마스크 (스무딩 금지 대상) ───────────────────────────
        protected = (
            out["is_evidence_event"]
            | out["is_boundary_event"]
            | out["is_stop"]
            | out["is_recovery"]
            | out["is_avoidance"]
            | out["is_s_curve"]
        )
        pad = max(1, self.smooth_window // 2)
        out["is_protected"] = (
            protected.astype(float)
                     .groupby(grp, group_keys=False)
                     .transform(lambda s: s.rolling(2 * pad + 1, center=True,
                                                    min_periods=1).max())
        ) > 0.5

        return out