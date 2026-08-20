# scenario.py
import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from sensor_evidence import compute_sensor_evidence

def load_and_parse_files(input_files: list) -> pd.DataFrame:
    """파일 로드 및 파일명 기반 Prior(메타데이터) 추출"""
    df_list = []
    for i, file_path_str in enumerate(input_files):
        path = Path(file_path_str)
        if not path.exists(): continue
        
        temp_df = pd.read_csv(path, on_bad_lines='warn').dropna().reset_index(drop=True)
        temp_df["_src_idx"] = i
        
        # 파일명 Prior: 파일 이름에 'stop'이 들어가면 강한 정지 의도로 간주
        temp_df["_prior_stop"] = "stop" in path.name.lower()
        df_list.append(temp_df)
        
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def apply_hierarchical_intent_filter(df: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    grp = df["_src_idx"]
    throttle_raw = df["target_throttle"].copy()
    
    # 1. 센서 증거 확보
    ev = compute_sensor_evidence(df, smooth_window, grp)
    
    # 조향 스무딩
    if smooth_window > 1:
        df["target_steering"] = df["target_steering"].groupby(grp, group_keys=False).apply(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
        ).clip(-1.0, 1.0)

    # 2. Zero-event 그룹화 및 Baseline 계산
    is_near_zero = throttle_raw < ZERO_EVENT_THRESH
    run_change = is_near_zero.groupby(grp, group_keys=False).apply(lambda s: s.ne(s.shift()).cumsum())
    event_key = (grp.astype(str) + "_" + run_change.astype(str)).where(is_near_zero)

    driving_mask = throttle_raw > THROTTLE_IDLE_MAX
    front_baseline = ev['front_dist'][driving_mask].groupby(grp[driving_mask]).median()
    lane_baseline = ev['lane_sum'][driving_mask].groupby(grp[driving_mask]).median()

    is_transient_noise = pd.Series(False, index=df.index)
    is_event_evidence  = pd.Series(False, index=df.index)
    is_event_boundary  = pd.Series(False, index=df.index)
    is_confirmed_stop  = pd.Series(False, index=df.index)

    # 3. 이벤트별 증거 판정
    for gid, sub in df.groupby(grp):
        idx = sub.index
        base_f = front_baseline.get(gid, LIDAR_CLEAR_M)
        base_l = lane_baseline.get(gid, np.nan)
        prior_stop = sub["_prior_stop"].iloc[0] # 파일명 Prior
        
        for eid in event_key.loc[idx].dropna().unique():
            ev_idx = idx[event_key.loc[idx] == eid]
            local_s, local_e = idx.get_loc(ev_idx.min()), idx.get_loc(ev_idx.max())
            is_boundary = (local_s == 0) or (local_e == len(idx) - 1)
            
            ps, pe = idx[max(0, local_s - CONTEXT_PAD)], idx[min(len(idx) - 1, local_e + CONTEXT_PAD)]
            dmin = ev['front_dist'].loc[ps:pe].min()
            cam_hit = bool(ev['cam_threat'].loc[ps:pe].any())
            
            trend_s = idx[max(0, local_s - APPROACH_TREND_WINDOW)]
            approach_drop = ev['front_dist'].loc[trend_s] - dmin
            is_approaching = approach_drop > APPROACH_DROP_M
            
            lane_hit = False
            if not np.isnan(base_l) and base_l > 0:
                lane_hit = ev['lane_sum'].loc[ev_idx].mean() < LANE_VIS_DROP_RATIO * base_l

            # [핵심] 파일명 prior가 'stop'이면 증거 기준을 대폭 완화하여 원본 보존
            evidence = prior_stop or (dmin < LIDAR_DANGER_M) or (dmin < LIDAR_RATIO * base_f and dmin < LIDAR_CLEAR_M) or (is_approaching and dmin < LIDAR_CLEAR_M) or cam_hit or lane_hit

            if is_boundary: is_event_boundary.loc[ev_idx] = True
            elif evidence:
                is_event_evidence.loc[ev_idx] = True
                if len(ev_idx) >= MIN_STOP_FRAMES: is_confirmed_stop.loc[ev_idx] = True
            else: is_transient_noise.loc[ev_idx] = True

    # 노이즈 보간
    throttle_raw = throttle_raw.mask(is_transient_noise).groupby(grp).transform(lambda s: s.interpolate(method="linear", limit_direction="both"))

    # 4. 상태 머신 (RECOVERY & AVOIDANCE)
    frame_front_base = grp.map(front_baseline).fillna(LIDAR_CLEAR_M)
    front_threat = (ev['front_dist'] < LIDAR_DANGER_M) | ((ev['front_dist'] < LIDAR_RATIO * frame_front_base) & (ev['front_dist'] < LIDAR_CLEAR_M))
    obstacle_cleared = (~front_threat) & (~ev['cam_threat'])
    
    stop_state = is_confirmed_stop | is_event_boundary
    is_recovery_launch = pd.Series(False, index=df.index)
    
    for gid, sub in df.groupby(grp):
        idx = sub.index
        ss, thr, clr = stop_state.loc[idx].values, throttle_raw.loc[idx].values, obstacle_cleared.loc[idx].values
        awaiting = False
        for i in range(len(idx)):
            if ss[i]: awaiting = False; continue
            if i > 0 and ss[i-1] and not ss[i]: awaiting = True
            if awaiting and thr[i] > MOTOR_DEAD_ZONE_MAX and clr[i]:
                is_recovery_launch.loc[idx[i]] = True
                awaiting = False

    side_threat = ev['side_dist'] < LIDAR_DANGER_M
    is_avoidance = (front_threat | side_threat | ev['cam_threat']) & (throttle_raw >= ZERO_EVENT_THRESH)

    # 5. 스무딩 및 마스킹
    is_edge_intent = (is_event_evidence | is_event_boundary | is_recovery_launch | is_avoidance)
    pad = max(1, smooth_window // 2)
    is_edge_protected = is_edge_intent.astype(float).groupby(grp, group_keys=False).apply(
        lambda s: s.rolling(2 * pad + 1, center=True, min_periods=1).max()
    ) > 0.5

    if smooth_window > 1:
        smoothed_throttle = throttle_raw.groupby(grp, group_keys=False).apply(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
        )
        df["target_throttle"] = np.where(is_edge_protected, throttle_raw, smoothed_throttle)
    else:
        df["target_throttle"] = throttle_raw

    df.drop(columns=["_prior_stop"], inplace=True)
    return df