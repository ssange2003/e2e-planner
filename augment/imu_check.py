#!/usr/bin/env python3
"""
imu_check.py — 수집된 CSV 로 IMU 설정이 맞는지 검증한다.
=========================================================

camera.py --imu-check 는 "지금 이 순간" 의 IMU 를 보는 실시간 도구이고,
이 스크립트는 "실제로 주행하며 쌓인 데이터" 로 임계값이 타당한지 판정한다.

사용법:
    python augment/imu_check.py data/normal_course1.csv
    python augment/imu_check.py data/*.csv

출력 3가지:
  [1] IMU 탑재 여부 — 컬럼이 있는가, 값이 전부 0 은 아닌가
  [2] IMU_MOTION_THRESH 타당성 — 정지/주행이 실제로 갈라지는가
  [3] MOTOR_DEAD_ZONE_MAX 추정 — 스로틀별 이동 비율의 무릎(knee)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    IMU_COLS, IMU_MOTION_THRESH, IMU_YAW_THRESH,
    ZERO_EVENT_THRESH, MOTOR_DEAD_ZONE_MAX,
)


def check_file(path: Path) -> bool:
    df = pd.read_csv(path).dropna()
    print("=" * 70)
    print(f"  {path.name}   ({len(df)} rows)")
    print("=" * 70)

    # ── [1] IMU 탑재 여부 ────────────────────────────────────────
    missing = [c for c in IMU_COLS if c not in df.columns]
    if missing:
        print(f"  [1] IMU 컬럼 없음 {missing}")
        print("      -> 이 파일은 IMU 도입 전에 수집된 것입니다.")
        print("         증강 파이프라인은 자동으로 라이다 기반으로 폴백합니다.")
        return False

    motion = df["imu_motion"].astype(float)
    yaw = df["imu_yaw_rate"].astype(float)
    zero_ratio = (motion == 0.0).mean()
    if zero_ratio > 0.99:
        print(f"  [1] IMU 컬럼은 있으나 {zero_ratio*100:.1f}% 가 0 입니다.")
        print("      -> IMU 가 실제로 동작하지 않았습니다(미탑재 기기이거나 스트림 실패).")
        print("         camera.py 실행 로그에서 IMU 관련 경고를 확인하세요.")
        return False
    print(f"  [1] IMU 정상  (motion 0인 프레임 {zero_ratio*100:.1f}%)")

    # ── [2] IMU_MOTION_THRESH 타당성 ─────────────────────────────
    # 스로틀이 확실히 걸린 프레임 = 주행 중, 스로틀 0 = 정지 후보.
    # 이 둘의 motion 분포가 겹치면 임계값으로 가를 수 없다는 뜻이다.
    driving = motion[df["target_throttle"] > MOTOR_DEAD_ZONE_MAX]
    zero_th = motion[df["target_throttle"] < ZERO_EVENT_THRESH]
    print()
    print("  [2] IMU_MOTION_THRESH 타당성")
    if len(driving) < 10 or len(zero_th) < 10:
        print(f"      표본 부족 (주행 {len(driving)} / 스로틀0 {len(zero_th)})")
    else:
        d_med, z_med = driving.median(), zero_th.median()
        print(f"      주행 중  motion 중앙={d_med:.4f}  p05={np.percentile(driving,5):.4f}")
      # 스로틀 0 은 글리치(움직임)와 진짜 정지가 섞여 있으므로 양쪽 꼬리를 본다
        print(f"      스로틀0  motion 중앙={z_med:.4f}  p95={np.percentile(zero_th,95):.4f}")
        sep = d_med / z_med if z_med > 1e-9 else float("inf")
        print(f"      분리비(주행중앙/정지중앙) = {sep:.2f}배")
        if sep >= 3.0:
            print("      -> 3배 이상. IMU 기반 판정이 신뢰할 만합니다.")
        elif sep >= 1.5:
            print("      -> 1.5~3배. 경계가 애매합니다. 임계값을 신중히 잡으세요.")
        else:
            print("      -> 1.5배 미만. ★ 진동만으로는 이동/정지를 가를 수 없습니다.")
            print("         엔코더 추가를 검토하세요.")
        lo, hi = np.percentile(zero_th, 75), np.percentile(driving, 25)
        if lo < hi:
            print(f"      권장 IMU_MOTION_THRESH 범위: {lo:.4f} ~ {hi:.4f}"
                  f"   (현재 {IMU_MOTION_THRESH})")
            if not (lo <= IMU_MOTION_THRESH <= hi):
                print("      ★ 현재 값이 권장 범위 밖입니다. config.py 를 고치세요.")

    # ── [3] MOTOR_DEAD_ZONE_MAX 추정 ─────────────────────────────
    # 스로틀 구간별로 "실제 움직인 비율" 을 본다. 모터가 돌기 시작하는
    # 지점에서 이 비율이 급격히 올라가며, 그 무릎이 곧 breakaway 다.
    print()
    print("  [3] MOTOR_DEAD_ZONE_MAX 추정 (스로틀 구간별 실제 이동 비율)")
    moving = motion > IMU_MOTION_THRESH
    bins = np.arange(0.0, 1.05, 0.05)
    prev = None
    knee = None
    for i in range(len(bins) - 1):
        m = (df["target_throttle"] >= bins[i]) & (df["target_throttle"] < bins[i + 1])
        n = int(m.sum())
        if n < 5:
            continue
        r = float(moving[m].mean())
        bar = "#" * int(r * 40)
        print(f"      {bins[i]:.2f}~{bins[i+1]:.2f}  n={n:4d}  {r*100:5.1f}% {bar}")
        if prev is not None and prev < 0.5 <= r and knee is None:
            knee = bins[i]
        prev = r
    if knee is not None:
        print(f"      -> 이동 비율이 50% 를 넘는 지점: {knee:.2f}")
        print(f"         현재 MOTOR_DEAD_ZONE_MAX = {MOTOR_DEAD_ZONE_MAX}")
        if abs(knee - MOTOR_DEAD_ZONE_MAX) > 0.07:
            print(f"      ★ 차이가 큽니다. {knee:.2f} 근처로 조정을 검토하세요.")
    else:
        print("      -> 무릎을 찾지 못했습니다(표본 부족 또는 전 구간 이동/정지).")

    # ── 참고: 회전 판정 ──────────────────────────────────────────
    turning = yaw.abs() > IMU_YAW_THRESH
    print()
    print(f"  [참고] 회전 판정: |yaw| > {IMU_YAW_THRESH} 인 프레임 "
          f"{turning.mean()*100:.1f}%   |yaw| p95={np.percentile(yaw.abs(),95):.3f}")
    print()
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python augment/imu_check.py <csv> [csv ...]")
        sys.exit(1)
    any_ok = False
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"파일 없음: {p}")
            continue
        any_ok |= check_file(p)
    if not any_ok:
        print("IMU 데이터가 있는 파일이 없습니다. IMU 수집 후 다시 실행하세요.")
