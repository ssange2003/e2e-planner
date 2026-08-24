# -*- coding: utf-8 -*-
"""오프라인 폐루프 롤아웃.

센서(objects/lane/lidar/scenario)는 기록된 실제값을 그대로 쓰고,
ego 두 채널만 모델 자신의 직전 출력으로 되먹인다.
= 실주행에서 일어나는 compounding error 를 오프라인에서 재현.

한계: 차가 실제로 다른 경로를 갔다면 센서값도 달라졌을 것이므로
      완전한 폐루프는 아니다. 다만 ego 되먹임의 영향만은 정확히 분리된다.
"""
import sys, os, warnings, glob
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"c:/Users/USER/Desktop/e2e-planner-1")
os.chdir(r"c:/Users/USER/Desktop/e2e-planner-1")
from planner_model import PlannerModel, GRID_ROWS, GRID_COLS, MAX_THROTTLE

SPD = r"C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Desktop-e2e-planner-1/dcfbdd3e-9507-409e-b7db-6eb99fd2d653/scratchpad/"
OF = ("valid", "class_norm", "conf", "dist_norm", "lat_offset", "width_norm", "height_norm", "lane_overlap")
LC = ["lane_r%dc%d" % (r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def sensors(df):
    T = lambda a, d=torch.float32: torch.tensor(a, dtype=d)
    return (T(np.stack([df["obj%d_%s" % (i, f)].astype(float).to_numpy() for i in range(5) for f in OF], 1)),
            T(np.stack([df[c].astype(float).to_numpy() for c in LC], 1)),
            T(np.stack([df["lidar_s%d" % i].astype(float).to_numpy() for i in range(5)], 1)),
            T(df["scenario"].astype(int).to_numpy(), torch.long))


def load(p):
    m = PlannerModel()
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def rollout(m, df, warm=1.0):
    """ego 를 자기 출력으로 되먹이며 한 프레임씩 전진. planner_inference.py 재현."""
    o, l, li, sc = sensors(df)
    n = len(df)
    out = np.zeros((n, 2))
    ps, pt = 0.0, warm          # planner_inference.py:400-401 초기값
    with torch.no_grad():
        for i in range(n):
            ego = torch.tensor([[ps, pt]], dtype=torch.float32)
            y = m(o[i:i + 1], l[i:i + 1], li[i:i + 1], ego, sc[i:i + 1]).numpy()[0]
            out[i] = y
            ps, pt = float(y[0]), float(y[1])     # :570-571 되먹임
    return out


def teacher(m, df):
    """기록된 ego 를 그대로 쓰는 개루프(오프라인 평가와 동일)."""
    o, l, li, sc = sensors(df)
    ego = torch.tensor(np.stack([df["ego_steering"].astype(float).to_numpy(),
                                 df["ego_throttle"].astype(float).to_numpy()], 1), dtype=torch.float32)
    with torch.no_grad():
        return m(o, l, li, ego, sc).numpy()


MODELS = {}
for nm, p in [("순정", "planner_model_lidar.pth"),
              ("m_base", "experiments/checkpoints/m_base.pth"),
              ("m_oldnew", SPD + "m_oldnew.pth"),
              ("m_raw", SPD + "m_raw.pth"),
              ("m_all", SPD + "m_all.pth"),
              ("m_ho_d45", SPD + "m_ho_d45.pth")]:
    if os.path.exists(p):
        MODELS[nm] = load(p)

print("=" * 96)
print("오프라인 폐루프 롤아웃 — ego 만 자기 출력으로 되먹임 (센서는 기록값)")
print("=" * 96)

for f in ["raw/stop_course1.csv", "raw/stop_course2.csv", "raw/avoidance_course2.csv"]:
    df = pd.read_csv(f)
    tT = df["target_throttle"].astype(float).to_numpy()
    tS = df["target_steering"].astype(float).to_numpy()
    fc = df["lidar_front_clear"].astype(float).to_numpy()
    stop = np.abs(tT) < 0.05
    run = tT > 0.5
    blk = fc < 1.0
    print()
    print("### %s  (%d행, 정지 %d, 막힘 %d)" % (os.path.basename(f)[:-4], len(df), stop.sum(), blk.sum()))
    print("%-12s %-30s %-30s" % ("", "개루프 (기록된 ego)", "폐루프 (자기 출력 되먹임)"))
    print("%-12s %10s%10s%10s %10s%10s%10s" % ("모델", "정지시", "막힘시", "스로MAE", "정지시", "막힘시", "스로MAE"))
    for k, m in MODELS.items():
        a = teacher(m, df)
        b = rollout(m, df)
        print("%-12s %10.4f%10.4f%10.4f %10.4f%10.4f%10.4f" % (
            k, a[stop, 1].mean(), a[blk, 1].mean(), np.abs(a[:, 1] - tT).mean(),
            b[stop, 1].mean(), b[blk, 1].mean(), np.abs(b[:, 1] - tT).mean()))

# 정지 진입 순간만
print()
print("=" * 96)
print("정지 '진입' 순간만 (직전 프레임은 달리고 있었고 지금 멈춰야 하는 프레임)")
print("=" * 96)
for f in ["raw/stop_course1.csv", "raw/stop_course2.csv"]:
    df = pd.read_csv(f)
    tT = df["target_throttle"].astype(float).to_numpy()
    z = np.abs(tT) < 0.05
    onset = np.zeros(len(df), bool)
    onset[1:] = z[1:] & ~z[:-1]
    print()
    print("### %s  진입 %d회" % (os.path.basename(f)[:-4], onset.sum()))
    print("%-12s%14s%14s%14s" % ("모델", "개루프 예측", "폐루프 예측", "정답"))
    for k, m in MODELS.items():
        a = teacher(m, df)
        b = rollout(m, df)
        print("%-12s%14.4f%14.4f%14.4f" % (k, a[onset, 1].mean(), b[onset, 1].mean(), tT[onset].mean()))
