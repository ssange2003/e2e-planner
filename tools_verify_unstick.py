# -*- coding: utf-8 -*-
"""unstick 안전성 꼼꼼 검증."""
import sys, os, warnings, glob
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"c:/Users/USER/Desktop/e2e-planner-1")
os.chdir(r"c:/Users/USER/Desktop/e2e-planner-1")
from planner_model import PlannerModel, GRID_ROWS, GRID_COLS

SPD = r"C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Desktop-e2e-planner-1/dcfbdd3e-9507-409e-b7db-6eb99fd2d653/scratchpad/"
OF = ("valid", "class_norm", "conf", "dist_norm", "lat_offset", "width_norm", "height_norm", "lane_overlap")
LC = ["lane_r%dc%d" % (r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
LOW, GATE, HOLD, RESET = 0.05, 0.80, 5, 0.50


def sens(df):
    T = lambda a, d=torch.float32: torch.tensor(a, dtype=d)
    return (T(np.stack([df["obj%d_%s" % (i, f)].astype(float).to_numpy() for i in range(5) for f in OF], 1)),
            T(np.stack([df[c].astype(float).to_numpy() for c in LC], 1)),
            T(np.stack([df["lidar_s%d" % i].astype(float).to_numpy() for i in range(5)], 1)),
            T(df["scenario"].astype(int).to_numpy(), torch.long))


def load(p):
    m = PlannerModel(); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def roll(m, df, unstick=True, freeze=False):
    """freeze=True: 갇혔을 때 센서를 정지 시점에 고정 (실차에서 차가 안 움직이는 상황)"""
    o, l, li, sc = sens(df)
    s2 = df["lidar_s2"].astype(float).to_numpy()
    n = len(df)
    out = np.zeros(n); ps, pt = 0.0, 1.0; cnt = 0
    fires = []          # (프레임, s2)
    frozen_at = None
    with torch.no_grad():
        for i in range(n):
            j = i if (frozen_at is None) else frozen_at
            y = m(o[j:j+1], l[j:j+1], li[j:j+1],
                  torch.tensor([[ps, pt]], dtype=torch.float32), sc[j:j+1]).numpy()[0]
            out[i] = y[1]; ps, pt = float(y[0]), float(y[1])
            if freeze:
                if pt < LOW and frozen_at is None: frozen_at = i     # 멈추면 센서 정지
                elif pt >= LOW: frozen_at = None                      # 다시 움직이면 해제
            if unstick:
                cnt = cnt + 1 if (pt < LOW and s2[j] > GATE) else 0
                if cnt >= HOLD:
                    pt = RESET; cnt = 0; fires.append((i, s2[j]))
    return out, fires


files = sorted(glob.glob("data/*_course*.csv")) + sorted(glob.glob("raw/*_course*.csv"))
m = load(SPD + "m_all.pth")

print("=" * 92)
print("검증 4 — 리셋 53회가 '멈춰야 할 때' 발동하는가 (위험) '가야 할 때'인가 (정상)")
print("=" * 92)
tot_run = tot_stop = 0
det = []
for f in files:
    df = pd.read_csv(f)
    tT = df["target_throttle"].astype(float).to_numpy()
    _, fires = roll(m, df)
    for i, s2v in fires:
        if np.abs(tT[i]) < 0.05: tot_stop += 1; det.append((os.path.basename(f)[:-4], i, s2v, tT[i]))
        else: tot_run += 1
print("  정답이 '주행' 일 때 발동  %3d 회   ← 정상 (갇힘 해소)" % tot_run)
print("  정답이 '정지' 일 때 발동  %3d 회   ← 위험 (멈춰야 하는데 품)" % tot_stop)
if det:
    print()
    print("  위험 발동 상세")
    print("  %-22s%8s%10s%10s" % ("코스", "프레임", "s2(m)", "정답"))
    for a, b, c, d in det[:15]:
        print("  %-22s%8d%10.2f%10.3f" % (a, b, c, d))

print()
print("=" * 92)
print("검증 5 — 늘어난 폭주 154프레임은 어디서 나오나")
print("=" * 92)
print("%-22s%10s%10s%10s" % ("코스", "unstick없음", "unstick", "증가"))
d0 = d1 = 0
for f in files:
    df = pd.read_csv(f)
    tT = df["target_throttle"].astype(float).to_numpy()
    st = np.abs(tT) < 0.05
    a, _ = roll(m, df, unstick=False)
    b, _ = roll(m, df, unstick=True)
    x, y = int((a[st] > 0.6).sum()), int((b[st] > 0.6).sum())
    d0 += x; d1 += y
    if y - x != 0:
        print("%-22s%10d%10d%+10d" % (os.path.basename(f)[:-4], x, y, y - x))
print("%-22s%10d%10d%+10d" % ("합계", d0, d1, d1 - d0))

print()
print("=" * 92)
print("검증 6 — 센서 고정(실차에 더 가까운 조건)에서도 결론이 유지되나")
print("  갇히면 차가 안 움직이므로 라이다도 그 시점에 멈춘다고 가정")
print("=" * 92)
print("%-22s%12s%12s%12s%12s" % ("조건", "갇힘프레임", "폭주", "리셋", "MAE"))


def runs(mask, n=20):
    r = []; c = None
    for i, v in enumerate(mask):
        if v and c is None: c = i
        elif not v and c is not None:
            if i - c >= n: r.append((c, i));
            c = None
    if c is not None and len(mask) - c >= n: r.append((c, len(mask)))
    return r


for frz in [False, True]:
    for un in [False, True]:
        sf = rw = nr = 0; err = []
        for f in files:
            df = pd.read_csv(f)
            tT = df["target_throttle"].astype(float).to_numpy()
            b, fires = roll(m, df, unstick=un, freeze=frz)
            nr += len(fires)
            stk = (b < 0.1) & (tT > 0.5)
            for a, e in runs(stk): sf += (e - a)
            rw += int((b[np.abs(tT) < 0.05] > 0.6).sum()); err.append(np.abs(b - tT))
        print("%-22s%12d%12d%12d%12.4f" % (
            ("센서고정 " if frz else "센서진행 ") + ("unstick" if un else "없음"),
            sf, rw, nr, np.concatenate(err).mean()))
