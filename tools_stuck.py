# -*- coding: utf-8 -*-
"""갇힘의 원인 판별 — A(라이다 오독) vs B(ego 자기고정).

판별법: 갇힌 프레임에서 ego_throttle 만 0.7 로 강제로 되돌린다.
  회복하면  -> B (ego 자기고정). 탈출 조건만 넣으면 해결.
  안 하면   -> A (라이다를 '막힘'으로 읽음). 입력 표현을 고쳐야 함.
"""
import sys, os, warnings, glob
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"c:/Users/USER/Desktop/e2e-planner-1")
os.chdir(r"c:/Users/USER/Desktop/e2e-planner-1")
from planner_model import PlannerModel, GRID_ROWS, GRID_COLS

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
    m = PlannerModel(); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def rollout(m, df, warm=1.0, rescue=None):
    """rescue: (임계값, 리셋값) — 스로틀이 임계값 미만이면 ego 를 리셋값으로 되돌린다."""
    o, l, li, sc = sensors(df)
    out = np.zeros((len(df), 2)); ps, pt = 0.0, warm; nres = 0
    with torch.no_grad():
        for i in range(len(df)):
            y = m(o[i:i+1], l[i:i+1], li[i:i+1],
                  torch.tensor([[ps, pt]], dtype=torch.float32), sc[i:i+1]).numpy()[0]
            out[i] = y; ps, pt = float(y[0]), float(y[1])
            if rescue is not None and pt < rescue[0]:
                pt = rescue[1]; nres += 1
    return out, nres


def runs(mask, minlen=20):
    """연속 True 구간 [(start,end)]"""
    r = []; c = None
    for i, v in enumerate(mask):
        if v and c is None: c = i
        elif not v and c is not None:
            if i - c >= minlen: r.append((c, i))
            c = None
    if c is not None and len(mask) - c >= minlen: r.append((c, len(mask)))
    return r


M = {"m_all": load(SPD + "m_all.pth"), "m_raw": load(SPD + "m_raw.pth")}
files = sorted(glob.glob("data/*_course*.csv")) + sorted(glob.glob("raw/*_course*.csv"))

for name, m in M.items():
    print("=" * 96)
    print("### %s — 갇힘 구간 목록 (스로틀<0.1 이 2초 이상, 정답은 주행)" % name)
    print("=" * 96)
    print("%-24s%9s%9s%9s%12s%12s%14s" % ("코스", "시작", "길이(s)", "정답", "front중앙", "left/right", "ego리셋시"))
    tot = 0; recovered = 0; frames = 0
    for f in files:
        df = pd.read_csv(f)
        tT = df["target_throttle"].astype(float).to_numpy()
        col = lambda c, alt: (df[c].astype(float).to_numpy() if c in df.columns
                              else df[alt].astype(float).to_numpy())
        fc = col("lidar_front_clear", "lidar_s2")
        lg = col("lidar_left_gap", "lidar_s0")
        rg = col("lidar_right_gap", "lidar_s4")
        b, _ = rollout(m, df)
        stuck = (b[:, 1] < 0.1) & (tT > 0.5)
        for a, e in runs(stuck, 20):
            tot += 1; frames += (e - a)
            # 그 구간의 센서를 그대로 두고 ego 만 0.7 로 강제
            o, l, li, sc = sensors(df)
            with torch.no_grad():
                p = m(o[a:e], l[a:e], li[a:e],
                      torch.tensor([[0.0, 0.7]] * (e - a), dtype=torch.float32), sc[a:e]).numpy()[:, 1]
            rec = p.mean()
            if rec > 0.5: recovered += 1
            print("%-24s%9d%9.1f%9.3f%12.3f%12s%14.3f" % (
                os.path.basename(f)[:-4], a, (e - a) / 10.0, tT[a:e].mean(),
                np.median(fc[a:e]), "%.2f/%.2f" % (np.median(lg[a:e]), np.median(rg[a:e])), rec))
    print("-" * 96)
    print("  갇힘 %d회 · %d프레임(%.1f초)  ·  ego 리셋으로 회복 %d회 (%.0f%%)" %
          (tot, frames, frames / 10.0, recovered, 100 * recovered / max(tot, 1)))
    print()

print("=" * 96)
print("탈출 조건(rescue)을 넣으면 갇힘이 사라지는가 — 전 코스")
print("=" * 96)
print("%-12s%-22s%12s%12s%12s%12s" % ("모델", "조건", "갇힘프레임", "폭주", "리셋횟수", "스로틀MAE"))
for name, m in M.items():
    for rc in [None, (0.05, 0.5), (0.10, 0.7)]:
        sf = rw = nr = 0; err = []; N = 0
        for f in files:
            df = pd.read_csv(f)
            tT = df["target_throttle"].astype(float).to_numpy()
            b, n = rollout(m, df, rescue=rc)
            nr += n; N += len(df)
            stuck = (b[:, 1] < 0.1) & (tT > 0.5)
            for a, e in runs(stuck, 20): sf += (e - a)
            rw += int((b[np.abs(tT) < 0.05, 1] > 0.6).sum())
            err.append(np.abs(b[:, 1] - tT))
        lab = "없음" if rc is None else "스로틀<%.2f → ego=%.1f" % rc
        print("%-12s%-22s%12d%12d%12d%12.4f" % (name, lab, sf, rw, nr, np.concatenate(err).mean()))
