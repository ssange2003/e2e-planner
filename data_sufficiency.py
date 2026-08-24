#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_sufficiency.py — 데이터가 부족한가, 정보가 부족한가

saliency.py 가 "학습된 모델이 무엇을 보는가" 를 재는 도구라면,
이 파일은 "데이터를 더 모으면 좋아지는가" 를 잰다. 목적이 다르므로 따로 둔다.

두 가지를 잰다.

  1. 학습곡선   학습에 쓰는 코스 수를 늘려가며 hold-out 성능을 본다.
                계속 오르면 데이터 부족, 평평해지면 정보 한계다.
                모델을 40에폭 돌리는 대신 gradient boosting 을 쓴다.
                신경망보다 표본 효율이 높아서, 여기서 안 오르면
                신경망에서도 안 오른다(상한 추정).

  2. 코스 기여도  코스를 하나씩 빼고 학습해 성능이 얼마나 떨어지는지 본다.
                많이 떨어지면 그 코스가 유일한 정보원 = 비슷한 걸 더 찍어야 한다.
                안 떨어지면 이미 중복 = 더 찍어도 소용없다.

사용법
  python data_sufficiency.py                       전체 진단
  python data_sufficiency.py --task stop           특정 과제만
  python data_sufficiency.py --dirs data raw       입력 폴더 지정
  python data_sufficiency.py --plot curve.png      그래프 저장
  python data_sufficiency.py --repeats 20          반복 늘려 잡음 줄이기

판정 기준 (--slope-th 로 조정)
  마지막 두 점의 기울기가 코스당 +0.010 이상   -> 데이터 부족. 더 찍으면 오른다
  +0.003 ~ +0.010                            -> 수확체감. 많이 찍어야 조금 오른다
  +0.003 미만                                 -> 정보 한계. 더 찍어도 안 오른다
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "augment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner_model import GRID_ROWS, GRID_COLS          # noqa: E402
from dataset_loader import DatasetLoader, discover_files  # noqa: E402

from sklearn.ensemble import (                            # noqa: E402
    HistGradientBoostingClassifier as GC,
    HistGradientBoostingRegressor as GR,
)
from sklearn.metrics import balanced_accuracy_score, r2_score  # noqa: E402

LANE_COLS = [f"lane_r{r}c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
SECT_COLS = [f"lidar_s{i}" for i in range(5)]
CORR_COLS = ["lidar_front_clear", "lidar_left_gap", "lidar_right_gap"]

SLOPE_SHORT = 0.010   # 코스당 이만큼 이상 오르면 "데이터 부족"
SLOPE_LIMIT = 0.003   # 이만큼도 안 오르면 "정보 한계"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 적재
# ─────────────────────────────────────────────────────────────────────────────
def load_all(dirs: list[str]) -> pd.DataFrame:
    """여러 폴더의 코스 CSV 를 파일명 충돌 없이 한 번에 읽는다.

    data/ 와 raw/ 에 같은 이름(avoidance_course1 등)이 있어도 둘 다 살린다.
    _src_idx 가 파일별로 부여되므로 모든 시계열 연산이 파일 경계를 넘지 않는다.
    """
    tmp = _HERE / "_ds_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    import re as _re
    _RE = _re.compile(r"^(?P<sc>.+)_course(?P<n>\d+)$", _re.IGNORECASE)
    used: set[str] = set()
    try:
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, "*_course*.csv"))):
                b = os.path.basename(f)
                stem, ext = os.path.splitext(b)
                # 이름이 겹치면 코스 번호를 밀어 새 코스로 만든다.
                # dataset_loader 의 규약이 {scenario}_course{N} 이라
                # 접미사를 붙이면 파일이 조용히 버려진다(실측: 13개 -> 10개).
                m = _RE.match(stem)
                out = b
                if stem in used and m:
                    sc, k = m.group("sc"), int(m.group("n"))
                    while True:
                        k += 10
                        cand = "%s_course%d" % (sc, k)
                        if cand not in used:
                            out = cand + ext
                            stem = cand
                            break
                elif stem in used:
                    continue        # 규약 밖 이름이 겹치면 건너뛴다
                used.add(stem)
                shutil.copy(f, tmp / out)
        df = DatasetLoader().load(discover_files(str(tmp)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return df


def build_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    grp = df["_src_idx"].to_numpy()

    def diff(a: np.ndarray, k: int) -> np.ndarray:
        o = np.zeros_like(a)
        for g in np.unique(grp):
            m = grp == g
            o[m] = pd.Series(a[m]).diff(k).fillna(0.0).to_numpy()
        return o

    sect = df[SECT_COLS].astype(float).to_numpy()
    lane = df[LANE_COLS].astype(float).to_numpy()
    delta = np.stack([diff(sect[:, j], k) for k in (1, 3, 5) for j in range(5)], 1)
    have_corr = all(c in df.columns for c in CORR_COLS)
    corr = df[CORR_COLS].astype(float).to_numpy() if have_corr else None
    return {"sect": sect, "lane": lane, "delta": delta, "corr": corr, "grp": grp}


def make_tasks(df: pd.DataFrame, F: dict) -> list[tuple]:
    t = df["target_throttle"].astype(float).to_numpy()
    s = df["target_steering"].astype(float).to_numpy()
    stop = np.abs(t) < 0.05
    sect, lane, delta = F["sect"], F["lane"], F["delta"]
    tasks = [
        ("stop",       "정지 판정 · 라이다 5",        sect,                          stop, "clf"),
        ("stop_d",     "정지 판정 · 라이다 5 + Δ",    np.hstack([sect, delta]),      stop, "clf"),
        ("steer",      "조향 · 라이다 5 + Δ",         np.hstack([sect, delta]),      s,    "reg"),
        ("steer_lane", "조향 · 차선 72",              lane,                          s,    "reg"),
        ("steer_all",  "조향 · 차선 + 라이다 + Δ",     np.hstack([lane, sect, delta]), s,   "reg"),
        ("thr",        "스로틀 · 차선 + 라이다 + Δ",    np.hstack([lane, sect, delta]), t,   "reg"),
    ]
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# 측정
# ─────────────────────────────────────────────────────────────────────────────
def score_once(X, y, kind, mtr, mte) -> float | None:
    if kind == "clf":
        if len(np.unique(y[mtr])) < 2 or len(np.unique(y[mte])) < 2:
            return None
        m = GC(max_iter=200).fit(X[mtr], y[mtr])
        return balanced_accuracy_score(y[mte], m.predict(X[mte]))
    m = GR(max_iter=200).fit(X[mtr], y[mtr])
    return r2_score(y[mte], m.predict(X[mte]))


def learning_curve(X, y, kind, grp, sizes, repeats, rng):
    """각 크기에서 (평균, 표준오차) 를 함께 돌려준다.

    표준오차를 같이 봐야 '안 오른다' 와 '모른다' 를 구분할 수 있다.
    마지막 두 점의 차이가 표준오차 안에 있으면 판정하지 않는다.
    """
    courses = np.unique(grp)
    means, sems = [], []
    for n in sizes:
        if n >= len(courses):
            means.append(np.nan); sems.append(np.nan); continue
        sc = []
        for _ in range(repeats):
            tr = rng.choice(courses, size=n, replace=False)
            te = np.setdiff1d(courses, tr)
            v = score_once(X, y, kind, np.isin(grp, tr), np.isin(grp, te))
            if v is not None:
                sc.append(v)
        if sc:
            means.append(float(np.mean(sc)))
            sems.append(float(np.std(sc, ddof=1) / np.sqrt(len(sc))) if len(sc) > 1 else np.nan)
        else:
            means.append(np.nan); sems.append(np.nan)
    return means, sems


def verdict(curve, sems, sizes) -> tuple[str, float, float]:
    """전체 추세로 판정한다. 마지막 두 점만 보면 잡음 한 번에 뒤집힌다.

    log(코스 수) 에 대한 선형 회귀 기울기를 쓰고, 그 기울기를 코스당으로 환산한다.
    학습곡선은 보통 표본 수의 로그에 선형이므로 두 점 기울기보다 안정적이다.
    기울기의 크기가 관측 잡음(표준오차)보다 작으면 '판정불가' 로 둔다 —
    '안 오른다' 와 '모른다' 를 섞지 않기 위해.
    """
    ok = [(s, c, e) for s, c, e in zip(sizes, curve, sems) if not np.isnan(c)]
    if len(ok) < 3:
        return "판정불가(점부족)", float("nan"), float("nan")
    xs = np.log(np.array([o[0] for o in ok], dtype=float))
    ys = np.array([o[1] for o in ok])
    b = float(np.polyfit(xs, ys, 1)[0])           # log 코스당 상승폭
    # 마지막 구간을 코스당으로 환산
    s_last = ok[-1][0]
    slope = b / s_last
    noise = float(np.nanmean([o[2] for o in ok]))
    # 전체 상승폭이 잡음의 2배도 안 되면 판정하지 않는다
    span = ys[-1] - ys[0]
    if not np.isnan(noise) and abs(span) < 2 * noise:
        return "판정불가(잡음)", slope, noise
    if slope >= SLOPE_SHORT:
        return "데이터 부족", slope, noise
    if slope >= SLOPE_LIMIT:
        return "수확체감", slope, noise
    return "정보 한계", slope, noise


def course_contribution(X, y, kind, grp, names, repeats, rng):
    """코스를 하나 빼고 학습 -> 나머지 전체로 평가. 많이 떨어질수록 그 코스가 중요."""
    courses = np.unique(grp)
    base = []
    for _ in range(repeats):
        tr = rng.choice(courses, size=len(courses) - 2, replace=False)
        te = np.setdiff1d(courses, tr)
        v = score_once(X, y, kind, np.isin(grp, tr), np.isin(grp, te))
        if v is not None:
            base.append(v)
    b = float(np.mean(base)) if base else float("nan")
    rows = []
    for c in courses:
        pool = courses[courses != c]
        sc = []
        for _ in range(repeats):
            tr = rng.choice(pool, size=len(courses) - 3, replace=False)
            te = np.setdiff1d(courses, tr)
            v = score_once(X, y, kind, np.isin(grp, tr), np.isin(grp, te))
            if v is not None:
                sc.append(v)
        rows.append((names.get(c, str(c)), float(np.mean(sc)) if sc else float("nan")))
    return b, rows


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="데이터가 부족한가 정보가 부족한가")
    ap.add_argument("--dirs", nargs="+", default=["data", "raw"], help="코스 CSV 폴더")
    ap.add_argument("--task", default=None, help="한 과제만 (stop/stop_d/steer/steer_lane/steer_all/thr)")
    ap.add_argument("--repeats", type=int, default=10, help="무작위 분할 반복 수")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="학습 코스 수 목록")
    ap.add_argument("--contrib", action="store_true", help="코스별 기여도까지 계산 (느림)")
    ap.add_argument("--plot", type=Path, default=None, help="학습곡선 PNG 저장 경로")
    ap.add_argument("--slope-th", type=float, default=SLOPE_SHORT)
    args = ap.parse_args()

    df = load_all(args.dirs)
    if df.empty:
        print("[data_sufficiency] 코스 CSV 를 찾지 못했습니다.")
        return
    F = build_features(df)
    grp = F["grp"]
    names = {int(i): str(n) for i, n in
             df.groupby("_src_idx")["_source_file"].first().items()}
    ncourse = len(np.unique(grp))
    sizes = args.sizes or [s for s in (2, 4, 6, 8, 10, 12, 14, 16) if s < ncourse]
    if not sizes:
        sizes = [max(1, ncourse - 1)]

    print("=" * 92)
    print("data_sufficiency — 코스 %d개 / %d행 / 반복 %d회" % (ncourse, len(df), args.repeats))
    print("  판정: 코스당 기울기 >= %.3f 데이터부족 / >= %.3f 수확체감 / 미만 정보한계"
          % (args.slope_th, SLOPE_LIMIT))
    print("=" * 92)

    tasks = make_tasks(df, F)
    if args.task:
        tasks = [t for t in tasks if t[0] == args.task]
        if not tasks:
            print("[data_sufficiency] 그런 과제가 없습니다.")
            return

    rng = np.random.default_rng(0)
    print("%-28s%s%11s%9s%16s" % ("과제", "".join("%9s" % f"{n}" for n in sizes),
                                  "코스당기울기", "잡음", "판정"))
    print("%-28s%s" % ("  (괄호는 표준오차)", "".join("%9s" % "코스" for _ in sizes)))
    results = {}
    for key, label, X, y, kind in tasks:
        curve, sems = learning_curve(X, y, kind, grp, sizes, args.repeats, rng)
        v, slope, noise = verdict(curve, sems, sizes)
        results[key] = (label, curve, sems, slope, v)
        print("%-28s%s%+11.4f%9.3f%16s"
              % (label, "".join("%9.3f" % c for c in curve), slope, noise, v))
        print("%-28s%s" % ("", "".join("%9s" % ("±%.2f" % e if not np.isnan(e) else "-")
                                       for e in sems)))

    if args.contrib:
        print()
        print("=" * 92)
        print("코스별 기여도 — 그 코스를 빼면 성능이 얼마나 떨어지나")
        print("  크게 떨어짐 = 유일한 정보원. 비슷한 코스를 더 찍어야 한다")
        print("  안 떨어짐   = 이미 중복. 더 찍어도 소용없다")
        print("=" * 92)
        for key, label, X, y, kind in tasks:
            b, rows = course_contribution(X, y, kind, grp, names, max(4, args.repeats // 2), rng)
            print()
            print("[%s]  전체 사용 시 %.3f" % (label, b))
            rows.sort(key=lambda r: r[1])
            for nm, v in rows:
                print("   %-28s %.3f   %+.3f" % (nm, v, v - b))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for fam in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
                try:
                    import matplotlib.font_manager as fm
                    fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False)
                    plt.rcParams["font.family"] = fam
                    break
                except Exception:
                    pass
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
            for key, (label, curve, sems, slope, v) in results.items():
                c = np.array(curve, dtype=float)
                e = np.array([0.0 if np.isnan(x) else x for x in sems], dtype=float)
                ax.errorbar(sizes, c, yerr=e, fmt="o-", lw=1.8, ms=5, capsize=3,
                            label="%s  [%s]" % (label, v))
            ax.set_xlabel("학습에 쓴 코스 수")
            ax.set_ylabel("hold-out 성능 (분류=balanced acc, 회귀=R²)")
            ax.set_title("데이터를 늘리면 좋아지는가", fontsize=11)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.savefig(args.plot, bbox_inches="tight", facecolor="white")
            print()
            print("[SAVE] %s" % args.plot)
        except Exception as e:  # noqa: BLE001
            print("[plot] 실패:", e)


if __name__ == "__main__":
    main()
