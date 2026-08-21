#!/usr/bin/env python3
"""
saliency.py — 학습된 planner 가 "무엇을 보고" 결정하는지 분석한다.
================================================================

픽셀 기반 saliency 와 달리 이 모델의 입력은 이름이 붙은 119개 숫자다.
    objects 40 / lane 72 (6x12 공간격자) / lidar 5 / ego 2
따라서 "이 픽셀이 중요"가 아니라 "lidar_s2 가 조향을 -0.31 밀었다" 처럼
바로 읽히는 귀속(attribution)이 나온다.

계산하는 것 세 가지
    [1] 그룹 차폐    obj/lane/lidar/ego 를 통째로 지우고 출력 변화 측정
                     -> 모델이 그 센서를 실제로 쓰는지 판별한다
    [2] 라이다 섹터  s0~s4 를 하나씩 차폐 -> 어느 방향이 결정적인가
    [3] lane 격자    6x12 각 칸을 차폐 -> 화면 어느 영역이 조향을 만드는가

[왜 기울기가 아니라 차폐인가]
기울기(gradient)는 현재 지점의 국소 민감도만 보여준다. 반면 차폐는
"그 입력이 없었다면 출력이 어떻게 달라지는가" 라는 반사실을 직접 측정한다.
모델이 어떤 센서를 통째로 무시하고 있는지는 차폐로만 드러난다 —
기울기는 작지만 0이 아닌 값을 내놓아 "조금은 쓰고 있다"고 오해하게 만든다.

[부호를 반드시 살린다]
크기만 보면 "이 칸이 중요하다"까지만 알 수 있다. 정작 알고 싶은 것은
"왼쪽으로 미는가 오른쪽으로 미는가" 이므로 모든 시각화에서 부호를 유지한다.
터미널은 좌/우 양방향 막대로, HTML 은 발산형 색상(파랑=좌, 빨강=우)으로 그린다.

[연산량]
153K 파라미터 MLP 라 순전파 1회가 1ms 미만이다. 격자 72칸을 모두 돌려도
프레임당 0.1초 수준이라 오프라인 분석에 충분하다. 주행 루프와는 무관하다.

[조향 부호]  + 왼쪽 / - 오른쪽
spline_expert.py 의 좌표계가 x = d*cos(theta), y = d*sin(theta) 이고
index 100~400 (36~144도) 이 좌측이므로 좌측 점은 y > 0 이다.
steering = atan2(y, x) 이므로 목표점이 왼쪽이면 양수가 된다.
실측 corr(조향, 차선 무게중심) = -0.16 도 이 부호와 일치한다.

사용법
    python3 saliency.py --frame 500                    한 프레임 귀속
    python3 saliency.py --frame 500 --explain          픽셀·각도 단위 자연어 설명
    python3 saliency.py --summary                      전체 평균
    python3 saliency.py --by-course                    코스(CSV)별로 나눠 비교
    python3 saliency.py --by-course --html report.html 브라우저용 리포트

    --html 은 모든 모드에서 쓸 수 있다. 외부 의존성 없는 단일 파일이라
    그대로 열린다.
"""

import argparse
import html as _html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from planner_model import (
    PlannerModel, row_to_tensors,
    GRID_ROWS, GRID_COLS, FRAME_W, FRAME_H,
)

BLOCKS = ("objects", "lane", "lidar", "ego")

# 섹터 이름과 실제 각도 범위(도). collect_data_planner.py 의 인덱스를
# 0.36 deg/idx 로 환산한 값이며, 양수가 왼쪽이다.
LIDAR_SECTORS = [
    ("s0 left ", 29.9, 59.8),
    ("s1 fl   ", 9.7, 29.9),
    ("s2 front", -10.1, 9.7),
    ("s3 fr   ", -30.2, -10.1),
    ("s4 right", -60.1, -30.2),
]


# ─────────────────────────────────────────────────────────────────────────────
# 계산
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, tensors):
    with torch.no_grad():
        out = model(*tensors)
    return float(out[0, 0]), float(out[0, 1])


def occlude(model, tensors, which, index=None):
    """블록(또는 그 안의 원소 하나)을 중립값으로 바꾸고 예측한다.

    무엇이 "정보 없음"인지는 특징마다 다르다. lane 격자는 0 = 차선 픽셀
    없음 이라 자연스럽지만, lidar 는 0 = 거리 0m 라 오히려 "코앞에 벽"을
    뜻해서 정반대 의미가 된다. 그래서 lidar 만 그 프레임의 최댓값
    (= 가장 열려 있는 방향)으로 채운다.
    """
    objects, lane, lidar, ego, scen = [t.clone() for t in tensors]
    target = {"objects": objects, "lane": lane, "lidar": lidar, "ego": ego}[which]
    fill = float(lidar.max()) if which == "lidar" else 0.0
    if index is None:
        target[:] = fill
    else:
        target[0, index] = fill
    return predict(model, (objects, lane, lidar, ego, scen))


def analyse_frame(model, row):
    """한 프레임의 귀속을 계산한다. 모든 값은 (원래출력 - 차폐후출력) 이라
    양수면 그 입력이 출력을 양(+, 왼쪽) 방향으로 밀고 있었다는 뜻이다."""
    tensors = row_to_tensors(row)
    s0, t0 = predict(model, tensors)

    res = {"pred": (s0, t0), "group": {}, "lidar": [], "lane": None}

    for blk in BLOCKS:
        s, t = occlude(model, tensors, blk)
        res["group"][blk] = (s0 - s, t0 - t)

    for i in range(5):
        s, t = occlude(model, tensors, "lidar", i)
        res["lidar"].append((s0 - s, t0 - t))

    grid = np.zeros((GRID_ROWS, GRID_COLS))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            s, _ = occlude(model, tensors, "lane", r * GRID_COLS + c)
            grid[r, c] = s0 - s
    res["lane"] = grid
    return res


def aggregate(model, df, limit):
    """여러 프레임 평균. 부호가 상쇄되지 않도록 절댓값 평균과 부호 평균을
    둘 다 남긴다 — 전자는 '얼마나 관여하는가', 후자는 '어느 쪽으로 미는가'."""
    step = max(1, len(df) // max(limit, 1))
    sub = df.iloc[::step]

    absg = {b: np.zeros(2) for b in BLOCKS}
    sgng = {b: np.zeros(2) for b in BLOCKS}
    absl = np.zeros(5)
    sgnl = np.zeros(5)
    absgrid = np.zeros((GRID_ROWS, GRID_COLS))
    sgngrid = np.zeros((GRID_ROWS, GRID_COLS))
    n = 0
    for _, row in sub.iterrows():
        r = analyse_frame(model, row)
        for b in BLOCKS:
            absg[b] += np.abs(r["group"][b])
            sgng[b] += np.array(r["group"][b])
        lv = np.array([x[0] for x in r["lidar"]])
        absl += np.abs(lv)
        sgnl += lv
        absgrid += np.abs(r["lane"])
        sgngrid += r["lane"]
        n += 1
    if n == 0:
        return None
    return {
        "n": n,
        "group_abs": {b: absg[b] / n for b in BLOCKS},
        "group_sgn": {b: sgng[b] / n for b in BLOCKS},
        "lidar_abs": absl / n,
        "lidar_sgn": sgnl / n,
        "lane_abs": absgrid / n,
        "lane_sgn": sgngrid / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 터미널 출력 — 부호를 좌/우 양방향 막대로 표현한다
# ─────────────────────────────────────────────────────────────────────────────

HALF = 14


def dual_bar(value, scale):
    """중앙 기준 좌우 막대. 왼쪽(+)은 왼쪽으로, 오른쪽(-)은 오른쪽으로 뻗는다."""
    if scale <= 1e-12:
        n = 0
    else:
        n = int(round(abs(value) / scale * HALF))
    n = min(n, HALF)
    if value >= 0:
        return (" " * (HALF - n)) + ("<" * n) + "|" + (" " * HALF)
    return (" " * HALF) + "|" + (">" * n) + (" " * (HALF - n))


def draw_grid_signed(grid, label):
    """6x12 격자. 왼쪽으로 미는 칸은 <, 오른쪽으로 미는 칸은 >, 세기는 농담."""
    mx = float(np.abs(grid).max()) or 1e-12
    lo = " .:-=+*#%@"
    print("      " + label)
    print("            " + "left" + " " * (GRID_COLS - 8) + "right")
    for r in range(GRID_ROWS):
        if r == 0:
            tag = "far "
        elif r == GRID_ROWS - 1:
            tag = "near"
        else:
            tag = "    "
        cells = ""
        for c in range(GRID_COLS):
            v = grid[r, c]
            lvl = min(int(abs(v) / mx * (len(lo) - 1)), len(lo) - 1)
            cells += lo[lvl]
        print("      " + tag + " |" + cells + "|")
    print("      최대 |기여| " + format(mx, ".4f"))


def draw_grid_dir(grid):
    """방향 전용 뷰: 왼쪽으로 미는 칸 '<', 오른쪽 '>', 미미하면 '.'"""
    mx = float(np.abs(grid).max()) or 1e-12
    print("      방향 (< 왼쪽 / > 오른쪽 / . 미미)")
    for r in range(GRID_ROWS):
        if r == 0:
            tag = "far "
        elif r == GRID_ROWS - 1:
            tag = "near"
        else:
            tag = "    "
        cells = ""
        for c in range(GRID_COLS):
            v = grid[r, c]
            if abs(v) < mx * 0.15:
                cells += "."
            elif v > 0:
                cells += "<"
            else:
                cells += ">"
        print("      " + tag + " |" + cells + "|")


def report_frame(row, res):
    s0, t0 = res["pred"]
    ts = float(row["target_steering"])
    tt = float(row["target_throttle"])
    fid = int(row["frame_id"]) if "frame_id" in row.index else -1

    print("=" * 72)
    print("  frame " + str(fid) + "        조향 부호:  + 왼쪽  /  - 오른쪽")
    print("=" * 72)
    print("  예측  steer=" + format(s0, "+.3f") + "   thr=" + format(t0, "+.3f"))
    print("  정답  steer=" + format(ts, "+.3f") + "   thr=" + format(tt, "+.3f"))
    print("  오차  steer=" + format(s0 - ts, "+.3f")
          + "   thr=" + format(t0 - tt, "+.3f"))

    print("")
    print("  [1] 그룹 차폐 — 지우면 조향이 어느 쪽으로 얼마나 움직이나")
    print("      " + " " * 10 + "left" + " " * (HALF * 2 - 8) + "right")
    g = res["group"]
    sc = max(abs(v[0]) for v in g.values()) or 1e-12
    for blk in BLOCKS:
        ds, dt = g[blk]
        print("      " + blk.ljust(9) + dual_bar(ds, sc)
              + "  " + format(ds, "+.4f") + "   thr " + format(dt, "+.4f"))
    dead = [b for b in BLOCKS if abs(g[b][0]) < 1e-4 and abs(g[b][1]) < 1e-4]
    if dead:
        print("      * 사실상 무시되는 입력: " + ", ".join(dead))

    print("")
    print("  [2] 라이다 섹터별 조향 기여  (실제 각도 표기)")
    sc = max(abs(v[0]) for v in res["lidar"]) or 1e-12
    for i, (nm, a1, a2) in enumerate(LIDAR_SECTORS):
        ds = res["lidar"][i][0]
        ang = "[" + format(a1, "+5.1f") + "~" + format(a2, "+5.1f") + "]"
        print("      " + nm + " " + ang + " " + dual_bar(ds, sc)
              + "  " + format(ds, "+.4f"))

    print("")
    print("  [3] lane 6x12 조향 기여")
    draw_grid_signed(res["lane"], "세기")
    print("")
    draw_grid_dir(res["lane"])


def by_course(model, data_dir, limit):
    """코스(CSV) 별로 따로 귀속을 재고 비교한다.

    코스마다 센서 환경이 완전히 다르다 — 개활 코스는 정면 라이다 baseline 이
    4.6m 인데 종이컵 협로는 0.8m 다. 전체를 한 덩어리로 평균 내면 그 차이가
    상쇄돼 "모델이 어느 코스에서 무엇을 보는가" 가 보이지 않는다.
    파일 단위로 나눠야 협로에서는 라이다를, 개활에서는 차선을 보는지
    같은 질문에 답할 수 있다.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_course*.csv"))
    out = []
    for f in files:
        df = pd.read_csv(f).dropna().reset_index(drop=True)
        if df.empty:
            continue
        agg = aggregate(model, df, limit)
        if agg is None:
            continue
        # 이 코스의 센서 성격도 함께 기록해 둔다 (해석에 필요)
        side = df[["lidar_s0", "lidar_s4"]].min(axis=1)
        lane_cols = ["lane_r" + str(r) + "c" + str(c)
                     for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        agg["name"] = f.stem
        agg["front_med"] = float(df["lidar_s2"].median())
        agg["side_med"] = float(side.median())
        agg["lane_med"] = float(df[lane_cols].sum(axis=1).median())
        agg["rows"] = len(df)
        out.append(agg)
    return out


def report_by_course(courses):
    print("=" * 78)
    print("  코스별 귀속 비교   (조향 |기여도|)")
    print("=" * 78)
    print("  코스마다 센서 환경이 달라 전체 평균으로는 보이지 않는 차이를 본다.")
    print("")
    print("  " + "코스".ljust(22) + "행수   정면m  측면m  차선   "
          + "lane   lidar  ego    obj")
    print("  " + "-" * 74)
    for a in courses:
        g = a["group_abs"]
        print("  " + a["name"].ljust(22)
              + str(a["rows"]).rjust(5) + "  "
              + format(a["front_med"], "5.2f") + "  "
              + format(a["side_med"], "5.2f") + "  "
              + format(a["lane_med"], "5.3f") + "  "
              + format(g["lane"][0], ".4f") + " "
              + format(g["lidar"][0], ".4f") + " "
              + format(g["ego"][0], ".4f") + " "
              + format(g["objects"][0], ".4f"))

    print("")
    print("  라이다 섹터별 |기여도|")
    print("  " + "코스".ljust(22)
          + "".join(nm.strip().ljust(9) for nm, _, _ in LIDAR_SECTORS))
    print("  " + "-" * 74)
    for a in courses:
        print("  " + a["name"].ljust(22)
              + "".join(format(v, ".4f").ljust(9) for v in a["lidar_abs"]))

    for a in courses:
        print("")
        print("  " + a["name"] + "  —  lane 기여 방향")
        draw_grid_dir(a["lane_sgn"])


def write_html_courses(path, courses, note):
    """코스별 섹션을 세로로 쌓은 리포트. 각 섹션에 부채꼴 + 격자."""
    secs = []
    for a in courses:
        g = a["group_sgn"]
        gmx = max(abs(g[b][0]) for b in BLOCKS) or 1e-12
        lsg = list(a["lidar_sgn"])
        lmx = max(abs(v) for v in lsg) or 1e-12
        rows = "".join(
            _bar_row(b, g[b][0], gmx, "thr " + format(g[b][1], "+.3f"))
            for b in BLOCKS
        )
        secs.append(
            '<section class="course"><h3>' + _html.escape(a["name"]) + "</h3>"
            + '<p class="cmeta">' + str(a["rows"]) + " rows &nbsp;·&nbsp; 정면 "
            + format(a["front_med"], ".2f") + "m &nbsp;·&nbsp; 측면 "
            + format(a["side_med"], ".2f") + "m &nbsp;·&nbsp; 차선 "
            + format(a["lane_med"], ".3f") + "</p>"
            + '<div class="split">'
            + '<div class="col"><h4>블록</h4>' + rows + "</div>"
            + '<div class="col"><h4>라이다 (실제 각도)</h4>'
            + '<svg viewBox="0 0 400 240" class="fan">'
            + _lidar_fan(lsg, lmx) + "</svg></div>"
            + '<div class="col"><h4>차선 격자</h4>'
            + _grid_svg(a["lane_sgn"]) + "</div>"
            + "</div></section>"
        )

    doc = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>코스별 귀속 비교</title><style>
:root{--bg:#f2f5f7;--fg:#0f151c;--mut:#5f6a75;--edge:#d5dbe1;--hair:#e6eaee;
--card:#fff;--left:#0d7d8c;--right:#bd5636}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#10151a;--fg:#e4e9ee;--mut:#8b96a2;--edge:#28313a;--hair:#1e262e;
--card:#161c23;--left:#2ba7b8;--right:#e0764f}}
:root[data-theme="dark"]{--bg:#10151a;--fg:#e4e9ee;--mut:#8b96a2;
--edge:#28313a;--hair:#1e262e;--card:#161c23;--left:#2ba7b8;--right:#e0764f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:400 15px/1.6 "IBM Plex Sans",ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:36px 20px 80px}
h1{font:700 30px/1.15 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
margin:0 0 8px}
.lede{color:var(--mut);margin:0 0 8px;max-width:62ch;font-size:14.5px}
.course{border-top:1px solid var(--edge);padding-top:22px;margin-top:34px}
h3{font:600 17px/1.2 "IBM Plex Mono",monospace;margin:0 0 4px}
.cmeta{color:var(--mut);font:400 12.5px "IBM Plex Mono",monospace;margin:0 0 16px}
h4{font:600 10.5px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;
text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.split{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)
minmax(0,1fr);gap:26px;align-items:start}
@media(max-width:900px){.split{grid-template-columns:1fr}}
.row{display:flex;align-items:center;gap:8px;margin:5px 0}
.nm{width:62px;font:12px ui-monospace,monospace;color:var(--mut);flex:none}
.track{position:relative;flex:1;height:17px;background:var(--card);
border:1px solid var(--edge);border-radius:3px;overflow:hidden}
.track::after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;
background:var(--edge)}
.track i{position:absolute;top:0;bottom:0;display:block}
.val{width:60px;text-align:right;font:12px ui-monospace,monospace;flex:none}
.extra{width:74px;font:11px ui-monospace,monospace;color:var(--mut);flex:none}
.fan{width:100%;height:auto;display:block;overflow:visible}
.wl{font:500 11px ui-monospace,monospace;fill:var(--fg);text-anchor:middle}
.wv{font:400 10px ui-monospace,monospace;fill:var(--mut);text-anchor:middle}
.rings{fill:none;stroke:var(--hair);stroke-width:1}
.axisline{stroke:var(--edge);stroke-width:1;stroke-dasharray:3 4}
.grid{width:100%;height:auto;display:block}
.key{display:flex;gap:18px;font:400 12.5px ui-monospace,monospace;
color:var(--mut);margin:14px 0 0}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block;
vertical-align:-1px;margin-right:5px}
footer{margin-top:48px;padding-top:14px;border-top:1px solid var(--edge);
color:var(--mut);font-size:12.5px}
</style></head><body><div class="wrap">
<h1>코스별 귀속 비교</h1>
<p class="lede">코스마다 센서 환경이 다르다 — 개활 코스는 정면 라이다 baseline 이 4.6m 인데
종이컵 협로는 0.8m 다. 전체를 한 덩어리로 평균 내면 그 차이가 상쇄되므로 파일 단위로 나눠 잰다.</p>
<div class="key"><span><i class="sw" style="background:var(--left)"></i>왼쪽으로 밈</span>
<span><i class="sw" style="background:var(--right)"></i>오른쪽으로 밈</span></div>
""" + "".join(secs) + """
<footer>""" + _html.escape(note) + """</footer>
</div></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def explain_frame(row, res):
    """사람이 읽는 문장으로 판단 근거를 설명한다.

    격자 인덱스를 실제 화면 픽셀 좌표로, 섹터 번호를 실제 각도와 거리로
    되돌려 준다. 숫자표만으로는 "r5c9 가 0.018" 이 무슨 뜻인지 알 수 없고,
    현장에서 카메라 화면과 대조하려면 픽셀 좌표가 있어야 하기 때문이다.
    """
    s0, t0 = res["pred"]
    ts = float(row["target_steering"])
    tt = float(row["target_throttle"])
    fid = int(row["frame_id"]) if "frame_id" in row.index else -1

    cw = FRAME_W // GRID_COLS      # 848 / 12 = 70 px
    chh = FRAME_H // GRID_ROWS     # 480 /  6 = 80 px

    print("=" * 72)
    print("  frame " + str(fid) + " 판단 설명")
    print("=" * 72)

    # ── 센서가 무엇을 봤는가 ──────────────────────────────────────
    print("  [센서 관측]")
    for i, (nm, a1, a2) in enumerate(LIDAR_SECTORS):
        dist = float(row["lidar_s" + str(i)])
        if dist < 0.45:
            state = "위협"
        elif dist < 0.80:
            state = "근접"
        elif dist < 2.0:
            state = "여유"
        else:
            state = "열림"
        print("      " + nm + " " + format(dist, "5.2f") + "m"
              + "  [" + format(a1, "+5.1f") + "°~" + format(a2, "+5.1f") + "°]"
              + "  " + state)

    lane_cols = ["lane_r" + str(r) + "c" + str(c)
                 for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    lane_vals = np.array([float(row[k]) for k in lane_cols]).reshape(
        GRID_ROWS, GRID_COLS)
    lane_sum = float(lane_vals.sum())
    if lane_sum < 1e-6:
        print("      차선     검출 없음 (BiSeNet 출력이 전부 0)")
    else:
        col_w = lane_vals.sum(axis=0)
        com = float((col_w * np.arange(GRID_COLS)).sum() / max(col_w.sum(), 1e-9))
        px = int((com + 0.5) * cw)
        side = "왼쪽" if com < 5.0 else ("오른쪽" if com > 6.0 else "중앙")
        print("      차선     합 " + format(lane_sum, ".3f")
              + "  무게중심 열 " + format(com, ".1f") + "/11"
              + "  (화면 x≈" + str(px) + "px, " + side + ")")

    # ── 모델이 무엇을 결정했는가 ─────────────────────────────────
    print("")
    print("  [모델 결정]")
    turn = "왼쪽" if s0 > 0.15 else ("오른쪽" if s0 < -0.15 else "거의 직진")
    move = "정지" if t0 < 0.1 else ("서행" if t0 < 0.6 else "주행")
    print("      steer " + format(s0, "+.3f") + "  -> " + turn
          + "        (정답 " + format(ts, "+.3f") + ")")
    print("      thr   " + format(t0, "+.3f") + "  -> " + move
          + "        (정답 " + format(tt, "+.3f") + ")")

    # ── 왜 그렇게 결정했는가 ─────────────────────────────────────
    print("")
    print("  [판단 근거]")
    g = res["group"]
    order = sorted(BLOCKS, key=lambda b: -abs(g[b][0]))
    top = order[0]
    tot = sum(abs(g[b][0]) for b in BLOCKS) or 1e-12
    print("      조향은 " + top + " 가 주도 ("
          + format(abs(g[top][0]) / tot * 100, ".0f") + "% 기여, "
          + format(g[top][0], "+.4f") + ")")
    for b in order[1:]:
        if abs(g[b][0]) < 1e-4:
            print("      " + b + " 는 이 프레임에서 출력에 영향 없음")
        else:
            print("      " + b + " " + format(g[b][0], "+.4f")
                  + " (" + format(abs(g[b][0]) / tot * 100, ".0f") + "%)")

    # 라이다 최대 기여 섹터
    li = int(np.argmax([abs(v[0]) for v in res["lidar"]]))
    lv = res["lidar"][li][0]
    nm, a1, a2 = LIDAR_SECTORS[li]
    push = "왼쪽" if lv > 0 else "오른쪽"
    print("      라이다 중에서는 " + nm.strip()
          + " (" + format(a1, "+.0f") + "°~" + format(a2, "+.0f") + "°, "
          + format(float(row["lidar_s" + str(li)]), ".2f") + "m) 가 "
          + push + "으로 " + format(lv, "+.4f"))

    # 차선 격자 최대 기여 칸 -> 픽셀
    grid = res["lane"]
    r, c = np.unravel_index(int(np.argmax(np.abs(grid))), grid.shape)
    v = float(grid[r, c])
    push = "왼쪽" if v > 0 else "오른쪽"
    depth = "먼 곳" if r <= 1 else ("중간" if r <= 3 else "가까운 곳")
    lr = "좌" if c <= 3 else ("우" if c >= 8 else "중앙")
    print("      차선 격자에서는 r" + str(r) + "c" + str(c)
          + " (" + depth + "/" + lr + ", 화면 x " + str(c * cw) + "~"
          + str((c + 1) * cw) + "px, y " + str(r * chh) + "~"
          + str((r + 1) * chh) + "px) 가 " + push + "으로 "
          + format(v, "+.4f"))

    # ── 경고 ────────────────────────────────────────────────────
    notes = []
    dead = [b for b in BLOCKS if abs(g[b][0]) < 1e-4 and abs(g[b][1]) < 1e-4]
    if dead:
        notes.append("입력 " + ", ".join(dead) + " 이(가) 출력에 전혀 기여하지 않음")
    if abs(g["ego"][1]) > abs(g["lane"][1]) + abs(g["lidar"][1]):
        notes.append("스로틀을 센서가 아니라 직전 스로틀(ego)이 주로 결정함 "
                     "— 관성 복사이지 판단이 아님")
    front = float(row["lidar_s2"])
    if front < 0.45 and t0 > 0.6:
        notes.append("정면 " + format(front, ".2f") + "m 로 막혔는데 스로틀 "
                     + format(t0, ".2f") + " 유지 — 정지를 배우지 못한 신호")
    if notes:
        print("")
        print("  [경고]")
        for n in notes:
            print("      * " + n)


def report_summary(agg):
    print("=" * 72)
    print("  평균 기여도   (" + str(agg["n"]) + " 프레임 표본)")
    print("=" * 72)
    print("  |기여| = 얼마나 관여하는가 / 부호평균 = 평균적으로 어느 쪽으로 미는가")
    print("")
    sc = max(agg["group_abs"][b][0] for b in BLOCKS) or 1e-12
    print("      " + "block".ljust(9) + "|steer|   부호평균")
    for b in BLOCKS:
        av = agg["group_abs"][b][0]
        sv = agg["group_sgn"][b][0]
        print("      " + b.ljust(9) + format(av, ".4f") + "   "
              + format(sv, "+.4f") + "  " + ("#" * int(av / sc * 20)))
    dead = [b for b in BLOCKS
            if agg["group_abs"][b][0] < 1e-4 and agg["group_abs"][b][1] < 1e-4]
    if dead:
        print("      * 모델이 통째로 무시하는 입력: " + ", ".join(dead))

    print("")
    print("  라이다 섹터별")
    sc = max(agg["lidar_abs"]) or 1e-12
    for i, (nm, a1, a2) in enumerate(LIDAR_SECTORS):
        av = agg["lidar_abs"][i]
        sv = agg["lidar_sgn"][i]
        print("      " + nm + " " + format(av, ".4f") + "   "
              + format(sv, "+.4f") + "  " + ("#" * int(av / sc * 20)))

    print("")
    draw_grid_signed(agg["lane_abs"], "lane 평균 |기여|")
    print("")
    draw_grid_dir(agg["lane_sgn"])


# ─────────────────────────────────────────────────────────────────────────────
# HTML 리포트 — 브라우저로 보는 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _diverging(v, mx):
    """발산형 색상. 양수(왼쪽)=파랑, 음수(오른쪽)=빨강, 0=중립회색."""
    if mx <= 1e-12:
        t = 0.0
    else:
        t = max(-1.0, min(1.0, v / mx))
    if t >= 0:
        r = int(240 - 150 * t)
        g = int(243 - 70 * t)
        b = int(246 + 9 * t)
    else:
        u = -t
        r = int(240 + 15 * u)
        g = int(243 - 150 * u)
        b = int(246 - 170 * u)
    return "rgb(" + str(r) + "," + str(g) + "," + str(b) + ")"


def _lidar_fan(values, mx):
    """라이다 5섹터를 실제 각도의 부채꼴로 그린다. 위쪽이 차량 전방."""
    import math
    cx, cy, r0, r1 = 200.0, 205.0, 26.0, 165.0
    parts = []
    for i, (nm, a1, a2) in enumerate(LIDAR_SECTORS):
        col = _diverging(values[i], mx)
        p = []
        # 각도 theta 는 전방 기준, 양수가 왼쪽. 화면에서 전방은 위(-y), 왼쪽은 -x.
        steps = 14
        for k in range(steps + 1):
            th = math.radians(a1 + (a2 - a1) * k / steps)
            p.append((cx - math.sin(th) * r1, cy - math.cos(th) * r1))
        for k in range(steps, -1, -1):
            th = math.radians(a1 + (a2 - a1) * k / steps)
            p.append((cx - math.sin(th) * r0, cy - math.cos(th) * r0))
        pts = " ".join(format(x, ".1f") + "," + format(y, ".1f") for x, y in p)
        mid = math.radians((a1 + a2) / 2)
        lx = cx - math.sin(mid) * (r1 * 0.72)
        ly = cy - math.cos(mid) * (r1 * 0.72)
        parts.append(
            '<polygon points="' + pts + '" fill="' + col
            + '" stroke="var(--edge)" stroke-width="1"/>'
        )
        parts.append(
            '<text x="' + format(lx, ".1f") + '" y="' + format(ly, ".1f")
            + '" text-anchor="middle" class="fanlab">' + nm.strip() + "</text>"
        )
        parts.append(
            '<text x="' + format(lx, ".1f") + '" y="' + format(ly + 14, ".1f")
            + '" text-anchor="middle" class="fanval">'
            + format(values[i], "+.3f") + "</text>"
        )
    parts.append('<circle cx="200" cy="205" r="5" fill="var(--fg)"/>')
    parts.append('<text x="200" y="232" text-anchor="middle" class="fanlab">차량</text>')
    return ('<svg viewBox="0 0 400 240" class="fan">' + "".join(parts) + "</svg>")


def _grid_svg(grid):
    mx = float(np.abs(grid).max()) or 1e-12
    cw, ch = 34, 26
    parts = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            v = grid[r, c]
            parts.append(
                '<rect x="' + str(c * cw) + '" y="' + str(r * ch)
                + '" width="' + str(cw - 2) + '" height="' + str(ch - 2)
                + '" rx="3" fill="' + _diverging(v, mx) + '">'
                + "<title>r" + str(r) + "c" + str(c) + "  "
                + format(v, "+.4f") + "</title></rect>"
            )
    w = GRID_COLS * cw
    h = GRID_ROWS * ch
    return ('<svg viewBox="0 0 ' + str(w) + " " + str(h)
            + '" class="grid">' + "".join(parts) + "</svg>")


def _bar_row(name, value, mx, extra=""):
    pct = 0.0 if mx <= 1e-12 else min(abs(value) / mx, 1.0) * 50.0
    if value >= 0:
        style = "right:50%;width:" + format(pct, ".1f") + "%;background:var(--left)"
    else:
        style = "left:50%;width:" + format(pct, ".1f") + "%;background:var(--right)"
    return ('<div class="row"><span class="nm">' + _html.escape(name)
            + '</span><span class="track"><i style="' + style + '"></i></span>'
            + '<span class="val">' + format(value, "+.4f") + "</span>"
            + '<span class="extra">' + extra + "</span></div>")


def write_html(path, title, pred, target, group, lidar_vals, grid, note):
    gmx = max(abs(v[0]) for v in group.values()) or 1e-12
    lmx = max(abs(v) for v in lidar_vals) or 1e-12

    rows = "".join(
        _bar_row(b, group[b][0], gmx, "thr " + format(group[b][1], "+.3f"))
        for b in BLOCKS
    )
    lrows = "".join(
        _bar_row(LIDAR_SECTORS[i][0].strip(), lidar_vals[i], lmx,
                 "[" + format(LIDAR_SECTORS[i][1], "+.0f") + "~"
                 + format(LIDAR_SECTORS[i][2], "+.0f") + "°]")
        for i in range(5)
    )
    dead = [b for b in BLOCKS
            if abs(group[b][0]) < 1e-4 and abs(group[b][1]) < 1e-4]
    warn = ""
    if dead:
        warn = ('<p class="warn">모델이 사실상 무시하는 입력: <b>'
                + ", ".join(dead) + "</b></p>")

    head = ""
    if pred is not None:
        head = ('<div class="cards">'
                + '<div class="card"><h4>예측</h4><p>steer '
                + format(pred[0], "+.3f") + "<br>thr " + format(pred[1], "+.3f")
                + "</p></div>"
                + '<div class="card"><h4>정답</h4><p>steer '
                + format(target[0], "+.3f") + "<br>thr "
                + format(target[1], "+.3f") + "</p></div>"
                + '<div class="card"><h4>오차</h4><p>steer '
                + format(pred[0] - target[0], "+.3f") + "<br>thr "
                + format(pred[1] - target[1], "+.3f") + "</p></div></div>")

    doc = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + _html.escape(title) + """</title><style>
:root{--bg:#fbfbfd;--fg:#1a1c1f;--mut:#6b7280;--edge:#d8dbe0;--card:#fff;
--left:#3b82f6;--right:#ef4444;--warnbg:#fff7ed;--warnfg:#9a3412}
@media(prefers-color-scheme:dark){:root{--bg:#141619;--fg:#e8eaed;--mut:#9aa0a6;
--edge:#2c3036;--card:#1c1f24;--warnbg:#2a1d12;--warnfg:#fbbf24}}
*{box-sizing:border-box}body{margin:0;padding:28px 20px 60px;background:var(--bg);
color:var(--fg);font:15px/1.6 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:32px 0 10px;
text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.sub{color:var(--mut);margin:0 0 6px;font-size:13.5px}
.cards{display:flex;gap:10px;margin:18px 0}
.card{flex:1;background:var(--card);border:1px solid var(--edge);border-radius:10px;
padding:12px 14px}.card h4{margin:0 0 6px;font-size:12px;color:var(--mut);
text-transform:uppercase;letter-spacing:.05em}.card p{margin:0;font:13px/1.5
ui-monospace,SFMono-Regular,Menlo,monospace}
.row{display:flex;align-items:center;gap:10px;margin:5px 0}
.nm{width:78px;font:12.5px ui-monospace,monospace;color:var(--mut);flex:none}
.track{position:relative;flex:1;height:19px;background:var(--card);
border:1px solid var(--edge);border-radius:4px;overflow:hidden}
.track::after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;
background:var(--edge)}
.track i{position:absolute;top:0;bottom:0;display:block}
.val{width:66px;text-align:right;font:12.5px ui-monospace,monospace;flex:none}
.extra{width:92px;font:11.5px ui-monospace,monospace;color:var(--mut);flex:none}
.axis{display:flex;justify-content:space-between;font-size:11.5px;color:var(--mut);
margin:0 0 6px;padding:0 168px 0 88px}
.grid{width:100%;max-width:420px;display:block}
.fan{width:100%;max-width:420px;display:block}
.fanlab{font:11px ui-monospace,monospace;fill:var(--fg)}
.fanval{font:10.5px ui-monospace,monospace;fill:var(--mut)}
.warn{background:var(--warnbg);color:var(--warnfg);padding:10px 13px;
border-radius:8px;font-size:13.5px;margin:14px 0}
.legend{display:flex;gap:16px;align-items:center;font-size:12.5px;color:var(--mut);
margin:8px 0 0}.sw{display:inline-block;width:13px;height:13px;border-radius:3px;
vertical-align:-2px;margin-right:5px}
.note{color:var(--mut);font-size:12.5px;margin-top:26px;border-top:1px solid
var(--edge);padding-top:12px}
</style></head><body><div class="wrap">
<h1>""" + _html.escape(title) + """</h1>
<p class="sub">차폐(occlusion) 기반 입력 귀속 &nbsp;·&nbsp; 조향 부호 + 왼쪽 / − 오른쪽</p>
<div class="legend"><span><i class="sw" style="background:var(--left)"></i>왼쪽으로 밈</span>
<span><i class="sw" style="background:var(--right)"></i>오른쪽으로 밈</span></div>
""" + head + warn + """
<h2>그룹 차폐 — 어느 센서를 실제로 쓰는가</h2>
<div class="axis"><span>← 왼쪽</span><span>오른쪽 →</span></div>
""" + rows + """
<h2>라이다 섹터 — 실제 각도 배치</h2>
""" + _lidar_fan(lidar_vals, lmx) + """
<div class="axis"><span>← 왼쪽</span><span>오른쪽 →</span></div>
""" + lrows + """
<h2>lane 6×12 격자 — 화면 어느 영역이 조향을 만드는가</h2>
<p class="sub">위 = 먼 곳, 아래 = 가까운 곳. 칸에 마우스를 올리면 값이 보입니다.</p>
""" + _grid_svg(grid) + """
<p class="note">""" + _html.escape(note) + """</p>
</div></body></html>"""

    Path(path).write_text(doc, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="planner 모델 입력 귀속 분석")
    ap.add_argument("--model", type=Path, default=Path("planner_model_lidar.pth"))
    ap.add_argument("--csv", type=Path, default=Path("data/planner_data.csv"))
    ap.add_argument("--frame", type=int, default=None,
                    help="분석할 frame_id (미지정 시 전체 요약)")
    ap.add_argument("--limit", type=int, default=200,
                    help="요약 시 표본 프레임 수 (기본 200)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--by-course", action="store_true",
                    help="data/*_course*.csv 를 코스별로 따로 분석")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="--by-course 가 훑을 디렉토리")
    ap.add_argument("--explain", action="store_true",
                    help="사람이 읽는 문장으로 판단 근거 설명 (--frame 과 함께)")
    ap.add_argument("--html", type=Path, default=None,
                    help="HTML 리포트 저장 경로 (브라우저로 열어보기)")
    args = ap.parse_args()

    if not args.model.exists():
        print("모델 파일 없음: " + str(args.model))
        sys.exit(1)

    model = PlannerModel()
    try:
        model.load_state_dict(torch.load(str(args.model), map_location="cpu"))
    except RuntimeError as exc:
        print("체크포인트가 현재 모델 구조와 맞지 않습니다:")
        print(str(exc)[:400])
        sys.exit(1)
    model.eval()

    raw = pd.read_csv(args.csv)
    df = raw.dropna().reset_index(drop=True)
    if len(df) < len(raw):
        print("[warn] NaN 이 있는 " + str(len(raw) - len(df))
              + " 행이 제외되었습니다. IMU 컬럼이 있는 파일과 없는 파일을"
              " 섞으면 구 파일 행 전체가 여기서 사라집니다.")
    print("[load] " + args.model.name + " / " + args.csv.name
          + "   (" + str(len(df)) + " rows)")
    print("")

    if args.by_course:
        courses = by_course(model, args.data_dir, args.limit)
        if not courses:
            print("코스 CSV 를 찾지 못했습니다: " + str(args.data_dir))
            sys.exit(1)
        report_by_course(courses)
        if args.html:
            write_html_courses(
                args.html, courses,
                "모델 " + args.model.name + " / " + str(args.data_dir)
                + " / 코스당 최대 " + str(args.limit) + " 프레임 표본",
            )
            print("")
            print("[html] " + str(args.html))
        return

    if args.frame is not None:
        sub = df[df["frame_id"] == args.frame]
        if sub.empty:
            lo = int(df["frame_id"].min())
            hi = int(df["frame_id"].max())
            print("frame_id=" + str(args.frame) + " 없음. 범위 "
                  + str(lo) + " ~ " + str(hi))
            sys.exit(1)
        row = sub.iloc[0]
        res = analyse_frame(model, row)
        if args.explain:
            explain_frame(row, res)
            print("")
        report_frame(row, res)
        if args.html:
            write_html(
                args.html, "Planner Saliency · frame " + str(args.frame),
                res["pred"],
                (float(row["target_steering"]), float(row["target_throttle"])),
                res["group"], [x[0] for x in res["lidar"]], res["lane"],
                "모델 " + args.model.name + " / 데이터 " + args.csv.name,
            )
            print("")
            print("[html] " + str(args.html))
    else:
        agg = aggregate(model, df, args.limit)
        if agg is None:
            print("분석할 프레임이 없습니다.")
            sys.exit(1)
        report_summary(agg)
        if args.html:
            write_html(
                args.html, "Planner Saliency · 전체 평균",
                None, None,
                {b: tuple(agg["group_sgn"][b]) for b in BLOCKS},
                list(agg["lidar_sgn"]), agg["lane_sgn"],
                "모델 " + args.model.name + " / 데이터 " + args.csv.name
                + " / 표본 " + str(agg["n"]) + " 프레임 (부호 평균)",
            )
            print("")
            print("[html] " + str(args.html))


if __name__ == "__main__":
    main()
