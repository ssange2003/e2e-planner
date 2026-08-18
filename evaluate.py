#!/usr/bin/env python3
"""
Planner Offline Evaluation
===========================
Runs the trained PlannerModel against a held-out dataset (or the full
dataset) and reports per-scenario error metrics + plots.

Metrics reported
----------------
  • Steering MAE / RMSE  per scenario and overall
  • Throttle  MAE / RMSE  per scenario and overall
  • Steering error histogram
  • Throttle error histogram
  • Prediction vs. ground-truth scatter plot
  • Per-scenario comparison bar chart

Plots are saved to  data/eval/  as PNG files.
A summary is printed to stdout and saved as  data/eval/summary.txt.

Usage
-----
  python evaluate.py [--csv data/planner_data.csv]
                     [--model planner_model.pth]
                     [--out-dir data/eval]
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch

# 💡 [추가 및 변경된 부분: SCENARIO_NAMES 임포트]
# 하드코딩된 시나리오 목록을 지우고, planner_model.py에서 최신 시나리오 
# 이름 매핑 사전(SCENARIO_NAMES)을 직접 가져오도록 추가되었습니다. 
# 이제 6번(CUSTOM_SPLINE) 시나리오도 완벽하게 평가 지표에 이름이 찍혀 나옵니다.
from planner_model import (
    PlannerModel,
    csv_columns,
    N_MAX_OBJECTS,
    MAX_THROTTLE,
    N_SCENARIOS,
    GRID_ROWS, GRID_COLS,
    SCENARIO_NAMES, 
)

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
DEFAULT_CSV = DATA_DIR / "planner_data.csv"
MODEL_PATH  = SCRIPT_DIR / "planner_model.pth"
OUT_DIR     = DATA_DIR / "eval"

# 💡 [삭제된 부분: 과거의 하드코딩된 _SCENARIO_NAMES 딕셔너리 삭제]
# 예전에 쓰던 0~3번만 들어있던 딕셔너리를 삭제했습니다.
# 이제는 planner_model.py의 SCENARIO_NAMES를 전역적으로 공유해서 사용합니다.


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (reuses PlannerDataset schema without torch dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tensors(csv_path: Path, device):
    """Load the full CSV and return model-ready tensors + targets."""
    df = pd.read_csv(csv_path)

    # Validate
    expected = set(csv_columns())
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.dropna().reset_index(drop=True)

    obj_cols  = [f"obj{i}_{f}"
                 for i in range(N_MAX_OBJECTS)
                 for f in ("valid", "class_norm", "conf", "dist_norm",
                           "lat_offset", "width_norm", "height_norm", "lane_overlap")]
    lane_cols = [f"lane_r{r}c{c}"
                 for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    
    # 💡 [추가 및 변경된 부분: 라이다 데이터 컬럼 정의]
    # 원본 코드에서는 카메라 기반 피처(objects, lane)와 차량 상태(ego)만 불러왔습니다.
    # 하지만 라이다 센서 연동이 추가되면서, CSV 파일에 저장된 5개 구역의 
    # 라이다 거리 데이터(lidar_s0 ~ lidar_s4)를 읽어오기 위한 컬럼 리스트가 신규로 추가되었습니다.
    lidar_cols = ["lidar_s0", "lidar_s1", "lidar_s2", "lidar_s3", "lidar_s4"]
    ego_cols  = ["ego_steering", "ego_throttle"]

    objects  = torch.tensor(df[obj_cols].values,  dtype=torch.float32, device=device)
    lane     = torch.tensor(df[lane_cols].values, dtype=torch.float32, device=device)
    
    # 💡 [추가 및 변경된 부분: 라이다 데이터를 파이토치 텐서로 변환]
    # 위에서 정의한 lidar_cols를 바탕으로 Pandas DataFrame에서 라이다 값을 추출한 뒤,
    # AI 모델 연산이 가능하도록 GPU/CPU 메모리에 올라가는 PyTorch 텐서(Tensor) 형태로 변환하는 로직이 추가되었습니다.
    lidar    = torch.tensor(df[lidar_cols].values, dtype=torch.float32, device=device)  
    
    ego      = torch.tensor(df[ego_cols].values,  dtype=torch.float32, device=device)
    scenario = torch.tensor(df["scenario"].values, dtype=torch.long,   device=device)
    target   = torch.tensor(
        df[["target_steering", "target_throttle"]].values,
        dtype=torch.float32, device=device)

    # 💡 [추가 및 변경된 부분: 반환(Return) 데이터 개수 증가]
    # 원본 코드는 총 6개의 값(objects, lane, ego, scenario, target, df["scenario"])만 반환했지만,
    # 평가 과정에서 라이다 입력값이 필요해지면서 'lidar' 텐서가 중간에 끼어들어가 총 7개의 값을 반환하도록 구조가 수정되었습니다.
    return objects, lane, lidar, ego, scenario, target, df["scenario"].values


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    csv_path:  Path = DEFAULT_CSV,
    model_path:Path = MODEL_PATH,
    out_dir:   Path = OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print("  Planner Offline Evaluation")
    print("=" * 58)
    print(f"  Dataset : {csv_path}")
    print(f"  Model   : {model_path}")
    print(f"  Output  : {out_dir}")
    print()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}  — run train_planner.py first")
        sys.exit(1)
    model = PlannerModel().to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found: {csv_path}")
        sys.exit(1)

    # 💡 [추가 및 변경된 부분: 7개의 반환값 언패킹(Unpacking)]
    # 위에서 수정한 _load_tensors 함수가 이제 라이다(lidar)를 포함해 총 7개의 값을 뱉어내므로,
    # 데이터를 넘겨받는 이 부분에서도 원본과 달리 'lidar' 변수를 추가하여 7개를 빠짐없이 받아내도록 수정되었습니다.
    objects, lane, lidar, ego, scenario_t, target, scenario_np = _load_tensors(csv_path, device)
    n = len(target)
    print(f"  Samples : {n}")
    print()

    # ── Inference (batched to avoid OOM) ──────────────────────────────────────
    batch_size = 256
    preds_list = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            
            # 💡 [추가 및 변경된 부분: AI 모델 추론 시 라이다 입력 추가]
            # 원본 코드에서는 model(objects, lane, ego, scenario_t) 이렇게 4개의 피처만 넘겨주었습니다.
            # 하지만 라이다 데이터를 함께 학습하도록 구조가 확장된 새로운 PlannerModel을 평가하기 위해,
            # 세 번째 입력값으로 라이다 텐서(lidar[start:end])를 추가로 밀어넣어 총 5개의 인자를 전달하도록 변경되었습니다.
            out = model(objects[start:end], lane[start:end], 
                        lidar[start:end], ego[start:end], scenario_t[start:end])
            preds_list.append(out.cpu())
    preds = torch.cat(preds_list, dim=0)   # (N, 2)

    target_cpu = target.cpu().numpy()
    preds_np   = preds.numpy()

    # Denormalise throttle for human-readable reporting
    pred_steer  = preds_np[:, 0]
    pred_thtl   = preds_np[:, 1] * MAX_THROTTLE
    true_steer  = target_cpu[:, 0]
    true_thtl   = target_cpu[:, 1] * MAX_THROTTLE

    # ── Overall metrics ───────────────────────────────────────────────────────
    def _mae(a, b):  return float(np.mean(np.abs(a - b)))
    def _rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

    overall_steer_mae  = _mae(pred_steer, true_steer)
    overall_steer_rmse = _rmse(pred_steer, true_steer)
    overall_thtl_mae   = _mae(pred_thtl, true_thtl)
    overall_thtl_rmse  = _rmse(pred_thtl, true_thtl)

    lines = []
    lines.append("=" * 58)
    lines.append("  OVERALL RESULTS")
    lines.append("=" * 58)
    lines.append(f"  Samples          : {n}")
    lines.append(f"  Steering MAE     : {overall_steer_mae:.4f}")
    lines.append(f"  Steering RMSE    : {overall_steer_rmse:.4f}")
    lines.append(f"  Throttle MAE     : {overall_thtl_mae:.4f} (m/s equiv)")
    lines.append(f"  Throttle RMSE    : {overall_thtl_rmse:.4f}")
    lines.append("")

    # ── Per-scenario metrics ──────────────────────────────────────────────────
    lines.append("  PER-SCENARIO RESULTS")
    lines.append("-" * 58)
    lines.append(f"  {'Scenario':<20} {'N':>6}  {'Steer MAE':>10}  {'Steer RMSE':>11}  {'Thtl MAE':>9}")
    lines.append("-" * 58)

    scenario_metrics = {}
    for sc in sorted(np.unique(scenario_np)):
        mask   = scenario_np == sc
        n_sc   = mask.sum()
        sm     = _mae(pred_steer[mask],  true_steer[mask])
        sr     = _rmse(pred_steer[mask], true_steer[mask])
        tm     = _mae(pred_thtl[mask],   true_thtl[mask])
        
        # 💡 [변경된 부분: 임포트된 SCENARIO_NAMES 동적 매핑]
        # 이제 하드코딩된 _SCENARIO_NAMES가 아니라 planner_model.py에서 가져온 SCENARIO_NAMES를 사용합니다.
        name   = SCENARIO_NAMES.get(int(sc), str(sc))
        
        scenario_metrics[name] = {'n': n_sc, 'steer_mae': sm, 'steer_rmse': sr, 'thtl_mae': tm}
        lines.append(f"  {name:<20} {n_sc:>6}  {sm:>10.4f}  {sr:>11.4f}  {tm:>9.4f}")

    lines.append("=" * 58)

    summary = "\n".join(lines)
    print(summary)

    # Save summary text
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    print(f"\n[EVAL] Summary → {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless — no display required
        import matplotlib.pyplot as plt

        # 1. Steering: prediction vs ground truth scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(true_steer, pred_steer, alpha=0.25, s=8, label="samples")
        lim = max(abs(true_steer).max(), abs(pred_steer).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label="perfect")
        ax.set_xlabel("Ground Truth Steering")
        ax.set_ylabel("Predicted Steering")
        ax.set_title(f"Steering Prediction (MAE={overall_steer_mae:.4f})")
        ax.legend()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "steering_scatter.png", dpi=120)
        plt.close(fig)

        # 2. Throttle: prediction vs ground truth scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(true_thtl, pred_thtl, alpha=0.25, s=8, color="orange")
        lim2 = max(true_thtl.max(), pred_thtl.max()) * 1.1
        ax.plot([0, lim2], [0, lim2], 'r--', linewidth=1)
        ax.set_xlabel("Ground Truth Throttle")
        ax.set_ylabel("Predicted Throttle")
        ax.set_title(f"Throttle Prediction (MAE={overall_thtl_mae:.4f})")
        ax.set_xlim(0, lim2)
        ax.set_ylim(0, lim2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "throttle_scatter.png", dpi=120)
        plt.close(fig)

        # 3. Steering error histogram
        steer_err = pred_steer - true_steer
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(steer_err, bins=50, edgecolor='black', linewidth=0.4, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
        ax.set_xlabel("Steering Error (pred − truth)")
        ax.set_ylabel("Count")
        ax.set_title(f"Steering Error Distribution  (MAE={overall_steer_mae:.4f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "steering_error_hist.png", dpi=120)
        plt.close(fig)

        # 4. Per-scenario steering MAE bar chart
        if scenario_metrics:
            names = list(scenario_metrics.keys())
            maes  = [scenario_metrics[n]['steer_mae'] for n in names]
            colors= ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728'][:len(names)]
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(names, maes, color=colors, edgecolor='black', linewidth=0.5)
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
            ax.set_ylabel("Steering MAE")
            ax.set_title("Per-Scenario Steering MAE")
            ax.set_ylim(0, max(maes) * 1.25)
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "per_scenario_mae.png", dpi=120)
            plt.close(fig)

        # 5. Time-series: first 200 samples (shows temporal tracking quality)
        n_plot = min(200, n)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        t = np.arange(n_plot)
        ax1.plot(t, true_steer[:n_plot],  label="truth",     linewidth=1.2)
        ax1.plot(t, pred_steer[:n_plot],  label="predicted", linewidth=1.2, linestyle='--')
        ax1.set_ylabel("Steering")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.plot(t, true_thtl[:n_plot],  label="truth",     linewidth=1.2, color='green')
        ax2.plot(t, pred_thtl[:n_plot],  label="predicted", linewidth=1.2, linestyle='--', color='olive')
        ax2.set_ylabel("Throttle")
        ax2.set_xlabel("Frame index")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"Prediction vs Ground Truth (first {n_plot} frames)")
        fig.tight_layout()
        fig.savefig(out_dir / "timeseries.png", dpi=120)
        plt.close(fig)

        print(f"[EVAL] Plots saved to {out_dir}/")
        print(f"       steering_scatter.png")
        print(f"       throttle_scatter.png")
        print(f"       steering_error_hist.png")
        print(f"       per_scenario_mae.png")
        print(f"       timeseries.png")

    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots")
        print("       pip install matplotlib  to enable plots")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation of the planner model")
    parser.add_argument('--csv',     type=Path, default=DEFAULT_CSV,
                        help=f'Dataset CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--model',   type=Path, default=MODEL_PATH,
                        help=f'Model .pth file (default: {MODEL_PATH})')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR,
                        help=f'Output directory for plots/summary (default: {OUT_DIR})')
    args = parser.parse_args()

    evaluate(csv_path=args.csv, model_path=args.model, out_dir=args.out_dir)