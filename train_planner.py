#!/usr/bin/env python3
"""
Planner Training Script
========================
Trains PlannerModel on the structured feature dataset.

  Input  : data/augmented_data.csv  (or planner_data.csv as fallback)
  Output : planner_model.pth

Architecture recap
------------------
  objects  (40-dim) → MLP → 64-dim  ─┐
  lane      (5-dim) → MLP → 32-dim  ─┤ concat (120-dim)
  ego       (2-dim) → MLP → 16-dim  ─┤   → trunk → steering_head  → steering ∈ [-1, 1]
  scenario  (int)   → Emb →  8-dim  ─┘           → throttle_head  → throttle ∈ [ 0, 1]

Loss: MSELoss on both outputs jointly (equal weight).
      Training is fast — no GPU memory pressure like image models.

Usage
-----
  python train_planner.py [--csv data/augmented_data.csv]
                          [--epochs 100]
                          [--lr 3e-4]
                          [--batch-size 64]
                          [--output planner_model.pth]
"""

import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from planner_model import (
    PlannerModel,
    csv_columns,
    row_to_tensors,
    N_MAX_OBJECTS, OBJ_FEATURES, LANE_FEATURES, EGO_FEATURES,
    GRID_ROWS, GRID_COLS,
    N_SCENARIOS, MAX_THROTTLE,
    LIDAR_CLIP_M,          # 💡 [추가] 라이다 상한 클리핑 — 추론과 같은 상수를 공유
)

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
DEFAULT_CSV = DATA_DIR / "augmented_data.csv"
FALLBACK_CSV= DATA_DIR / "planner_data.csv"
SAVE_PATH   = SCRIPT_DIR / "planner_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PlannerDataset(Dataset):
    """
    Each sample: (objects, lane, ego, scenario, target)
      objects  : (40,)  float32
      lane     : ( 5,)  float32
      ego      : ( 2,)  float32
      scenario : ()     long
      target   : ( 2,)  float32  [steering, throttle_norm]
    """

    def __init__(self, csv_path: Path, ego_dropout: float = 0.0,
                 lidar_clip: float = 0.0):
        # 💡 [추가] 두 실험 옵션. 기본값이 꺼짐이라 미지정 시 기존 동작과 동일하다.
        #
        # ego_dropout: 학습 중 확률적으로 ego 를 0 으로 만든다.
        #   [근거] 폐루프 주행에서 ego 는 모델 자기 출력이 되돌아온 값이라
        #   오차가 누적된다. 반면 학습 때 ego 는 항상 정답이라
        #   corr(ego_throttle, target_throttle)=+0.78 인 지름길이 된다.
        #   차폐 측정에서 ego 가 스로틀 기여의 67%(0.1868)를 차지했고,
        #   S구간에서는 라이다(0.2308)의 1.9배(0.4368)였다.
        #   일정 비율로 ego 를 가리면 그 프레임은 센서로만 풀어야 하므로
        #   모델이 두 경로를 모두 갖추게 된다.
        #
        # lidar_clip: 라이다 거리를 상한으로 자른다(0 이면 비활성).
        #   [근거] 라이다는 정규화 없이 raw 미터로 들어가고, 전체의 27.4%가
        #   5.0 에 포화돼 있다. 그런데 5.0 은 "무반사"와 "5m 밖"을 겸해
        #   의미가 모호하고, 개활 테스트장에서만 나오는 값이다.
        #   실측: 코스별 평균 |좌우 비대칭| 1.704 → 2.0m 클립 시 0.010 으로
        #   테스트장 지문이 소거되며, S구간 구조(+0.03/+0.05)와
        #   근거리(<0.45m) 프레임은 100% 보존된다.
        self.ego_dropout = float(ego_dropout)
        self.lidar_clip = float(lidar_clip)

        df = pd.read_csv(csv_path)

        # Validate schema
        expected = set(csv_columns())
        missing  = expected - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        # Drop rows with NaN
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        if len(df) < before:
            print(f"[data] Dropped {before - len(df)} rows with NaN values")

        # Drop rows where both ego and target throttle are ~0 (startup frames before motion)
        before = len(df)
        df = df[~((df["ego_throttle"].abs() < 0.05) & (df["target_throttle"].abs() < 0.05))].reset_index(drop=True)
        if len(df) < before:
            print(f"[data] Dropped {before - len(df)} startup rows (ego_thr<0.1 & target_thr<0.1)")

        # ── Zero-Cost Hard Negative Injection ──────────────────────────────
        hard_mask = (df["target_steering"].abs() > 0.3) | (df["target_throttle"] < 0.1)
        df_hard = df[hard_mask]
        if len(df_hard) > 0:
            df = pd.concat([df, df_hard, df_hard], axis=0) \
                   .sample(frac=1.0, random_state=42) \
                   .reset_index(drop=True)
            print(f"[data] Hard-negative injection: amplified {len(df_hard)} corner/brake "
                  f"rows (x3) -> {len(df)} total rows")
        # ─────────────────────────────────────────────────────────────────

        self.df = df
        print(f"[data] Loaded {len(df)} rows from {csv_path.name}")

        # Warn if any scenario has effectively no lane features — indicates BiSeNet
        # wasn't detecting lanes during that collection session.  The model will
        # never generalise to lane=YES for that scenario.
        from planner_model import SCENARIO_NAMES, GRID_ROWS, GRID_COLS
        lane_cols = [f"lane_r{r}c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        for sc, grp in df.groupby("scenario"):
            lane_mean = grp[lane_cols].values.mean()
            if lane_mean < 0.01:
                sc_name = SCENARIO_NAMES.get(sc, str(sc))
                print(f"[WARN] Scenario {sc} ({sc_name}): lane features are all ~0 "
                      f"(mean={lane_mean:.4f}). Looks like BiSeNet had no detections "
                      f"during this collection session. Re-collect with lanes visible.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Object features: (N_MAX_OBJECTS * OBJ_FEATURES,)
        obj_vals = [float(row[f"obj{i}_{f}"])
                    for i in range(N_MAX_OBJECTS)
                    for f in ("valid", "class_norm", "conf", "dist_norm",
                              "lat_offset", "width_norm", "height_norm", "lane_overlap")]
        objects = torch.tensor(obj_vals, dtype=torch.float32)

        # Lane features: (LANE_FEATURES,) — 4×8 spatial grid
        lane_vals = [float(row[f"lane_r{r}c{c}"])
                     for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        lane = torch.tensor(lane_vals, dtype=torch.float32)

        # Ego features: (EGO_FEATURES,)
        ego = torch.tensor([float(row["ego_steering"]),
                            float(row["ego_throttle"])], dtype=torch.float32)
        # 💡 [추가] ego dropout — 매 에폭 다르게 가려야 하므로 학습 단계에서만 가능
        if self.ego_dropout > 0.0 and random.random() < self.ego_dropout:
            ego = torch.zeros_like(ego)
                            
        # 💡 [추가된 부분: 라이다 센서 데이터 추출 및 텐서화]
        # CSV 파일의 각 행에서 lidar_s0 ~ lidar_s4 컬럼 값을 읽어와 5차원 PyTorch 텐서로 변환합니다.
        lidar_vals = [float(row[f"lidar_s{i}"]) for i in range(5)]
        # 💡 [추가] 라이다 상한 클리핑
        if self.lidar_clip > 0.0:
            lidar_vals = [min(v, self.lidar_clip) for v in lidar_vals]
        lidar = torch.tensor(lidar_vals, dtype=torch.float32)

        # Scenario token
        scenario = torch.tensor(int(row["scenario"]), dtype=torch.long)

        # Targets: [steering, throttle_norm]
        target = torch.tensor([float(row["target_steering"]),
                                float(row["target_throttle"])], dtype=torch.float32)

        # 💡 [변경된 부분: 라이다 데이터를 포함하여 총 6개의 반환값(Tuple) 제공]
        return objects, lane, lidar, ego, scenario, target


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    csv_path:    Path         = DEFAULT_CSV,
    epochs:      int          = 100,
    lr:          float        = 3e-4,
    batch_size:  int          = 64,
    save_path:   Path         = SAVE_PATH,
    finetune_from: Path | None = None,
    ego_dropout: float = 0.0,
    lidar_clip: float = 0.0,
) -> None:

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 58)
    print("  Planner Training")
    print("=" * 58)
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        free, total = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
        print(f"  VRAM      : {free:.0f} MB free / {total:.0f} MB total")
    print(f"  CSV       : {csv_path}")
    print(f"  Epochs    : {epochs}")
    print(f"  LR        : {lr}")
    print(f"  Batch     : {batch_size}")
    print(f"  Save      : {save_path}")
    print(f"  Finetune  : {finetune_from if finetune_from else 'NO (train from scratch)'}")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not csv_path.exists():
        if FALLBACK_CSV.exists():
            print(f"[WARN] {csv_path.name} not found — using {FALLBACK_CSV.name}")
            csv_path = FALLBACK_CSV
        else:
            print(f"[ERROR] No dataset found. Run collect_data_planner.py first.")
            sys.exit(1)

    dataset = PlannerDataset(csv_path, ego_dropout=ego_dropout, lidar_clip=lidar_clip)

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty.")
        sys.exit(1)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    print("=" * 58)
    print("  Dataset Statistics")
    print("=" * 58)
    df = dataset.df
    print(f"  Total samples   : {len(df)}")
    print()
    print("  Scenario distribution:")
    from planner_model import SCENARIO_NAMES as scenario_map
    sc_counts = df["scenario"].value_counts().sort_index()
    for sc, cnt in sc_counts.items():
        print(f"    {scenario_map.get(sc, sc):20s}: {cnt:>6d}  ({100*cnt/len(df):.1f}%)")
    if len(sc_counts) > 1:
        imbalance = sc_counts.max() / sc_counts.min()
        if imbalance > 5:
            print(f"  [WARN] Scenario imbalance {imbalance:.0f}× — dominant scenario will overpower others.")
            print(f"         Collect more data for under-represented scenarios before training.")
    print()
    print("  Target steering:")
    print(f"    mean={df['target_steering'].mean():+.4f}  "
          f"std={df['target_steering'].std():.4f}  "
          f"min={df['target_steering'].min():+.4f}  "
          f"max={df['target_steering'].max():+.4f}")
    print("  Target throttle (normalised):")
    print(f"    mean={df['target_throttle'].mean():.4f}  "
          f"std={df['target_throttle'].std():.4f}  "
          f"min={df['target_throttle'].min():.4f}  "
          f"max={df['target_throttle'].max():.4f}")
    print()

    # ── Train / val split ─────────────────────────────────────────────────────
    n_total = len(dataset)
    n_train = int(0.85 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train / Val     : {n_train} / {n_val}")
    print()

    # Structured data is tiny — use larger batch & more workers than image models
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PlannerModel().to(device)
    if finetune_from is not None:
        if not finetune_from.exists():
            print(f"[ERROR] Finetune checkpoint not found: {finetune_from}")
            sys.exit(1)
        state = torch.load(str(finetune_from), map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"  Loaded weights  : {finetune_from}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params    : {n_params:,}")
    print()

    # ── Quick forward pass check ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        o_t = sample[0].unsqueeze(0).to(device)
        l_t = sample[1].unsqueeze(0).to(device)
        
        # 💡 [추가 및 변경된 부분: 모델 초기화 점검용 더미 테스트에 라이다 텐서 추가]
        # 데이터셋 반환값이 6개로 늘어남에 따라 인덱스를 순서대로 밀어주고, 라이다(ld_t)를 세 번째 입력으로 전달합니다.
        ld_t = sample[2].unsqueeze(0).to(device) 
        e_t = sample[3].unsqueeze(0).to(device)
        s_t = sample[4].unsqueeze(0).to(device)
        out = model(o_t, l_t, ld_t, e_t, s_t)
        
    print(f"  Forward check   : input shapes obj={tuple(o_t.shape)} "
          f"lane={tuple(l_t.shape)} ego={tuple(e_t.shape)} sc={tuple(s_t.shape)}")
    print(f"  Output sample   : steer={out[0,0].item():+.4f}  "
          f"throttle={out[0,1].item():.4f}")
    print("  PASSED")
    print()

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    print("=" * 58)
    print(f"  Training  ({epochs} epochs, Adam+CosineAnnealing, MSELoss)")
    print("=" * 58)
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Steer MAE':>10}  {'Thtl MAE':>9}  LR")
    print("-" * 62)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        
        # 💡 [추가 및 변경된 부분: 학습 데이터 로더 언패킹 시 라이다 포함]
        for objects, lane, lidar, ego, scenario, target in train_loader:
            objects  = objects.to(device)
            lane     = lane.to(device)
            
            # 💡 [추가된 부분: 라이다 데이터를 GPU/CPU 메모리로 이동]
            lidar    = lidar.to(device)    
            ego      = ego.to(device)      
            scenario = scenario.to(device)
            target   = target.to(device)

            optimizer.zero_grad()
            
            # 💡 [변경된 부분: 모델 추론에 라이다(lidar) 인자 추가]
            # 새롭게 변경된 PlannerModel의 forward 시그니처에 맞추어 정확한 순서로 5개의 인자를 넘겨줍니다.
            pred = model(objects, lane, lidar, ego, scenario)   
            loss = criterion(pred, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(target)

        train_loss = running_loss / n_train
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        running_loss = 0.0
        steer_abs    = 0.0
        thtl_abs     = 0.0
        with torch.no_grad():
            
            # 💡 [추가 및 변경된 부분: 검증 데이터 로더 언패킹 시 라이다 포함 및 모델 추론 반영]
            # 훈련 루프와 완벽히 동일한 방식으로 라이다 데이터를 텐서화하고 검증 추론에 활용합니다.
            for objects, lane, lidar, ego, scenario, target in val_loader:
                objects  = objects.to(device)
                lane     = lane.to(device)
                ego      = ego.to(device)
                lidar    = lidar.to(device) 
                scenario = scenario.to(device)
                target   = target.to(device)
                pred     = model(objects, lane, lidar, ego, scenario)
                
                loss     = criterion(pred, target)
                running_loss += loss.item() * len(target)
                steer_abs    += (pred[:, 0] - target[:, 0]).abs().sum().item()
                thtl_abs     += (pred[:, 1] - target[:, 1]).abs().sum().item()

        val_loss  = running_loss / n_val
        steer_mae = steer_abs / n_val
        thtl_mae  = thtl_abs  / n_val
        cur_lr    = scheduler.get_last_lr()[0]

        # ── Save best ─────────────────────────────────────────────────────────
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            marker = "  ★"

        # Print every epoch up to 20, then every 10
        if epoch <= 20 or epoch % 10 == 0 or marker:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{steer_mae:>10.4f}  {thtl_mae:>9.4f}  {cur_lr:.2e}{marker}")

    print()
    print(f"  Best val loss   : {best_val_loss:.6f}")
    print(f"  Model saved  →  {save_path}")
    print()
    print("  Next steps:")
    print("   1. Run evaluate.py to check per-scenario error")
    print("   2. Collect more data for under-performing scenarios")
    print("   3. Run planner_inference.py --motor to test on vehicle")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the structured planner model")
    parser.add_argument('--csv',        type=Path,  default=DEFAULT_CSV,
                        help=f'Dataset CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--epochs',     type=int,   default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--lr',         type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int,   default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--output',       type=Path,  default=SAVE_PATH,
                        help=f'Model output path (default: {SAVE_PATH})')
    parser.add_argument('--ego-dropout', type=float, default=0.0,
                        help='학습 중 ego 를 0 으로 만들 확률 (0=비활성). '
                             '폐루프에서 ego 자기출력 되먹임 의존을 끊는다')
    parser.add_argument('--lidar-clip', type=float, default=LIDAR_CLIP_M,
                        help='라이다 거리 상한 [m] (0=비활성). 2.0 권장 — '
                             '테스트장 지문(좌우 비대칭)을 소거한다')
    parser.add_argument('--finetune',     type=Path,  default=None,
                        help='Start from an existing checkpoint instead of random init '
                             '(e.g. --finetune planner_model.pth). Use a lower LR, e.g. --lr 5e-5')
    args = parser.parse_args()

    train(
        csv_path      = args.csv,
        epochs        = args.epochs,
        lr            = args.lr,
        batch_size    = args.batch_size,
        save_path     = args.output,
        finetune_from = args.finetune,
        ego_dropout   = args.ego_dropout,
        lidar_clip    = args.lidar_clip,
    )
