#!/usr/bin/env python3
"""
augment.py — 파이프라인 오케스트레이션 + CLI
=============================================

    CSV
     ↓  DatasetLoader        (로드 + 파일명→scenario prior)
     ↓  compute_sensor_evidence  (센서 증거만)
     ↓  ScenarioClassifier   (prior + 증거 → 시나리오 라벨)
     ↓  LabelProcessor       (보간 + 조건부 스무딩)
     ↓  Augmentor            (시나리오별 차등 증강)
    Training CSV

사용법:
    python augment.py --inputs data/normal_course1.csv data/stop_course1.csv
    python augment.py --input-dir data --output data/augmented_data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import INTERNAL_COLS, KNOWN_SCENARIOS, csv_columns
from dataset_loader import DatasetLoader, discover_files
from sensor_evidence import compute_sensor_evidence
from scenario import ScenarioClassifier
from label_processor import LabelProcessor
from augmentation import Augmentor


def run_pipeline(input_files: list, output_csv: Path,
                 seed: int = 42, smooth_window: int = 5) -> None:
    print("=" * 68)
    print("  Planner Dataset — Load / Classify / Process / Augment")
    print("=" * 68)
    print(f"  Output : {output_csv}")
    print(f"  Seed   : {seed}   Smooth window: {smooth_window}")
    print()

    # ── 1. 로드 + 파일명 prior ───────────────────────────────────────
    print("[1/5] Loading files")
    df = DatasetLoader().load(input_files)
    if df.empty:
        print("[ERROR] 유효한 데이터가 없습니다.")
        return
    print(f"  총 {len(df)} rows, {df['_src_idx'].nunique()} files")

    missing = set(csv_columns()) - set(df.columns)
    if missing:
        print(f"[ERROR] 필수 컬럼 누락: {sorted(missing)[:8]}")
        return

    grp = df["_src_idx"]

    # ── 2. 센서 증거 ─────────────────────────────────────────────────
    print("\n[2/5] Computing sensor evidence")
    ev = compute_sensor_evidence(df, grp, smooth_window)
    print(f"  front_dist median={ev['front_dist'].median():.3f}m  "
          f"side_dist median={ev['side_dist'].median():.3f}m  "
          f"cam_threat={int(ev['cam_threat'].sum())} frames")

    # ── 3. 시나리오 분류 ─────────────────────────────────────────────
    print("\n[3/5] Classifying scenarios")
    decisions = ScenarioClassifier(smooth_window).classify(df, ev)

    counts = decisions["scenario_label"].value_counts()
    for name in KNOWN_SCENARIOS:
        n = int(counts.get(name, 0))
        if n:
            print(f"  {name:11s}: {n:6d} frames")
    print(f"  {'protected':11s}: {int(decisions['is_protected'].sum()):6d} frames "
          f"(스무딩 제외)")

    # ── 4. 라벨 처리 ─────────────────────────────────────────────────
    print("\n[4/5] Processing labels")
    steer_before = df["target_steering"].std()
    thtl_before = df["target_throttle"].std()

    processor = LabelProcessor(smooth_window)
    df = processor.process(df, decisions)

    print(f"  노이즈 보간: {processor.last_stats['interpolated']} frames")
    print(f"  Steering std {steer_before:.4f} → {df['target_steering'].std():.4f}")
    print(f"  Throttle std {thtl_before:.4f} → {df['target_throttle'].std():.4f}")

    # 필터링된 원본(증강 전) 저장
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_path = output_csv.parent / "planner_data.csv"
    clean = df.drop(columns=[c for c in INTERNAL_COLS if c in df.columns])
    clean = clean.copy()
    clean["frame_id"] = np.arange(1, len(clean) + 1)
    clean.to_csv(merged_path, index=False)
    print(f"  [SAVE] 필터링된 원본 → {merged_path}")

    # ── 5. 증강 ──────────────────────────────────────────────────────
    print("\n[5/5] Augmenting")
    augmentor = Augmentor(seed)
    aug_df = augmentor.augment(df, decisions["scenario_label"])

    stats = augmentor.last_stats
    if stats["excluded_noise_rows"]:
        print(f"  noise 파일 제외: {stats['excluded_noise_rows']} rows "
              f"(학습 데이터에 포함하지 않음)")
    if stats["mirror_skipped"]:
        print(f"  mirror 증강 생략: {stats['mirror_skipped']} rows "
              f"(avoidance 방향성 보존)")

    if aug_df.empty:
        print("[ERROR] 증강 결과가 비어 있습니다.")
        return

    aug_df.to_csv(output_csv, index=False)

    print()
    print("=" * 68)
    print(f"[RESULT] {stats['input_rows']} → {stats['output_rows']} rows "
          f"(×{stats['output_rows'] / max(stats['input_rows'], 1):.1f})")
    print(f"[RESULT] Saved → {output_csv}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Planner dataset: merge + scenario classification + augmentation"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", nargs="+", help="입력 CSV 파일들")
    src.add_argument("--input-dir", type=Path, help="CSV를 자동 탐색할 디렉토리")

    parser.add_argument("--output", type=Path,
                        default=Path("data/augmented_data.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth", type=int, default=5,
                        help="라벨 스무딩 rolling window (1이면 비활성)")
    args = parser.parse_args()

    if args.input_dir:
        files = discover_files(args.input_dir)
        if not files:
            print(f"[ERROR] {args.input_dir} 에 CSV가 없습니다.")
            return
    else:
        files = [Path(p) for p in args.inputs]

    run_pipeline(files, args.output, args.seed, args.smooth)


if __name__ == "__main__":
    main()