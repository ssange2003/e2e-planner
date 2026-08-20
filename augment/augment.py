# augment.py
import argparse
from pathlib import Path
from scenario import load_and_parse_files, apply_hierarchical_intent_filter
from augmentation import apply_augmentations
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path("data/augmented_data.csv"))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smooth', type=int, default=5)
    args = parser.parse_args()

    # 1. 로드 및 파일명 파싱
    df = load_and_parse_files(args.inputs)
    if df.empty: return

    # 2. 계층형 의도 융합 (센서 증거 -> 라벨 판정 -> 엣지 보호)
    df = apply_hierarchical_intent_filter(df, args.smooth)
    
    # 중간 저장
    df.drop(columns=["_src_idx"]).to_csv(args.output.parent / "planner_data.csv", index=False)

    # 3. 데이터 증강
    aug_df = apply_augmentations(df, args.seed)
    aug_df["frame_id"] = np.arange(1, len(aug_df) + 1)
    aug_df.to_csv(args.output, index=False)
    
    print(f"[RESULT] {len(df)} -> {len(aug_df)} rows saved to {args.output}")

if __name__ == "__main__":
    main()