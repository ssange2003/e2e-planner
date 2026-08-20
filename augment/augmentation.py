# augmentation.py
import numpy as np
import pandas as pd
from config import *

# (기존 _clone, aug_identity ~ aug_mirror_and_noise 함수들 그대로 위치)
def _clone(row: np.ndarray) -> np.ndarray: return row.copy()
def aug_identity(row, rng): return _clone(row)
# ... [중략: 기존 aug_* 함수들 원본 그대로 100% 삽입] ...

AUGMENTATIONS = [
    ("identity",         aug_identity,         1),
    ("mirror",           aug_mirror,           1), # 등등 기존 리스트 유지
]

def apply_augmentations(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data_np = df[_COLS].to_numpy(dtype=float)
    
    aug_rows = []
    # 향후 시나리오별(마스크 기반) 증강 제외 로직을 여기에 쉽게 씌울 수 있음
    for orig_row in data_np:
        for name, fn, weight in AUGMENTATIONS:
            for _ in range(weight):
                aug_rows.append(fn(orig_row, rng))
                
    aug_df = pd.DataFrame(np.stack(aug_rows, axis=0), columns=_COLS)
    
    # 클리핑 후처리
    for i in range(N_MAX_OBJECTS):
        aug_df[f"obj{i}_valid"] = aug_df[f"obj{i}_valid"].clip(0, 1).round()
        # ... [중략: 기존 클리핑 루프 그대로 삽입]
        
    aug_df["target_steering"] = aug_df["target_steering"].clip(-1, 1)
    aug_df["target_throttle"] = aug_df["target_throttle"].clip(-1, 1)
    
    return aug_df