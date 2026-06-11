"""공용 유틸리티: 시드 고정, 디바이스 선택, JSON 입출력."""
import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """random/numpy/torch(cuda 포함) 시드를 고정해 재현성을 보장한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = "cuda") -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA를 사용할 수 없어 CPU로 대체합니다.")
        return torch.device("cpu")
    return torch.device(name)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
