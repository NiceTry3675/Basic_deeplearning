"""dev teacher logits(학습 60k 전체)에서 검증 인덱스만 뽑아 teacher/앙상블의 val 정확도 확인"""
import itertools
import sys

import numpy as np
import torch

from common import SPLIT_SEED, VAL_SIZE, DATA_PATH, ROOT

y = np.load(DATA_PATH)["y_train"]
idx_val = np.sort(np.random.RandomState(SPLIT_SEED).permutation(len(y))[:VAL_SIZE])
yv = torch.from_numpy(y[idx_val])
names = sys.argv[1:]
P = {n: torch.load(f"{ROOT}/results/teachers/{n}.pt")["logits"][idx_val].float().softmax(1) for n in names}
for n, p in P.items():
    print(f"{n:28s} val {(p.argmax(1) == yv).float().mean() * 100:.3f}")
for r in range(2, len(names) + 1):
    for c in itertools.combinations(names, r):
        p = sum(P[n] for n in c)
        print(f"{' + '.join(c):60s} val {(p.argmax(1) == yv).float().mean() * 100:.3f}")
