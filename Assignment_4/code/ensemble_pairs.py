"""
예산 내 2-모델 앙상블 사후 분석 (val logits 사용, 추가 학습 없음)
  python ensemble_pairs.py <jobA> <jobB>
- jobA의 모델 i와 jobB의 모델 j를 짝지어 softmax 평균 → val 정확도 (모든 i,j 조합)
- 두 구조를 합친 전체 Params/MACs (thop/ptflops)도 측정 → 제한(30K/10M) 확인
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from common import ROOT, count_complexity, load_mnist
from models import TokenNet


class Ensemble(nn.Module):
    """두 모델의 클래스 확률 평균 (제출 시 이 모듈 하나가 '모델')"""

    def __init__(self, *ms):
        super().__init__()
        self.ms = nn.ModuleList(ms)

    def forward(self, x):
        return torch.stack([m(x).softmax(1) for m in self.ms]).mean(0)


ja, jb = sys.argv[1], sys.argv[2]
y = load_mnist("cpu", "dev")["y_val"].numpy()
la = torch.load(os.path.join(ROOT, "results", "logits", f"{ja}.pt"))["val"].float().softmax(-1).numpy()
lb = torch.load(os.path.join(ROOT, "results", "logits", f"{jb}.pt"))["val"].float().softmax(-1).numpy()
acc = lambda p: (p.argmax(-1) == y).mean() * 100
sa = [acc(la[:, i]) for i in range(la.shape[1])]
sb = [acc(lb[:, j]) for j in range(lb.shape[1])]
pairs = np.array([[acc((la[:, i] + lb[:, j]) / 2) for j in range(lb.shape[1])] for i in range(la.shape[1])])
print(f"{ja}: 단독 평균 {np.mean(sa):.3f}   {jb}: 단독 평균 {np.mean(sb):.3f}")
print(f"교차 앙상블 {pairs.size}쌍: 평균 {pairs.mean():.3f} ± {pairs.std():.3f}  (min {pairs.min():.2f}, max {pairs.max():.2f})")
spec = lambda j: json.load(open(os.path.join(ROOT, "results", "runs", f"{j}_s0.json")))["spec"]
cx = count_complexity(Ensemble(TokenNet(spec(ja)), TokenNet(spec(jb))))
print(f"앙상블 전체: params={cx['max_params']:,}  MACs={cx['max_macs']:,}  제한 이내={cx['ok']}")
