"""
저장된 logits(results/logits/<job>.pt, (N,K,10))로 사후 분석
  python analyze_logits.py <job> [<job> ...] [--split val|test]
- 모델별 정확도, K개 전체 앙상블 정확도, 오분류 샘플 겹침
"""
import argparse
import os

import numpy as np
import torch

from common import ROOT, load_mnist

p = argparse.ArgumentParser()
p.add_argument("jobs", nargs="+")
p.add_argument("--split", default="val")
args = p.parse_args()

D = load_mnist("cpu", "dev" if args.split == "val" else "full")
y = (D["y_val"] if args.split == "val" else D["y_te"]).numpy()
allp = []
for job in args.jobs:
    lg = torch.load(os.path.join(ROOT, "results", "logits", f"{job}.pt"))[args.split].float()
    pr = lg.softmax(-1).numpy()  # (N,K,10)
    accs = (pr.argmax(-1) == y[:, None]).mean(0) * 100
    ens = (pr.mean(1).argmax(-1) == y).mean() * 100
    wrong = (pr.argmax(-1) != y[:, None])
    print(f"{job:40s} K={pr.shape[1]} acc mean {accs.mean():.3f} [{accs.min():.2f}~{accs.max():.2f}]  "
          f"ensemble-all {ens:.3f}  | 모든 모델이 틀린 샘플 {wrong.all(1).sum()}, 하나라도 틀린 샘플 {wrong.any(1).sum()}")
    allp.append(pr)
if len(allp) > 1:
    P = np.concatenate(allp, 1)
    print(f"{'ALL jobs ensemble':40s} K={P.shape[1]} ensemble-all {(P.mean(1).argmax(-1) == y).mean() * 100:.3f}")
