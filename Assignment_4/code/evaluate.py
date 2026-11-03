"""
체크포인트 평가: 복잡도(thop/ptflops) + 테스트 정확도
  python evaluate.py ../results/ckpt/<name>.pt [--raw]
"""
import argparse

import torch

from common import accuracy, count_complexity, load_mnist
from models import TokenNet

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="+")
p.add_argument("--raw", action="store_true", help="EMA 대신 raw 가중치 평가")
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = load_mnist(dev, "full")
for path in args.ckpts:
    ck = torch.load(path, map_location=dev)
    m = TokenNet(ck["spec"], act=ck.get("act", "relu")).to(dev)
    m.load_state_dict(ck["raw_state_dict" if args.raw else "state_dict"])
    cx = count_complexity(m)
    acc = accuracy(m, D["x_te"], D["y_te"])
    n_err = int(round((100 - acc) / 100 * len(D["y_te"])))
    print(f"{path}\n  spec: {ck['spec']}\n  params={cx['max_params']:,}  MACs thop={cx['thop_macs']:,} "
          f"ptflops={cx['ptflops_macs']:,}  within_limit={cx['ok']}\n  TEST ACC = {acc:.2f}%  ({n_err} errors / 10000)")
