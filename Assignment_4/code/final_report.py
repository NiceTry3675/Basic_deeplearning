"""
최종(full split) 결과 정리 — 트랙별
  python final_report.py <정규식: 그룹 이름> [...]
  각 그룹: 모든 seed의 test acc(EMA 최종 가중치) → 평균±표준편차(트랙 A), 최고 seed와 체크포인트(트랙 B)
  트랙 A 제출 모델은 사전에 고정: 각 그룹에서 이름순 첫 번째 모델 (j0_s0)
"""
import glob
import json
import os
import re
import sys

import numpy as np

from common import ROOT

pats = sys.argv[1:] or ["^fin"]
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*.json")))]
runs = [r for r in runs if r.get("final_test_ema") is not None]
for pat in pats:
    rs = sorted([r for r in runs if re.search(pat, r["name"])], key=lambda r: r["name"])
    if not rs:
        print(f"[{pat}] 결과 없음"); continue
    te = np.array([r["final_test_ema"] for r in rs])
    raw = np.array([r["final_test"] for r in rs])
    best = rs[int(te.argmax())]
    first = rs[0]
    cx = first["complexity"]
    print(f"=== {pat}  (n={len(rs)})  spec: {first['spec']}")
    print(f"  params={cx['max_params']:,}  MACs={cx['max_macs']:,}  ok={cx['ok']}")
    print(f"  [트랙 A] test(EMA) 평균 {te.mean():.3f} ± {te.std(ddof=1) if len(te) > 1 else 0:.3f}  "
          f"(median {np.median(te):.2f}, min {te.min():.2f}, max {te.max():.2f}) | raw 평균 {raw.mean():.3f}")
    print(f"           사전 고정 제출 모델: {first['name']} → test {first['final_test_ema']:.2f}%")
    print(f"  [트랙 B] 최고 seed: {best['name']} → test {best['final_test_ema']:.2f}%  "
          f"ckpt: results/ckpt/{best['name']}.pt")
    vals, cnt = np.unique(np.round(te, 2), return_counts=True)
    print("  분포: " + "  ".join(f"{v:.2f}×{c}" for v, c in zip(vals, cnt)))
