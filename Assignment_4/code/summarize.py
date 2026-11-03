"""results/runs/*.json 집계: 이름에서 _s{seed}를 떼어 그룹별 평균/표준편차"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

from common import ROOT

pat = sys.argv[1] if len(sys.argv) > 1 else ""
key_sort = sys.argv[2] if len(sys.argv) > 2 else "val"
groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*.json"))):
    r = json.load(open(f))
    if "hist" not in r or not re.search(pat, r["name"]):
        continue
    g = re.sub(r"_s\d+$", "", r["name"])
    groups[g].append(r)

rows = []
for g, rs in groups.items():
    def arr(k):
        v = [r.get(k) for r in rs if r.get(k) is not None]
        return np.array(v) if v else None
    va, vae, te, tee = arr("final_val"), arr("final_val_ema"), arr("final_test"), arr("final_test_ema")
    # 마지막 5 epoch EMA val 평균 (노이즈 완화용)
    tail = [np.mean(r["hist"]["val_acc_ema"][-5:]) for r in rs if r["hist"].get("val_acc_ema")]
    rows.append(dict(g=g, n=len(rs), params=rs[0]["complexity"]["max_params"],
                     macs=rs[0]["complexity"]["max_macs"] / 1e6,
                     va=va.mean() if va is not None else np.nan,
                     vae=vae.mean() if vae is not None else np.nan,
                     vae_sd=vae.std() if vae is not None and len(vae) > 1 else np.nan,
                     tail=np.mean(tail) if tail else np.nan,
                     te=te.mean() if te is not None else np.nan,
                     tee=tee.mean() if tee is not None else np.nan,
                     tee_sd=tee.std() if tee is not None and len(tee) > 1 else np.nan,
                     tee_max=tee.max() if tee is not None else np.nan,
                     t=np.mean([r["time"] for r in rs])))

k = {"val": "vae", "tail": "tail", "test": "tee"}[key_sort]
rows.sort(key=lambda r: -np.nan_to_num(r[k], nan=-1))
print(f"{'group':42s} {'n':>2s} {'params':>6s} {'MACs':>5s} | {'val':>6s} {'valEMA':>6s} {'±':>5s} {'tail5':>6s} | "
      f"{'test':>6s} {'testEMA':>7s} {'±':>5s} {'max':>6s} | {'sec':>4s}")
for r in rows:
    print(f"{r['g']:42s} {r['n']:2d} {r['params']:6d} {r['macs']:5.2f} | {r['va']:6.2f} {r['vae']:6.3f} "
          f"{r['vae_sd']:5.3f} {r['tail']:6.3f} | {r['te']:6.2f} {r['tee']:7.3f} {r['tee_sd']:5.3f} "
          f"{r['tee_max']:6.2f} | {r['t']:4.0f}")
