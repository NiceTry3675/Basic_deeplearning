"""
OOF teacher logits 합치기: 각 학습 이미지에 대해 '그 이미지를 학습에 쓰지 않은' fold teacher의 logits 사용
  python combine_oof.py --name <out_name> --parts <fold0>,<fold1>,...
  (또는 예전 방식) python combine_oof.py <out_name> <fold0> <fold1> ...
  학습셋에 속하지 않는 이미지(dev의 val 5k)는 모든 fold teacher 평균 → val 정확도 보고에 사용
  큐 데몬에서 실행할 수 있도록 results/runs/<out_name>.json 완료 표시를 남긴다.
"""
import argparse
import json
import sys

import numpy as np
import torch

from common import DATA_PATH, ROOT, SPLIT_SEED, VAL_SIZE

if "--name" in sys.argv:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--parts", required=True)
    a = p.parse_args()
    out, names = a.name, a.parts.split(",")
else:
    out, names = sys.argv[1], sys.argv[2:]

L = [torch.load(f"{ROOT}/results/teachers/{n}.pt") for n in names]
z = torch.stack([l["logits"] for l in L]).mean(0)  # 기본: 평균
covered = torch.zeros(len(z), dtype=torch.bool)
for l in L:
    h = l["heldout_idx"]
    z[h] = l["logits"][h]
    covered[h] = True
y = torch.from_numpy(np.load(DATA_PATH)["y_train"].astype(np.int64))
acc_oof = ((z[covered].argmax(1) == y[covered]).float().mean() * 100).item()
res = {"name": out, "parts": names, "oof_coverage": int(covered.sum()), "oof_acc": acc_oof}
msg = f"OOF coverage {covered.sum().item()} imgs, OOF acc {acc_oof:.3f}"
if L[0]["split"] == "dev":
    idx_val = np.sort(np.random.RandomState(SPLIT_SEED).permutation(len(y))[:VAL_SIZE])
    res["val_acc_foldavg"] = ((z[idx_val].argmax(1) == y[idx_val]).float().mean() * 100).item()
    msg += f", val acc (fold 평균) {res['val_acc_foldavg']:.3f}"
print(msg)
torch.save({"logits": z, "split": L[0]["split"], "name": out, "parts": names}, f"{ROOT}/results/teachers/{out}.pt")
with open(f"{ROOT}/results/runs/{out}.json", "w") as f:  # 데몬 완료 표시 (summarize.py는 hist 없는 항목 무시)
    json.dump(res, f)
print(f"DONE {out}")
