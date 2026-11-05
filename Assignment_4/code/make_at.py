"""
Attention transfer(AT)용 OOF teacher attention map 생성
  python make_at.py --teacher t_res_dev_oof5        (fold teacher: ckpt/t_res_dev_oof5_f{0..4}.pt)
- 각 학습 이미지의 map은 그 이미지를 학습하지 않은 fold teacher(= 그 이미지가 held-out인 fold)로 계산 → logit KD와 같은 OOF 원칙
- attention map q = normalize( mean_c F^2 ) (Zagoruyko & Komodakis 2017), teacher 탭: 14×14 (R128 출력), 7×7 (R256 출력)
- 검증: 재계산한 held-out logits == 저장된 fold logits, 커버리지(dev 55,000 / full 60,000)
결과: results/teachers/{teacher}_at.pt = {A14 (60000,196) fp16, A7 (60000,49) fp16, covered (60000,) bool}
"""
import argparse
import os

import torch
import torch.nn.functional as F

from common import ROOT, load_mnist
from models import TokenNet

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--nfolds", type=int, default=5)
p.add_argument("--name", default=None)  # qdaemon 호환 (무시)
args = p.parse_args()
dev = torch.device("cuda")

d = load_mnist(dev, "full")  # 원본 60k 순서 그대로 (heldout_idx가 원본 기준 인덱스)
x_all = d["x_tr"]
A = {14: torch.zeros(60000, 196, dtype=torch.float16), 7: torch.zeros(60000, 49, dtype=torch.float16)}
covered = torch.zeros(60000, dtype=torch.bool)


def att(f):
    return F.normalize(f.float().pow(2).mean(1).flatten(1), dim=1)


for fold in range(args.nfolds):
    fname = f"{args.teacher}_f{fold}"
    ck = torch.load(os.path.join(ROOT, "results", "ckpt", f"{fname}.pt"), map_location=dev)
    tl = torch.load(os.path.join(ROOT, "results", "teachers", f"{fname}.pt"), map_location="cpu")
    m = TokenNet(ck["spec"], act=ck["act"]).to(dev).eval()
    m.load_state_dict(ck["state_dict"])
    taps = {}
    for i, layer in enumerate(m.net):  # 마지막 14×14 / 7×7 블록 출력에 hook
        layer.register_forward_hook(lambda mod, inp, out, i=i: taps.__setitem__(i, out))
    hid = tl["heldout_idx"].long()
    assert not covered[hid].any(), "fold held-out이 겹침"
    logits = []
    with torch.no_grad():
        for s in range(0, len(hid), 1000):
            idx = hid[s:s + 1000].to(dev)
            taps.clear()
            logits.append(m(x_all[idx]).float().cpu())
            by_hw = {}
            for i, out in taps.items():
                if out.ndim == 4:
                    by_hw[out.shape[-1]] = out  # 같은 해상도면 뒤쪽(더 깊은) 층이 남음
            for hw in (14, 7):
                A[hw][idx.cpu()] = att(by_hw[hw]).half().cpu()
    logits = torch.cat(logits)
    diff = (logits - tl["logits"][hid].float()).abs().max().item()
    agree = (logits.argmax(1) == tl["logits"][hid].float().argmax(1)).float().mean().item() * 100
    covered[hid] = True
    print(f"{fname}: held-out {len(hid)}  logits 재계산 max|diff|={diff:.2e} argmax 일치 {agree:.2f}%  "
          f"14×14 채널={by_hw[14].shape[1]} 7×7 채널={by_hw[7].shape[1]}", flush=True)
    assert agree > 99.9, "저장된 fold logits와 불일치"

print(f"커버리지 {int(covered.sum())} / 60000")
out = os.path.join(ROOT, "results", "teachers", f"{args.teacher}_at.pt")
torch.save({"A14": A[14], "A7": A[7], "covered": covered, "teacher": args.teacher}, out)
print("saved", out)
