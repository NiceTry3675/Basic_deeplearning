"""
MNIST 학습 스크립트 (student / teacher 공용)

예)
  python train.py --model vgg_a --split dev --epochs 60 --seed 0 --name vgg_a_s0
  python train.py --model t_res --split full --nfolds 5 --fold 0 --save-logits --name t_res_full_oof5_f0   # OOF teacher
  python combine_oof.py --name t_res_full_oof5 --parts t_res_full_oof5_f0,...,t_res_full_oof5_f4
  python train.py --model vgg_a --split full --kd ../results/teachers/t_res_full_oof5.pt --name ...
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import EMA, ROOT, accuracy, count_complexity, load_mnist, predict_logits
from models import build

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--act", default="relu")
p.add_argument("--split", default="dev", choices=["dev", "full"])
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--opt", default="sgd", choices=["sgd", "adamw", "adam"])
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--sched", default="onecycle", choices=["onecycle", "cosine", "const"])
p.add_argument("--warmup", type=float, default=0.15, help="warmup 비율 (전체 step 대비)")
p.add_argument("--final-lr-frac", type=float, default=1e-3)
p.add_argument("--ls", type=float, default=0.0, help="label smoothing")
p.add_argument("--ema", type=float, default=0.999, help="0이면 EMA 사용 안 함")
p.add_argument("--kd", default="", help="teacher logits .pt 경로 (쉼표로 여러 개 → 평균)")
p.add_argument("--kd-alpha", type=float, default=0.9)
p.add_argument("--kd-T", type=float, default=4.0)
p.add_argument("--kd-mode", default="logit_avg", choices=["logit_avg", "prob_avg"])
p.add_argument("--amp", action="store_true", help="bf16 autocast (teacher 속도용)")
p.add_argument("--compile", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save-logits", action="store_true", help="학습 60k 전체에 대한 logits 저장 (teacher)")
p.add_argument("--test-every-epoch", action="store_true", help="[트랙B 분석용] 매 epoch test acc 기록")
p.add_argument("--nfolds", type=int, default=0, help="OOF teacher: 학습셋을 nfolds로 나눔")
p.add_argument("--fold", type=int, default=0, help="OOF teacher: 이 fold를 빼고 학습 → 이 fold를 예측")
p.add_argument("--name", required=True)
p.add_argument("--outdir", default=os.path.join(ROOT, "results", "runs"))
args = p.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

D = load_mnist(dev, args.split)
heldout_idx = None
if args.nfolds > 0:
    # 현재 학습셋(dev: 55k / full: 60k)을 고정 시드로 nfolds 분할, fold번째를 제외하고 학습
    g = torch.Generator().manual_seed(777)
    perm = torch.randperm(len(D["x_tr"]), generator=g).to(dev)
    folds = perm.chunk(args.nfolds)
    keep = torch.cat([f for i, f in enumerate(folds) if i != args.fold])
    heldout_idx = D["idx_tr"][folds[args.fold]].cpu()  # 원본 60k 기준
    for k in ("x_tr", "y_tr", "idx_tr"):
        D[k] = D[k][keep]
    print(f"[OOF] fold {args.fold}/{args.nfolds}: train {len(keep)}, held-out {len(heldout_idx)}")
x_tr, y_tr = D["x_tr"], D["y_tr"]
N = len(x_tr)

# ---------------------------------------------------------------- teacher logits
if args.kd:
    zs = [torch.load(pth, map_location=dev)["logits"].float() for pth in args.kd.split(",")]
    if args.kd_mode == "logit_avg":
        z = torch.stack(zs).mean(0)
        t_prob_full = F.softmax(z / args.kd_T, 1)
    else:
        t_prob_full = torch.stack([F.softmax(z / args.kd_T, 1) for z in zs]).mean(0)
    t_prob = t_prob_full[D["idx_tr"]]  # 원본 60k 인덱스 → 현재 학습 셋 순서
    t_acc = (t_prob.argmax(1) == y_tr).float().mean().item() * 100
    print(f"[KD] teachers={len(zs)} T={args.kd_T} alpha={args.kd_alpha} teacher-train-acc={t_acc:.3f}")

# ---------------------------------------------------------------- model / optim
model = build(args.model, act=args.act).to(dev).to(memory_format=torch.channels_last)
cx = count_complexity(model)
print(f"[model] {model.spec}  params={cx['max_params']:,}  MACs(thop)={cx.get('thop_macs')}  "
      f"MACs(ptflops)={cx.get('ptflops_macs')}  ok={cx['ok']}")

decay, no_decay = [], []
for n_, p_ in model.named_parameters():
    (no_decay if p_.ndim <= 1 else decay).append(p_)
groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
if args.opt == "sgd":
    opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9, nesterov=True)
elif args.opt == "adamw":
    opt = torch.optim.AdamW(groups, lr=args.lr)
else:
    opt = torch.optim.Adam(groups, lr=args.lr)

steps_per_epoch = N // args.bs
total = steps_per_epoch * args.epochs
warm = int(total * args.warmup)


def lr_at(step):
    if args.sched == "const":
        return args.lr
    if step < warm:
        return args.lr * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    lo = args.lr * args.final_lr_frac
    return lo + 0.5 * (args.lr - lo) * (1 + math.cos(math.pi * t))


ema = EMA(model, args.ema) if args.ema > 0 else None
fwd = torch.compile(model) if args.compile else model


def loss_fn(logits, y, idx):
    ce = F.cross_entropy(logits, y, label_smoothing=args.ls)
    if not args.kd:
        return ce
    T = args.kd_T
    kl = F.kl_div(F.log_softmax(logits / T, 1), t_prob[idx], reduction="batchmean") * T * T
    return (1 - args.kd_alpha) * ce + args.kd_alpha * kl


hist = {"epoch": [], "train_loss": [], "val_acc": [], "val_acc_ema": [], "test_acc": [], "test_acc_ema": []}
t0 = time.time()
step = 0
for ep in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(N, device=dev)
    loss_sum = torch.zeros((), device=dev)
    for i in range(steps_per_epoch):
        idx = perm[i * args.bs:(i + 1) * args.bs]
        xb, yb = x_tr[idx], y_tr[idx]
        xb = xb.contiguous(memory_format=torch.channels_last)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            out = fwd(xb)
        loss = loss_fn(out.float(), yb, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema:
            ema.update(model)
        loss_sum += loss.detach()
        step += 1
    hist["epoch"].append(ep)
    hist["train_loss"].append(loss_sum.item() / steps_per_epoch)
    msg = f"ep {ep:3d} loss {hist['train_loss'][-1]:.4f}"
    if D["x_val"] is not None:
        va = accuracy(model, D["x_val"], D["y_val"])
        hist["val_acc"].append(va)
        msg += f" val {va:.2f}"
        if ema:
            vae = accuracy(ema.ema, D["x_val"], D["y_val"])
            hist["val_acc_ema"].append(vae)
            msg += f" val_ema {vae:.2f}"
    if args.test_every_epoch or ep == args.epochs and args.split == "full":
        te = accuracy(model, D["x_te"], D["y_te"])
        hist["test_acc"].append(te)
        msg += f" test {te:.2f}"
        if ema:
            tee = accuracy(ema.ema, D["x_te"], D["y_te"])
            hist["test_acc_ema"].append(tee)
            msg += f" test_ema {tee:.2f}"
    if ep % 5 == 0 or ep == args.epochs or ep == 1:
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)

final_model = ema.ema if ema else model
res = dict(name=args.name, args=vars(args), spec=model.spec, complexity=cx, hist=hist,
           time=time.time() - t0)
if D["x_val"] is not None:
    res["final_val"] = hist["val_acc"][-1]
    res["final_val_ema"] = hist["val_acc_ema"][-1] if ema else None
    res["val_errors"] = int(round((100 - (res["final_val_ema"] or res["final_val"])) / 100 * len(D["y_val"])))
if args.split == "full":
    res["final_test"] = hist["test_acc"][-1]
    res["final_test_ema"] = hist["test_acc_ema"][-1] if ema else None

os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, f"{args.name}.json"), "w") as f:
    json.dump(res, f, indent=1)
ck_dir = os.path.join(ROOT, "results", "ckpt")
os.makedirs(ck_dir, exist_ok=True)
torch.save({"spec": model.spec, "act": args.act, "state_dict": final_model.state_dict(),
            "raw_state_dict": model.state_dict()}, os.path.join(ck_dir, f"{args.name}.pt"))

if args.save_logits:
    # 학습 데이터 60k 전체(분할 무관, 원본 순서)에 대한 teacher logits 저장. test 데이터는 사용하지 않음.
    Dall = load_mnist(dev, "full")
    z = predict_logits(final_model, Dall["x_tr"])
    tdir = os.path.join(ROOT, "results", "teachers")
    os.makedirs(tdir, exist_ok=True)
    torch.save({"logits": z.cpu(), "split": args.split, "spec": model.spec, "name": args.name,
                "heldout_idx": heldout_idx, "train_idx": D["idx_tr"].cpu()},
               os.path.join(tdir, f"{args.name}.pt"))
    print(f"[logits] saved; teacher acc on full-train = {(z.argmax(1) == Dall['y_tr']).float().mean() * 100:.3f}")

print(f"DONE {args.name} " + json.dumps({k: res.get(k) for k in
                                         ["final_val", "final_val_ema", "final_test", "final_test_ema"]})
      + f" time={res['time']:.0f}s")
