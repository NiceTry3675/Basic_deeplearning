"""
K개 독립 student를 한 프로세스에서 동시 학습 (seed 스윕/하이퍼파라미터 탐색용)
결과: results/runs/{name}_s{k}.json (모델별), full split이면 results/ckpt/{name}_s{k}.pt (일반 TokenNet 가중치)

  python train_packed.py --model vgg_a --K 8 --split dev --name xxx
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common import EMA, ROOT, count_complexity, load_mnist
from models import STUDENTS, TokenNet
from packed import PackedTokenNet

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--act", default="relu")
p.add_argument("--K", type=int, default=8)
p.add_argument("--split", default="dev", choices=["dev", "full"])
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--opt", default="sgd", choices=["sgd", "adamw", "adam"])
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--mom", type=float, default=0.9)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--sched", default="onecycle", choices=["onecycle", "const", "linear"])
p.add_argument("--warmup", type=float, default=0.15)
p.add_argument("--final-lr-frac", type=float, default=1e-3)
p.add_argument("--ls", type=float, default=0.0)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--kd", default="")
p.add_argument("--kd-alpha", type=float, default=0.9)
p.add_argument("--kd-T", type=float, default=4.0)
p.add_argument("--kd-mode", default="logit_avg", choices=["logit_avg", "prob_avg"])
p.add_argument("--sam", type=float, default=0.0, help="SAM rho (0: 끔)")
p.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기 (기본: 켬, 속도 2배)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--test-every-epoch", action="store_true")
p.add_argument("--save-dev-ckpt", action="store_true")
p.add_argument("--name", required=True)
args = p.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")
torch.manual_seed(args.seed)
K = args.K

D = load_mnist(dev, args.split)
x_tr, y_tr = D["x_tr"][:, 0], D["y_tr"]  # (N,28,28)
N = len(x_tr)

if args.kd:
    zs = [torch.load(pth, map_location=dev)["logits"].float() for pth in args.kd.split(",")]
    if args.kd_mode == "logit_avg":
        t_prob = F.softmax(torch.stack(zs).mean(0) / args.kd_T, 1)
    else:
        t_prob = torch.stack([F.softmax(z / args.kd_T, 1) for z in zs]).mean(0)
    t_prob = t_prob[D["idx_tr"]]
    print(f"[KD] teachers={len(zs)} T={args.kd_T} a={args.kd_alpha} "
          f"teacher-train-acc={(t_prob.argmax(1) == y_tr).float().mean() * 100:.3f}", flush=True)

spec = STUDENTS.get(args.model, args.model)
cx = count_complexity(TokenNet(spec, act=args.act))
print(f"[model] {spec} params={cx['max_params']:,} MACs={cx['max_macs']:,} ok={cx['ok']} K={K}", flush=True)
model = PackedTokenNet(spec, K, act=args.act).to(dev)

decay, no_decay = [], []
for n_, p_ in model.named_parameters():
    (no_decay if p_.ndim <= 1 else decay).append(p_)
groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
if args.opt == "sgd":
    opt = torch.optim.SGD(groups, lr=args.lr, momentum=args.mom, nesterov=True)
elif args.opt == "adamw":
    opt = torch.optim.AdamW(groups, lr=args.lr)
else:
    opt = torch.optim.Adam(groups, lr=args.lr)

spe = N // args.bs
total = spe * args.epochs
warm = int(total * args.warmup)


def lr_at(step):
    if args.sched == "const":
        return args.lr
    if step < warm:
        return args.lr * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    lo = args.lr * args.final_lr_frac
    if args.sched == "linear":
        return lo + (args.lr - lo) * (1 - t)
    return lo + 0.5 * (args.lr - lo) * (1 + math.cos(math.pi * t))


ema = EMA(model, args.ema) if args.ema > 0 else None
bns = [m_ for m_ in model.modules() if isinstance(m_, torch.nn.BatchNorm2d)]


@torch.no_grad()
def acc_per_model(m, x, y, bs=2000):
    m.eval()
    correct = torch.zeros(K, device=dev)
    for i in range(0, len(x), bs):
        out = m(x[i:i + bs])  # (B,K,10)  (평가는 fp32)
        correct += (out.argmax(2) == y[i:i + bs, None]).float().sum(0)
    return (correct / len(x) * 100).tolist()


hist = {"train_loss": [], "val": [], "val_ema": [], "test": [], "test_ema": []}
t0 = time.time()
step = 0
ar = torch.arange(K, device=dev)
for ep in range(1, args.epochs + 1):
    model.train()
    perms = torch.argsort(torch.rand(K, N, device=dev), dim=1)  # 모델별 독립 셔플
    loss_sum = torch.zeros((), device=dev)
    for i in range(spe):
        idx = perms[:, i * args.bs:(i + 1) * args.bs].T  # (B,K)
        xb, yb = x_tr[idx], y_tr[idx]  # (B,K,28,28), (B,K)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        def compute_loss():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
                out = model(xb).float()  # (B,K,10)
            ce = F.cross_entropy(out.reshape(-1, 10), yb.reshape(-1), label_smoothing=args.ls)
            if args.kd:
                T = args.kd_T
                logp = F.log_softmax(out / T, 2)
                tp = t_prob[idx]  # (B,K,10)
                kl = (tp * (torch.log(tp.clamp_min(1e-12)) - logp)).sum(2).mean() * T * T
                return (1 - args.kd_alpha) * ce + args.kd_alpha * kl
            return ce

        loss = compute_loss()
        opt.zero_grad(set_to_none=True)
        (loss * K).backward()  # 모델별 평균 손실의 합 → 각 모델의 gradient는 단독 학습과 동일
        if args.sam > 0:
            # SAM: 모델별 gradient norm으로 w + rho*g/||g|| 지점의 gradient 사용 (모델 간 독립 유지)
            params = [p_ for p_ in model.parameters() if p_.grad is not None]
            with torch.no_grad():
                sq = sum(p_.grad.float().reshape(K, -1).pow(2).sum(1) for p_ in params)  # (K,)
                scale = args.sam / (sq.sqrt() + 1e-12)
                eps = []
                for p_ in params:
                    e = p_.grad.reshape(K, -1) * scale[:, None]
                    e = e.reshape(p_.shape)
                    p_.add_(e)
                    eps.append(e)
            opt.zero_grad(set_to_none=True)
            for bn in bns:
                bn.momentum = 0.0  # 두 번째 forward에서는 BN running stats 갱신 안 함
            (compute_loss() * K).backward()
            for bn in bns:
                bn.momentum = 0.1
            with torch.no_grad():
                for p_, e in zip(params, eps):
                    p_.sub_(e)
        opt.step()
        if ema:
            ema.update(model)
        loss_sum += loss.detach()
        step += 1
    hist["train_loss"].append(loss_sum.item() / spe)
    msg = f"ep {ep:3d} loss {hist['train_loss'][-1]:.4f}"
    if D["x_val"] is not None:
        hist["val"].append(acc_per_model(model, D["x_val"], D["y_val"]))
        msg += f" val {sum(hist['val'][-1]) / K:.3f}"
        if ema:
            hist["val_ema"].append(acc_per_model(ema.ema, D["x_val"], D["y_val"]))
            msg += f" val_ema {sum(hist['val_ema'][-1]) / K:.3f} (max {max(hist['val_ema'][-1]):.2f})"
    if args.test_every_epoch or (ep == args.epochs and args.split == "full"):
        hist["test"].append(acc_per_model(model, D["x_te"], D["y_te"]))
        msg += f" test {sum(hist['test'][-1]) / K:.3f}"
        if ema:
            hist["test_ema"].append(acc_per_model(ema.ema, D["x_te"], D["y_te"]))
            msg += f" test_ema {sum(hist['test_ema'][-1]) / K:.3f} (max {max(hist['test_ema'][-1]):.2f})"
    if ep % 10 == 0 or ep == args.epochs or ep == 1:
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)

el = time.time() - t0
outdir = os.path.join(ROOT, "results", "runs")
ckdir = os.path.join(ROOT, "results", "ckpt")
os.makedirs(outdir, exist_ok=True)
os.makedirs(ckdir, exist_ok=True)
final = ema.ema if ema else model
for k in range(K):
    sub = lambda key: [h[k] for h in hist[key]]
    r = dict(name=f"{args.name}_s{k}", args=vars(args), spec=spec, complexity=cx, time=el / K,
             hist={"train_loss": hist["train_loss"], "val_acc": sub("val"), "val_acc_ema": sub("val_ema"),
                   "test_acc": sub("test"), "test_acc_ema": sub("test_ema")})
    if D["x_val"] is not None:
        r["final_val"] = hist["val"][-1][k]
        r["final_val_ema"] = hist["val_ema"][-1][k] if ema else None
    if hist["test"] and (args.split == "full" or args.test_every_epoch):
        r["final_test"] = hist["test"][-1][k]
        r["final_test_ema"] = hist["test_ema"][-1][k] if ema else None
    with open(os.path.join(outdir, f"{args.name}_s{k}.json"), "w") as f:
        json.dump(r, f)
    if args.split == "full" or args.save_dev_ckpt:
        torch.save({"spec": spec, "act": args.act, "state_dict": final.extract(k).state_dict(),
                    "raw_state_dict": model.extract(k).state_dict(), "args": vars(args), "k": k},
                   os.path.join(ckdir, f"{args.name}_s{k}.pt"))
# 사후 분석용 최종 logits 저장 (val: dev split, test: full split) — (N, K, 10) float16
@torch.no_grad()
def logits_all(m, x, bs=2000):
    m.eval()
    return torch.cat([m(x[i:i + bs]) for i in range(0, len(x), bs)]).half().cpu()


lgdir = os.path.join(ROOT, "results", "logits")
os.makedirs(lgdir, exist_ok=True)
lg = {}
if D["x_val"] is not None:
    lg["val"] = logits_all(final, D["x_val"])
if args.split == "full" or args.test_every_epoch:
    lg["test"] = logits_all(final, D["x_te"])
torch.save(lg, os.path.join(lgdir, f"{args.name}.pt"))

summ = {"val_ema": hist["val_ema"][-1] if hist["val_ema"] else None,
        "test_ema": hist["test_ema"][-1] if hist["test_ema"] else None}
print(f"DONE {args.name} time={el:.0f}s " + json.dumps(summ))
