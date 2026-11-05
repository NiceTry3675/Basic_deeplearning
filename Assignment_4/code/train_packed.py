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
p.add_argument("--kd-alpha-wrong", type=float, default=None,
               help="선택적 KD: teacher 예측이 학습 정답과 다른 샘플에만 쓸 α (기본: --kd-alpha와 동일)")
p.add_argument("--kd-T", type=float, default=4.0)
p.add_argument("--kd-mode", default="logit_avg", choices=["logit_avg", "prob_avg"])
p.add_argument("--sam", type=float, default=0.0, help="SAM rho (0: 끔)")
p.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기 (기본: 켬, 속도 2배)")
p.add_argument("--soup", type=int, default=0,
               help="가중치 평균(soup) 그룹 수 G: K개 멤버를 G개 그룹으로 나눠 그룹 안에서는 같은 초기값으로 출발, "
                    "학습 후 그룹별로 EMA 가중치를 평균 → 그룹당 단일 모델 1개 (0: 끔)")
p.add_argument("--branch-ep", type=int, default=0,
               help="soup: 처음 이 epoch 수만큼은 그룹 안 멤버가 같은 미니배치 순서를 공유(=같은 모델), 이후 독립 순서로 분기")
p.add_argument("--rep", default="",
               help="학습 전용 구조 재매개변수화 가지 (예: k11,k13,k31,id) — 학습 후 일반 res_c로 정확히 합쳐 저장")
p.add_argument("--at", default="", help="attention transfer: make_at.py가 만든 OOF teacher attention map 파일")
p.add_argument("--at-beta", default="1000",
               help="AT 가중치 β (Zagoruyko 방식: β/2·Σ_tap mean(q_s-q_t)²). 쉼표 목록이면 멤버별 (K의 약수 개, 블록 단위 배정)")
p.add_argument("--at-decay", type=float, default=0.25, help="마지막 이 비율의 step 동안 β를 0까지 선형 감소")
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
    t_correct = t_prob.argmax(1) == y_tr  # teacher 예측 == 학습 정답 (학습 라벨만 사용)
    if args.kd_alpha_wrong is not None:
        print(f"[KD] 선택적 α: teacher 정답 {int(t_correct.sum())}개 α={args.kd_alpha}, "
              f"teacher 오답 {int((~t_correct).sum())}개 α={args.kd_alpha_wrong}", flush=True)
    print(f"[KD] teachers={len(zs)} T={args.kd_T} a={args.kd_alpha} "
          f"teacher-train-acc={(t_prob.argmax(1) == y_tr).float().mean() * 100:.3f}", flush=True)

spec = STUDENTS.get(args.model, args.model)
cx = count_complexity(TokenNet(spec, act=args.act))
print(f"[model] {spec} params={cx['max_params']:,} MACs={cx['max_macs']:,} ok={cx['ok']} K={K}", flush=True)
rep_branches = [b for b in args.rep.split(",") if b] or None
assert not (rep_branches and args.soup), "--rep와 --soup 동시 사용은 지원하지 않음"
model = PackedTokenNet(spec, K, act=args.act, rep=rep_branches).to(dev)
if rep_branches:
    print(f"[rep] 학습 그래프 params/모델 {sum(p_.numel() for p_ in model.parameters()) // K:,} "
          f"(가지: 3x3+{rep_branches}) → 학습 후 배포 res_c로 합침", flush=True)

if args.soup:
    assert K % args.soup == 0, "K는 soup 그룹 수의 배수여야 함"
    M = K // args.soup  # 그룹당 멤버 수
    grp_first = (torch.arange(K, device=dev) // M) * M  # 각 멤버가 속한 그룹의 첫 멤버


    @torch.no_grad()
    def per_member(m):
        """packed 모델의 모든 (모델별) 텐서를 (K, -1) view로 — 멤버 간 가중치 복사/평균용"""
        return [v.view(K, -1) for v in m.state_dict().values() if v.ndim > 0 and v.dtype.is_floating_point]


    with torch.no_grad():  # 그룹 안 멤버를 같은 초기값으로
        for vv in per_member(model):
            vv.copy_(vv[grp_first])
    print(f"[soup] G={args.soup} 그룹 × {M} 멤버, 처음 {args.branch_ep} epoch 공유 후 분기", flush=True)

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

# ---- attention transfer: student 14×14(net[3]) / 7×7(net[5]) 출력의 attention map을 OOF teacher map에 맞춤
at_on = bool(args.at)
if at_on:
    AT = torch.load(args.at, map_location=dev)
    assert bool(AT["covered"][D["idx_tr"]].all()), "학습 이미지 중 OOF attention map이 없는 것이 있음"
    at_t = {14: AT["A14"][D["idx_tr"]].float(), 7: AT["A7"][D["idx_tr"]].float()}  # (N,HW), 학습셋 순서
    betas = [float(b) for b in args.at_beta.split(",")]
    assert K % len(betas) == 0
    at_beta = torch.tensor(betas, device=dev).repeat_interleave(K // len(betas))  # (K,) 멤버별 β
    feats = {}
    for i_tap in (3, 5):  # EMA 복사 이후에 등록 → EMA 모델에는 hook 없음
        model.net[i_tap].register_forward_hook(lambda mod, inp, out, i=i_tap: feats.__setitem__(i, out))
    at_sum = torch.zeros(K, device=dev)
    print(f"[AT] teacher maps {args.at}  β(멤버별)={at_beta.tolist()}  마지막 {args.at_decay:.0%} step 동안 β→0", flush=True)


def at_loss_per_member(idx):
    """멤버별 AT 손실 (K,): Σ_tap mean_{B,HW} (q_s - q_t)²,  q = normalize(mean_c F²)"""
    tot = 0
    for i_tap, hw in ((3, 14), (5, 7)):
        f = feats[i_tap].float()
        B_, KC, H, W = f.shape
        q = F.normalize(f.view(B_, K, KC // K, H * W).pow(2).mean(2), dim=2)  # (B,K,HW)
        tot = tot + (q - at_t[hw][idx]).pow(2).mean((0, 2))
    return tot
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
    if args.soup and ep <= args.branch_ep:
        perms = perms[grp_first]  # 분기 전: 그룹 안 멤버가 같은 순서 → 같은 모델로 유지
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
                if args.kd_alpha_wrong is not None:
                    # 샘플별 α: 평균 내기 전에 CE/KL을 샘플 단위로 유지 → (B,K)
                    ce_i = F.cross_entropy(out.reshape(-1, 10), yb.reshape(-1), label_smoothing=args.ls,
                                           reduction="none").reshape(yb.shape)
                    kl_i = (tp * (torch.log(tp.clamp_min(1e-12)) - logp)).sum(2) * T * T
                    a = torch.where(t_correct[idx], args.kd_alpha, args.kd_alpha_wrong)
                    return ((1 - a) * ce_i + a * kl_i).mean()
                kl = (tp * (torch.log(tp.clamp_min(1e-12)) - logp)).sum(2).mean() * T * T
                loss_ = (1 - args.kd_alpha) * ce + args.kd_alpha * kl
            else:
                loss_ = ce
            if at_on:
                at_k = at_loss_per_member(idx)  # (K,)
                frac = step / total
                mult = 1.0 if frac < 1 - args.at_decay else max(0.0, (1 - frac) / args.at_decay)
                at_stat.append(at_k.detach())
                # (loss*K).backward() → 멤버 k의 gradient에는 β_k/2 · at_k 만 들어감
                loss_ = loss_ + (at_beta * mult / 2 * at_k).mean()
            return loss_

        at_stat = []
        loss = compute_loss()
        if at_on:
            at_sum += at_stat[0]
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
    if at_on:
        at_ep = (at_sum / spe).tolist()
        hist.setdefault("at_dist", []).append(at_ep)
        at_sum.zero_()
        grp = len(betas)
        msg += " at[" + " ".join(f"β{betas[g]:g}:{sum(at_ep[g * K // grp:(g + 1) * K // grp]) / (K // grp):.4f}"
                                 for g in range(grp)) + "]"
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
raw_final = model
if rep_branches:
    # 재매개변수화 가지를 합쳐 배포용 일반 res_c로 변환하고, 합치기 전후 출력이 같은지 확인
    xs_chk = D["x_val"] if D["x_val"] is not None else D["x_te"]
    with torch.no_grad():
        final.eval()
        plain_ema = final.to_plain().eval()
        dmax = (final(xs_chk[:2000]).float() - plain_ema(xs_chk[:2000]).float()).abs().max().item()
    a_rep = acc_per_model(final, xs_chk, D["y_val"] if D["x_val"] is not None else D["y_te"])
    a_pl = acc_per_model(plain_ema, xs_chk, D["y_val"] if D["x_val"] is not None else D["y_te"])
    print(f"[rep] 합치기 검증: max|logit diff|={dmax:.2e}  정확도 동일={a_rep == a_pl}", flush=True)
    assert dmax < 1e-2, "재매개변수화 합치기 오류"
    final, raw_final = plain_ema, model.to_plain().eval()

import copy as _copy


def bn_recalc(m_src):
    """BN running stats를 학습 데이터로 다시 추정 (누적 평균, dropout은 eval 유지)"""
    m2 = _copy.deepcopy(m_src)
    m2.eval()
    for m_ in m2.modules():
        if isinstance(m_, torch.nn.BatchNorm2d):
            m_.reset_running_stats(); m_.momentum = None; m_.train()
    with torch.no_grad():
        for i in range(0, N, 500):
            m2(x_tr[i:i + 500].unsqueeze(1))
    return m2


# EMA 가중치에 맞춰 BN running stats를 학습 데이터로 다시 추정한 버전도 평가 (기록용 비교)
bnrc = None
if ema:
    bnrc = bn_recalc(ema.ema)
    bn_val = acc_per_model(bnrc, D["x_val"], D["y_val"]) if D["x_val"] is not None else None
    bn_test = acc_per_model(bnrc, D["x_te"], D["y_te"]) if hist["test_ema"] else None
    if bn_val:
        print(f"[BN recalc] val_ema {sum(hist['val_ema'][-1]) / K:.3f} -> {sum(bn_val) / K:.3f}", flush=True)

# ---- soup: 그룹별 EMA 가중치 평균 → BN 재추정 → 평가 (그룹 안 멤버는 평균 후 모두 같은 모델)
soup = None
if args.soup:
    G = args.soup
    first = [g * M for g in range(G)]
    sp = _copy.deepcopy(final)
    with torch.no_grad():
        for vv in per_member(sp):
            v3 = vv.view(G, M, -1)
            v3.copy_(v3.mean(1, keepdim=True).expand_as(v3).clone())
    sp_bn = bn_recalc(sp)


    @torch.no_grad()
    def ens_acc(m, x, y, bs=2000):
        """참고용: 그룹 멤버 출력(softmax) 평균 앙상블 정확도 (파라미터 M배라 제출 불가, soup의 상한 비교용)"""
        m.eval()
        correct = torch.zeros(G, device=dev)
        for i in range(0, len(x), bs):
            pr = m(x[i:i + bs]).float().softmax(2).view(-1, G, M, 10).mean(2)
            correct += (pr.argmax(2) == y[i:i + bs, None]).float().sum(0)
        return (correct / len(x) * 100).tolist()


    soup = {}
    for tag, xs, ys in [("val", D["x_val"], D["y_val"]), ("test", D["x_te"], D["y_te"])]:
        if xs is None or (tag == "test" and not (args.split == "full" or args.test_every_epoch)):
            continue
        a_bn, a_nobn = acc_per_model(sp_bn, xs, ys), acc_per_model(sp, xs, ys)
        soup[f"soup_{tag}_bnrc"] = [a_bn[i] for i in first]
        soup[f"soup_{tag}_nobn"] = [a_nobn[i] for i in first]
        soup[f"ens_{tag}"] = ens_acc(final, xs, ys)
        mem = hist[f"{tag}_ema"][-1]
        soup[f"member_{tag}"] = [sum(mem[g * M:(g + 1) * M]) / M for g in range(G)]
        print(f"[soup {tag}] 멤버 평균 {sum(mem) / K:.3f} | soup(BN재추정) {sum(soup[f'soup_{tag}_bnrc']) / G:.3f} "
              f"{[round(v, 2) for v in soup[f'soup_{tag}_bnrc']]} | soup(BN평균) {sum(soup[f'soup_{tag}_nobn']) / G:.3f} "
              f"| 출력앙상블(참고) {sum(soup[f'ens_{tag}']) / G:.3f}", flush=True)
    if args.split == "full" or args.save_dev_ckpt:
        for g in range(G):
            torch.save({"spec": spec, "act": args.act, "state_dict": sp_bn.extract(first[g]).state_dict(),
                        "args": vars(args), "group": g, "members": list(range(g * M, (g + 1) * M))},
                       os.path.join(ckdir, f"{args.name}_g{g}.pt"))

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
    if bnrc is not None:
        if D["x_val"] is not None:
            r["final_val_ema_bnrc"] = bn_val[k]
        if bn_test is not None:
            r["final_test_ema_bnrc"] = bn_test[k]
    if at_on:
        r["at_beta"] = at_beta[k].item()
        r["hist"]["at_dist"] = [h[k] for h in hist["at_dist"]]
    if soup is not None:
        g = k // M
        r["soup_group"] = g
        for key, vals in soup.items():
            r[key] = vals[g]
    with open(os.path.join(outdir, f"{args.name}_s{k}.json"), "w") as f:
        json.dump(r, f)
    if args.split == "full" or args.save_dev_ckpt:
        torch.save({"spec": spec, "act": args.act, "state_dict": final.extract(k).state_dict(),
                    "raw_state_dict": raw_final.extract(k).state_dict(), "args": vars(args), "k": k},
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
if soup is not None:
    summ.update({k_: v_ for k_, v_ in soup.items() if k_.startswith("soup_") and k_.endswith("bnrc")})
print(f"DONE {args.name} time={el:.0f}s " + json.dumps(summ))
