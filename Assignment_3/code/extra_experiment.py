"""
추가 실험: 증강·앙상블 없이 단일 CNN 성능 끌어올리기 (과제 필수 범위 밖)
  A: model_v3 CNN(28) 그대로 + 학습 설정 개선
  B: A의 학습 설정 + 구조 개선(3×3 합성곱 2회씩 쌓은 블록 3단 + Global Average Pooling)

  python extra_experiment.py            # A, B × seed 3개 학습·테스트 (GPU 기준 약 5분)
  python extra_experiment.py --plot     # 그림만 다시 생성
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import load_mnist
from model_v3 import CNN

RESULTS = os.path.join("..", "results")
LOGS    = os.path.join(RESULTS, "logs", "extra")
MODELS  = os.path.join(RESULTS, "models")
FIGS    = os.path.join(RESULTS, "figures")
SEEDS   = [0, 1, 2]

# 학습 설정 (A, B 공통)
EPOCHS, BATCH, MAX_LR, WD, SMOOTH = 15, 128, 0.1, 5e-4, 0.1

#%% B 모델
class ConvBlock(nn.Sequential):
    """(3×3 conv → BN → ReLU) × 2: 5×5 수용영역을 더 적은 파라미터로"""
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

class CNNPlus(nn.Module):
    """
    28×28 → block1(32) → pool → 14×14 → block2(64) → pool → 7×7 → block3(128) → GAP → FC(10)
    세 번째 풀링은 하지 않음: 7 → 3으로 줄면 마지막 행·열이 버려지고 공간 정보가 너무 일찍 사라짐
    """
    def __init__(self, in_size: int=28, dropout: float=0.3):
        super().__init__()
        self.in_size = in_size
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), 1, self.in_size, self.in_size)
        x = self.pool(self.block1(x))          # (B, 32, 14, 14)
        x = self.pool(self.block2(x))          # (B, 64, 7, 7)
        x = self.block3(x)                     # (B, 128, 7, 7)
        x = x.mean(dim=(2, 3))                 # Global Average Pooling → (B, 128)
        return self.fc(self.dropout(x))

MODELS_BY_CFG = {"A": lambda: CNN(28), "B": lambda: CNNPlus(28)}

#%% 학습·평가
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, preds = 0.0, 0, []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            preds.append(p.cpu())
    n = len(loader.dataset)
    return loss_sum / n, 100.0 * correct / n, torch.cat(preds).numpy()

def run(cfg, seed, device):
    torch.manual_seed(seed)
    train_ds, val_ds, test_ds, _, _, y_test = load_mnist(use_small=False, valDB_portion=0.1, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1000)
    test_loader  = DataLoader(test_ds,  batch_size=1000)

    model = MODELS_BY_CFG[cfg]().to(device)
    # momentum=0.9는 초기값일 뿐: OneCycleLR(cycle_momentum=True)가 학습률과 반대로 0.95 → 0.85 → 0.95로 조절함
    optimizer = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=0.9, nesterov=True, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS,
                                                    steps_per_epoch=len(train_loader))
    train_crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    eval_crit  = nn.CrossEntropyLoss()   # 평가 손실은 스무딩 없이 (기준 실험과 비교 가능하도록)

    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = train_crit(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        vl, va, _ = evaluate(model, val_loader, eval_crit, device)
        hist["train_loss"].append(loss_sum / len(train_loader))
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        hist["lr"].append(scheduler.get_last_lr()[0])
        print(f"[{cfg} seed{seed} ep{epoch:2d}] train_loss {hist['train_loss'][-1]:.4f}  val_loss {vl:.4f}  val_acc {va:.2f}%")
    elapsed = time.perf_counter() - start

    # 하이퍼파라미터는 고정 레시피(검증셋으로 고르지 않음), 테스트는 마지막 에폭 모델로 한 번만
    tl, ta, pred = evaluate(model, test_loader, eval_crit, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg} seed{seed}] TEST acc {ta:.2f}%  wrong {int((pred != y_test).sum())}  params {n_params:,}  time {elapsed:.1f}s")
    torch.save(model.state_dict(), os.path.join(MODELS, f"extra_{cfg}_seed{seed}.pth"))
    log = {"cfg": cfg, "seed": seed, "epochs": EPOCHS, "batch": BATCH, "max_lr": MAX_LR, "weight_decay": WD,
           "label_smoothing": SMOOTH, "params": n_params, "train_time_sec": elapsed,
           "test_acc": ta, "test_loss": tl, "n_wrong": int((pred != y_test).sum()), "y_pred": pred.tolist(), **hist}
    with open(os.path.join(LOGS, f"extra_{cfg}_seed{seed}.json"), "w") as f:
        json.dump(log, f)

#%% 그림
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from thop import profile
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    COLORS = {"base": "#8a8985", "A": "#eb6834", "B": "#2a78d6"}
    NAMES  = {"base": "Baseline (lecture setting)", "A": "A: better training", "B": "B: A + deeper CNN + GAP"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "legend.frameon": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    load = lambda p: json.load(open(p))
    runs = {"base": [load(os.path.join(RESULTS, "logs", f"train_cnn_28_seed{k}.json")) for k in SEEDS],
            "A": [load(os.path.join(LOGS, f"extra_A_seed{k}.json")) for k in SEEDS],
            "B": [load(os.path.join(LOGS, f"extra_B_seed{k}.json")) for k in SEEDS]}

    # 1) 검증 오류율(로그 축) + 학습률 스케줄
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for c, rs in runs.items():
        err = 100 - np.array([r["val_acc"] for r in rs])
        m, s = err.mean(0), err.std(0, ddof=1)
        ep = np.arange(1, len(m) + 1)
        ax.plot(ep, m, "-o", ms=3.5, lw=2, color=COLORS[c], label=NAMES[c])
        ax.fill_between(ep, m - s, m + s, color=COLORS[c], alpha=0.15, lw=0)
        ax.annotate(f"{m[-1]:.2f}%", (ep[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.3, 0.5, 1, 2]); ax.set_yticklabels(["0.3", "0.5", "1", "2"]); ax.set_ylim(0.3, 3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation error (%, log scale)")
    ax.set_title("Validation error (mean ± std over 3 seeds)", color=INK)
    ax.set_xlim(0.5, 16.8); ax.set_xticks(range(1, 16)); ax.legend(loc="upper right")
    ax = axes[1]
    lr = runs["B"][0]["lr"]
    ax.plot(np.arange(1, len(lr) + 1), lr, "-o", ms=3.5, lw=2, color=COLORS["B"], label="A, B: OneCycleLR")
    ax.axhline(0.01, color=COLORS["base"], lw=2, ls="--", label="Baseline: constant 0.01")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate (end of epoch)")
    ax.set_title("Learning-rate schedule", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "extra_val_error.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # 2) B의 특징맵: 층이 깊어질수록 무엇을 보는가
    _, _, _, _, x_test, y_test = load_mnist(use_small=False)
    model = CNNPlus(28)
    model.load_state_dict(torch.load(os.path.join(MODELS, "extra_B_seed0.pth"), map_location="cpu"))
    model.eval()
    feats = {}
    for name in ("block1", "block2", "block3"):
        getattr(model, name).register_forward_hook(lambda m, i, o, n=name: feats.__setitem__(n, o[0]))
    idx = 12
    with torch.no_grad():
        prob = torch.softmax(model(torch.FloatTensor(x_test[idx].reshape(1, -1))), 1)[0]
    n_show = 8
    fig, axes = plt.subplots(4, n_show, figsize=(n_show * 1.25, 4 * 1.45))
    for a in axes.ravel():
        a.axis("off")
    axes[0, 0].imshow(x_test[idx], cmap="gray")
    axes[0, 0].set_title(f"input #{idx}\ntrue {y_test[idx]}, pred {int(prob.argmax())}", fontsize=8)
    axes[0, 1].text(0, 0.5, "Each row: 8 channels with the\nlargest mean activation (after ReLU)",
                    fontsize=8, color=INK2, va="center", transform=axes[0, 1].transAxes)
    for r, name in enumerate(("block1", "block2", "block3"), start=1):
        f = feats[name]
        order = f.mean(dim=(1, 2)).argsort(descending=True)[:n_show]
        for c, ch in enumerate(order):
            axes[r, c].imshow(f[ch], cmap="magma")
            axes[r, c].set_title(f"ch {int(ch)}", fontsize=7, color=INK2)
        axes[r, 0].text(-0.15, 0.5, f"{name}\n{tuple(f.shape[1:])}", fontsize=8, ha="right", va="center",
                        transform=axes[r, 0].transAxes)
    fig.suptitle("Feature maps of model B (seed 0) for one test image", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.01, hspace=0.45, wspace=0.08)
    fig.savefig(os.path.join(FIGS, "extra_feature_maps.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # 3) B가 seed 3개 모두에서 틀린 샘플
    b_preds = [np.array(r["y_pred"]) for r in runs["B"]]
    always = sorted(set.intersection(*[set(np.where(p != y_test)[0]) for p in b_preds]))
    cols = 7
    rows = int(np.ceil(len(always) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.95))
    for a in axes.ravel():
        a.axis("off")
    for a, i in zip(axes.ravel(), always):
        votes = [int(p[i]) for p in b_preds]
        a.imshow(x_test[i], cmap="gray")
        a.set_title(f"#{i} T:{y_test[i]}\nP:{','.join(map(str, votes))}", fontsize=8)
    fig.suptitle(f"Test images misclassified by model B in all 3 seeds ({len(always)} images; P: prediction per seed)",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, hspace=0.55, wspace=0.1)
    fig.savefig(os.path.join(FIGS, "extra_always_wrong.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # 4) 요약 표 (보고서 수치)
    lines = ["| 설정 | 파라미터 | MACs (1장) | Test Acc (%) | 오분류 수 | 학습 시간 (s) |", "|---|---|---|---|---|---|"]
    base_test = [load(os.path.join(RESULTS, "logs", f"test_cnn_28_seed{k}", "test_result_28.json"))["test_acc"] for k in SEEDS]
    base_wrong = [load(os.path.join(RESULTS, "logs", f"test_cnn_28_seed{k}", "test_result_28.json"))["n_wrong"] for k in SEEDS]
    for c in ("base", "A", "B"):
        m = CNN(28) if c in ("base", "A") else CNNPlus(28)
        macs, params = profile(m.eval(), inputs=(torch.randn(1, 784),), verbose=False)
        te = np.array(base_test if c == "base" else [r["test_acc"] for r in runs[c]])
        nw = np.array(base_wrong if c == "base" else [r["n_wrong"] for r in runs[c]])
        tt = np.array([r["train_time_sec"] for r in runs[c]])
        lines.append(f"| {NAMES[c]} | {int(params):,} | {macs / 1e6:.2f} M | {te.mean():.2f} ± {te.std(ddof=1):.2f} "
                     f"({', '.join(f'{v:.2f}' for v in te)}) | {nw.mean():.0f} | {tt.mean():.1f} |")
    for c in ("A", "B"):
        va = np.array([r["val_acc"] for r in runs[c]]).mean(0)
        lines.append(f"\n{c} val_acc by epoch (mean): " + ", ".join(f"{v:.2f}" for v in va))
    # 기준 모델과 B가 공통으로 틀린 샘플
    base_pred = [np.array(load(os.path.join(RESULTS, "logs", f"test_cnn_28_seed{k}", "test_result_28.json"))["y_pred"]) for k in SEEDS]
    b_pred = [np.array(r["y_pred"]) for r in runs["B"]]
    for k in SEEDS:
        wb, wB = set(np.where(base_pred[k] != y_test)[0]), set(np.where(b_pred[k] != y_test)[0])
        lines.append(f"seed{k}: baseline wrong {len(wb)}, B wrong {len(wB)}, both wrong {len(wb & wB)}")
    wrong_all = set.intersection(*[set(np.where(p != y_test)[0]) for p in b_pred])
    lines.append(f"B: wrong in all 3 seeds: {len(wrong_all)} -> {sorted(int(i) for i in wrong_all)}")
    with open(os.path.join(RESULTS, "tables", "extra_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="skip training, only make figures/tables")
    parser.add_argument("--configs", nargs="+", default=["A", "B"], choices=["A", "B"])
    args = parser.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if not args.plot:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for cfg in args.configs:
            for seed in SEEDS:
                run(cfg, seed, device)
    plot()
