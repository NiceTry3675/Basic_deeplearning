"""
보고서용 그림/표 생성 (run_experiments.py 실행 후)
  figures/learning_curves.png        : 입력 조건별 Train/Val Loss·Accuracy (3 seed 평균 ± 표본표준편차)
  figures/per_class_accuracy.png     : 클래스별 테스트 정확도 (28×28 / 7×7 stride / 7×7 avg)
  figures/confusion_errors.png       : 오분류 혼동행렬 (대각 제외, 28×28 vs 7×7 stride)
  figures/input_resolution.png       : 같은 샘플의 28×28 / 7×7(stride) / 7×7(avg) 입력과 예측 결과
  표준편차는 모두 표본 표준편차(ddof=1, n=3)
  figures/test_misclassified_28.png  : 28×28 모델 오분류 샘플 (seed 0)
  tables/summary.md                  : 결과 요약 표
"""
import os
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import load_mnist, downsample
from model_v3 import CNN

RESULTS = os.path.join("..", "results")
LOGS    = os.path.join(RESULTS, "logs")
FIGS    = os.path.join(RESULTS, "figures")
TABLES  = os.path.join(RESULTS, "tables")
MODELS  = os.path.join(RESULTS, "models")
SEEDS   = [0, 1, 2]
# 설정 이름: (in_size, 다운샘플 방식, 표시 이름)
CFGS    = {"28": (28, None, "28×28"), "7": (7, "stride", "7×7 (stride)"), "7avg": (7, "avg", "7×7 (avg)")}
MAIN    = ["28", "7"]   # 혼동행렬 비교 대상

# 색상: 범주형 slot 1(blue), slot 2(orange) / 텍스트는 중립 잉크
C1, C2, C3  = "#2a78d6", "#eb6834", "#1baf7a"
CFG_COLOR   = {"28": C1, "7": C2, "7avg": C3}
INK, INK2   = "#0b0b0b", "#52514e"
GRID        = "#e4e3df"
SEQ_BLUES   = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2, "figure.facecolor": "white", "axes.facecolor": "white",
})

def load_json(path):
    with open(path) as f:
        return json.load(f)

def savefig(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

#%% 데이터 수집
train_logs = {c: [load_json(os.path.join(LOGS, f"train_cnn_{c}_seed{k}.json")) for k in SEEDS] for c in CFGS}
test_logs  = {c: [load_json(os.path.join(LOGS, f"test_cnn_{c}_seed{k}", f"test_result_{CFGS[c][0]}.json"))
                  for k in SEEDS] for c in CFGS}

def stack(cfg, key):
    return np.array([log[key] for log in train_logs[cfg]])

def sd(v, axis=None):
    """표본 표준편차 (seed 3개)"""
    return np.std(v, axis=axis, ddof=1)

#%% 1) 학습 곡선
def plot_learning_curves():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    epochs = np.arange(1, len(train_logs["28"][0]["train_loss"]) + 1)
    for col, size in enumerate(CFGS):
        for row, metric in enumerate(["loss", "acc"]):
            ax = axes[row, col]
            last = {sp: stack(size, f"{sp}_{metric}")[:, -1].mean() for sp in ("train", "val")}
            for split, color, style in (("train", C1, "-"), ("val", C2, "--")):
                other = "val" if split == "train" else "train"
                above = last[split] >= last[other]   # 두 선 중 위에 있는 값은 위로, 아래 값은 아래로 라벨
                v = stack(size, f"{split}_{metric}")
                m, s = v.mean(0), sd(v, 0)
                ax.plot(epochs, m, style, color=color, marker="o", markersize=4,
                        label="Train" if split == "train" else "Validation")
                ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.15, linewidth=0)
                # 마지막 epoch 값 직접 라벨
                fmt = f"{m[-1]:.4f}" if metric == "loss" else f"{m[-1]:.2f}%"
                ax.annotate(fmt, (epochs[-1], m[-1]), xytext=(6, 5 if above else -12),
                            textcoords="offset points", fontsize=8, color=INK2)
            ax.set_title(f"{CFGS[size][2]} — {'Loss' if metric == 'loss' else 'Accuracy (%)'}")
            ax.set_xlabel("Epoch")
            ax.set_xticks(epochs)
            ax.set_xlim(0.5, epochs[-1] + 1.5)
            if row == 0 and col == 0:
                ax.legend(loc="upper right")
    fig.suptitle("Learning curves (mean ± std over 3 seeds)", color=INK, fontsize=12)
    fig.tight_layout()
    savefig(fig, "learning_curves.png")

#%% 2) 클래스별 정확도
def plot_per_class():
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    w = 0.27
    for i, cfg in enumerate(CFGS):
        v = np.array([t["class_acc"] for t in test_logs[cfg]])
        ax.bar(x + (i - 1) * (w + 0.02), v.mean(0), w, yerr=sd(v, 0), color=CFG_COLOR[cfg], label=CFGS[cfg][2],
               error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
    ax.set_xticks(x)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title("Per-class test accuracy (mean ± std over 3 seeds)", pad=28)
    savefig(fig, "per_class_accuracy.png")

#%% 3) 오분류 혼동행렬 (대각 제외, seed 0)
def plot_confusion():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", SEQ_BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, size in zip(axes, MAIN):
        cm = np.array(test_logs[size][0]["confusion_matrix"])
        err = cm.copy()
        np.fill_diagonal(err, 0)
        im = ax.imshow(err, cmap=cmap, vmin=0)
        for i in range(10):
            for j in range(10):
                if i != j and err[i, j] > 0:
                    dark = err[i, j] > err.max() * 0.55
                    ax.text(j, i, err[i, j], ha="center", va="center", fontsize=7,
                            color="white" if dark else INK)
                elif i == j:
                    ax.text(j, i, "·", ha="center", va="center", fontsize=8, color=INK2)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(f"{CFGS[size][2]}: misclassified counts (total {err.sum()})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, "confusion_errors.png")

#%% 4) 입력 해상도 비교 + 단일 샘플 추론
def plot_input_resolution(indices=(0, 12, 247, 8, 18)):
    _, _, _, _, x_test, y_test = load_mnist(use_small=False)
    models = {}
    for cfg, (size, _, _) in CFGS.items():
        m = CNN(size)
        m.load_state_dict(torch.load(os.path.join(MODELS, f"cnn_{cfg}_seed0.pth"), map_location="cpu"))
        m.eval()
        models[cfg] = m
    fig, axes = plt.subplots(3, len(indices), figsize=(2.2 * len(indices), 8.2))
    for c, idx in enumerate(indices):
        img = x_test[idx]
        for r, size in enumerate(CFGS):
            method = CFGS[size][1]
            inp = img if method is None else downsample(img, method)
            with torch.no_grad():
                prob = torch.softmax(models[size](torch.FloatTensor(inp.reshape(1, -1))), 1)[0]
            pred = int(prob.argmax())
            ax = axes[r, c]
            ax.imshow(inp, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            mark = "✓" if pred == y_test[idx] else "✗"
            ax.set_title(f"{mark} pred {pred} ({prob[pred]*100:.0f}%)", fontsize=9)
            if c == 0:
                ax.set_ylabel(CFGS[size][2], fontsize=10)
            if r == 0:
                ax.text(0.5, 1.13, f"#{idx}  true={y_test[idx]}", transform=ax.transAxes,
                        ha="center", fontsize=9, color=INK2)
    fig.suptitle("Same test samples: 28×28 vs 7×7 stride (::4) vs 7×7 4×4-average (seed-0 models)", color=INK, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.89, bottom=0.01, hspace=0.25, wspace=0.08)
    savefig(fig, "input_resolution.png")

#%% 5) 요약 표 (보고서에 쓰는 수치를 모두 여기서 계산)
def write_summary():
    os.makedirs(TABLES, exist_ok=True)
    prof = open(os.path.join(LOGS, "model_profile.txt")).read()
    L = ["표준편차: 표본 표준편차(ddof=1, seed 3개)", "",
         "| 입력 | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Test Acc (%) | 오분류 수 | 학습 시간 (s) |",
         "|---|---|---|---|---|---|---|---|"]
    for c in CFGS:
        tl, ta = stack(c, "train_loss")[:, -1], stack(c, "train_acc")[:, -1]
        vl, va = stack(c, "val_loss")[:, -1], stack(c, "val_acc")[:, -1]
        te = np.array([t["test_acc"] for t in test_logs[c]])
        nw = np.array([t["n_wrong"] for t in test_logs[c]])
        tt = np.array([l["train_time_sec"] for l in train_logs[c]])
        L.append(f"| {CFGS[c][2]} | {tl.mean():.4f} ± {sd(tl):.4f} | {ta.mean():.2f} ± {sd(ta):.2f} | "
                 f"{vl.mean():.4f} ± {sd(vl):.4f} | {va.mean():.2f} ± {sd(va):.2f} | "
                 f"**{te.mean():.2f} ± {sd(te):.2f}** | {nw.mean():.0f} | {tt.mean():.1f} |")
    L += ["", "seed별 테스트 정확도:", ""]
    for c in CFGS:
        L.append(f"- {CFGS[c][2]}: " + ", ".join(f"seed{k}={t['test_acc']:.2f}%" for k, t in zip(SEEDS, test_logs[c])))
    L += ["", "클래스별 테스트 정확도 (평균):", ""]
    for c in CFGS:
        v = np.array([t["class_acc"] for t in test_logs[c]]).mean(0)
        L.append(f"- {CFGS[c][2]}: " + ", ".join(f"{k}:{a:.2f}" for k, a in enumerate(v)))
    L += ["", "가장 많이 헷갈린 쌍 (3 seed 합산, 정답→예측: 횟수):", ""]
    for c in CFGS:
        cm = sum(np.array(t["confusion_matrix"]) for t in test_logs[c])
        np.fill_diagonal(cm, 0)
        order = np.argsort(-cm.ravel(), kind="stable")[:7]
        L.append(f"- {CFGS[c][2]} (총 {cm.sum()}): " +
                 ", ".join(f"{i // 10}→{i % 10}: {cm.flat[i]}" for i in order))
    L += ["", "에폭별 평균 (Train Acc / Val Acc / Train Loss / Val Loss):", ""]
    for c in CFGS:
        L.append(f"- {CFGS[c][2]}:")
        for e, row in enumerate(zip(*(stack(c, k).mean(0) for k in ("train_acc", "val_acc", "train_loss", "val_loss"))), 1):
            L.append(f"  - ep{e:2d}: {row[0]:.2f} / {row[1]:.2f} / {row[2]:.4f} / {row[3]:.4f}")
    L += ["", "모델 프로파일:", "", "```", prof.strip(), "```"]
    with open(os.path.join(TABLES, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    plot_learning_curves()
    plot_per_class()
    plot_confusion()
    plot_input_resolution()
    shutil.copy(os.path.join(LOGS, "test_cnn_28_seed0", "test_misclassified_28.png"), FIGS)
    write_summary()
