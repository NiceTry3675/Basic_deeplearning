"""
보고서 그림 생성 (report_stats.py 실행 후)  →  results/figures/*.png
  fig_arch.png            최종 모델(res_c) 구조
  fig_oof.png             5-fold OOF 지식 증류 개념도
  fig_val_curves.png      검증 정확도 학습 곡선 (OOF KD vs KD 없음, GELU)
  fig_final_test.png      최종 학습(60k) 레시피별 seed 분포와 트랙 A/B
  fig_effects_kd.png      검증 실험 효과 크기 ① KD 설정·teacher·구조 (기준 대비 차이, 95% 신뢰구간)
  fig_effects_train.png   검증 실험 효과 크기 ② 학습 설정·추가 기법
  fig_val_errors.png      검증 오답 구성 (몇 개 seed가 틀렸나 × teacher 앙상블 정오)
  fig_val_hard.png        12개 seed가 모두 틀린 검증 이미지 + 라벨 정제 예시(학습 이미지)
  fig_trackA_errors.png   트랙 A 모델의 테스트 오답 36개
색: 범주형 slot 1 #2a78d6(blue), slot 2 #eb6834(orange) — validate_palette.js 통과. 강조 외 표시는 회색.
"""
import json
import math
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from common import ROOT, load_mnist  # noqa: E402
from models import TokenNet  # noqa: E402

FIGS = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGS, exist_ok=True)
S = json.load(open(os.path.join(ROOT, "results", "report", "stats.json")))
RUNS = os.path.join(ROOT, "results", "runs")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = False          # 순수 fp32 평가 (CPU와 같은 결과)
torch.backends.cuda.matmul.allow_tf32 = False

# ---- 스타일 (dataviz 기준: 얇은 표시, 실선 헤어라인 격자, 텍스트는 잉크 색)
for f in ["/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
_names = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"] if n in _names), "DejaVu Sans")
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAYM = "#a3a19a"          # 강조하지 않는 표시
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
B100, B200, B250, NEUT = "#cde2fb", "#9ec5f4", "#86b6ef", "#f0efec"
plt.rcParams.update({
    "font.family": FONT, "axes.unicode_minus": False, "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def runs_of(group):
    import glob
    import re
    fs = glob.glob(os.path.join(RUNS, f"{group}_s*.json"))
    fs = sorted(fs, key=lambda f: int(re.search(r"_s(\d+)\.json$", f).group(1)))
    return [json.load(open(f)) for f in fs]


def img(x):  # 정규화 해제
    return (x * 0.3081 + 0.1307).clamp(0, 1).squeeze().cpu().numpy()


# =============================================================== 1. 모델 구조
def fig_arch():
    L = {r["token"] + str(i): r for i, r in enumerate(S["complexity"]["layers"])}
    lay = S["complexity"]["layers"]
    blocks = [  # (제목, 부제, 출력 shape, params, 높이, 색)  — 층별 세부는 보고서 표 4
        ("입력", "", "1×28×28", None, 2.5, "white"),
        ("Conv 3×3 ×2", "16채널\nBN · GELU", "16×28×28", lay[0]["params"] + lay[1]["params"], 2.5, B100),
        ("MaxPool", "2×2", "16×14×14", 0, 1.7, NEUT),
        ("잔차 블록", "3×3 → 3×3\n+ 1×1 지름길", "24×14×14", lay[3]["params"], 1.7, B200),
        ("MaxPool", "2×2", "24×7×7", 0, 1.15, NEUT),
        ("잔차 블록", "3×3 → 3×3\n+ 1×1 지름길", "32×7×7", lay[5]["params"], 1.15, B200),
        ("GAP", "+ Linear", "10", lay[7]["params"], 0.7, B100),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    ax.set_axis_off()
    w, gap, x = 1.0, 0.16, 0.0
    for i, (t, sub, shp, p, h, col) in enumerate(blocks):
        y0 = -h / 2
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, ec=AXIS if col == "white" else "none", lw=0.8))
        ax.text(x + w / 2, 0.2 if sub else 0, t, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, -0.02, sub, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.15)
        ax.text(x + w / 2, 1.42, shp, ha="center", va="bottom", fontsize=7.2, color=INK)
        if p is not None:
            ax.text(x + w / 2, -1.42, f"{p:,}" if p else "–", ha="center", va="top", fontsize=7.2,
                    color=INK2 if p else MUTED)
        if i < len(blocks) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.015, 0), (x + w + gap - 0.015, 0), arrowstyle="-|>",
                                         mutation_scale=7, color=MUTED, lw=0.9))
        x += w + gap
    ax.text(-0.05, 1.42, "출력", ha="right", va="bottom", fontsize=7.2, color=MUTED)
    ax.text(-0.05, -1.42, "파라미터", ha="right", va="top", fontsize=7.2, color=MUTED)
    c = S["complexity"]
    ax.text(x - gap, -1.95, f"합계 파라미터 {c['params']:,}개 (한도 30,000의 {c['param_budget_used']:.0%})  ·  "
            f"MACs {c['ptflops_macs']:,} (ptflops 기준, 한도 10M의 {c['mac_budget_used']:.0%})",
            ha="right", va="top", fontsize=7.5, color=INK)
    ax.set_xlim(-0.8, x - gap + 0.05)
    ax.set_ylim(-2.2, 1.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    save(fig, "fig_arch.png")


# =============================================================== 2. OOF 개념도
def fig_oof():
    fig, ax = plt.subplots(figsize=(6.3, 2.75))
    ax.set_axis_off()
    cw, ch = 0.62, 0.36
    x0, y0 = 1.05, 0.0
    for j in range(5):
        ax.text(x0 + j * cw + cw / 2, y0 + 5 * ch + 0.08, f"fold {j + 1}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.text(x0 + 2.5 * cw, y0 + 5 * ch + 0.34, "학습 이미지 60,000장 (fold당 12,000장)", ha="center", va="bottom",
            fontsize=8, color=INK)
    for i in range(5):
        yy = y0 + (4 - i) * ch
        ax.text(x0 - 0.08, yy + ch / 2, f"teacher {i + 1}", ha="right", va="center", fontsize=7.5, color=INK)
        for j in range(5):
            pred = i == j
            ax.add_patch(Rectangle((x0 + j * cw + 0.02, yy + 0.02), cw - 0.04, ch - 0.04,
                                   fc=ORANGE if pred else B100, ec="none"))
            ax.text(x0 + j * cw + cw / 2, yy + ch / 2, "예측" if pred else "학습", ha="center", va="center",
                    fontsize=7, color="white" if pred else INK2, weight="bold" if pred else "normal")
    # 화살표 → soft label 행
    xs = x0 + 5 * cw + 0.25
    ax.add_patch(FancyArrowPatch((x0 + 5 * cw + 0.03, y0 + 2.5 * ch), (xs + 0.02, y0 + 2.5 * ch), arrowstyle="-|>",
                                 mutation_scale=9, color=MUTED, lw=1))
    sw = 0.36
    ax.text(xs + 2.5 * sw, y0 + 3.5 * ch + 0.06, "soft label (OOF)", ha="center", va="bottom", fontsize=8, color=INK, weight="bold")
    for j in range(5):
        ax.add_patch(Rectangle((xs + j * sw + 0.015, y0 + 2 * ch + 0.02), sw - 0.03, ch - 0.04, fc=ORANGE, ec="none"))
        ax.text(xs + j * sw + sw / 2, y0 + 2.5 * ch, f"T{j + 1}", ha="center", va="center", fontsize=7, color="white", weight="bold")
    ax.text(xs + 2.5 * sw, y0 + 2 * ch - 0.06, "각 이미지의 soft label은\n그 이미지를 학습하지 않은\nteacher가 만든 것",
            ha="center", va="top", fontsize=7, color=INK2, linespacing=1.25)
    # → student
    xst = xs + 5 * sw + 0.25
    ax.add_patch(FancyArrowPatch((xs + 5 * sw + 0.03, y0 + 2.5 * ch), (xst + 0.02, y0 + 2.5 * ch), arrowstyle="-|>",
                                 mutation_scale=9, color=MUTED, lw=1))
    bw = 2.05
    ax.add_patch(FancyBboxPatch((xst, y0 + 1.45 * ch), bw, 2.1 * ch, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=B200, ec="none"))
    ax.text(xst + bw / 2, y0 + 3.0 * ch, "student (res_c)", ha="center", va="center", fontsize=8, weight="bold", color=INK)
    ax.text(xst + bw / 2, y0 + 2.5 * ch, "60,000장 전체로 학습", ha="center", va="center", fontsize=7, color=INK2)
    ax.text(xst + bw / 2, y0 + 2.0 * ch, "0.1·CE(라벨) + 0.9·T²·KL, T = 2", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xlim(-0.1, xst + bw + 0.1)
    ax.set_ylim(-0.15, y0 + 5 * ch + 0.65)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    save(fig, "fig_oof.png")


# =============================================================== 3. 학습 곡선 (검증)
def fig_val_curves():
    kd = runs_of("kd_gelu") + runs_of("kd_gelu_rep")
    nk = runs_of("nokd_gelu")
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    for rs, col, lab in [(nk, GRAYM, "KD 없음"), (kd, BLUE, "OOF KD")]:
        a = np.array([r["hist"]["val_acc_ema"] for r in rs])
        ep = np.arange(1, a.shape[1] + 1)
        ax.fill_between(ep, a.min(0), a.max(0), color=col, alpha=0.12, lw=0)
        ax.plot(ep, a.mean(0), color=col, lw=2, solid_capstyle="round")
        yl = 99.805 if lab == "OOF KD" else 99.47
        ax.text(59, yl, f"{lab} — 마지막 epoch {a[:, -1].mean():.3f}% (seed {len(rs)}개)", ha="right", va="center",
                fontsize=8, color=INK)
    ax.set_xlim(1, 60)
    ax.set_ylim(99.0, 99.83)
    ax.set_yticks(np.arange(99.0, 99.81, 0.1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("검증 정확도 (EMA, %)")
    ax.set_title("검증 정확도 학습 곡선 (res_c + GELU, dev split)")
    save(fig, "fig_val_curves.png")


# =============================================================== 4. 최종 테스트 분포
def fig_final_test():
    F = S["final"]
    rows = [("base_relu_nokd", "KD 없음 (ReLU)"), ("oofkd_relu", "OOF KD (ReLU)"), ("oofkd_gelu", "OOF KD + GELU (최종)")]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.grid(axis="y", visible=False)
    dy = 0.085
    for ri, (key, lab) in enumerate(rows):
        v = np.round(np.array(F[key]["test"]), 2)
        names = F[key]["names"]
        col = BLUE if key == "oofkd_gelu" else GRAYM
        seen = {}
        pos = {}
        for nm, t in zip(names, v):
            k = seen.get(t, 0); seen[t] = k + 1
            pos[nm] = (t, ri + 0.1 + k * dy)
            ax.plot(t, ri + 0.1 + k * dy, "o", ms=5.2, color=col, mec="white", mew=0.9, zorder=3)
        m = F[key]["mean"]
        ax.plot([m, m], [ri - 0.02, ri + 0.06], color=INK, lw=1.4, zorder=4)
        ax.text(m, ri - 0.06, f"평균 {m:.3f} ± {F[key]['sd']:.3f}", ha="center", va="top", fontsize=7.5, color=INK)
        if key == "oofkd_gelu":
            for tag, nm, off, ha in [("트랙 A (사전 고정)", F[key]["prereg"]["name"], (-16, 30), "right"),
                                     ("트랙 B (최고 seed)", F[key]["best"]["name"], (8, 26), "left")]:
                x, y = pos[nm]
                ax.plot(x, y, "o", ms=10, mfc="none", mec=INK, mew=1.1, zorder=5)
                ax.annotate(f"{tag} {x:.2f}%", (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=INK,
                            ha=ha, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
        if key == "base_relu_nokd":
            nm = F[key]["best"]["name"]; x, y = pos[nm]
            ax.annotate(f"KD 없이도 seed 하나는 {x:.2f}%", (x, y), xytext=(8, 10), textcoords="offset points",
                        fontsize=7.2, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab}\nn = {F[k]['n']}" for k, lab in rows], fontsize=8.5)
    ax.set_ylim(-0.35, len(rows) - 0.05)
    ax.set_xlim(99.47, 99.72)
    ax.set_xlabel("테스트 정확도 (%) — 점 하나가 모델 하나, 같은 값은 위로 쌓음")
    ax.set_title("최종 학습(학습 60,000장) 결과의 seed 분포")
    save(fig, "fig_final_test.png")


# =============================================================== 5~6. 효과 크기 (forest)
def diff_row(label, key, ref_key, adopted=False):
    a, r = S["ablation"][key], S["ablation"][ref_key]
    se = math.sqrt(a["sd"] ** 2 / a["n"] + r["sd"] ** 2 / r["n"])
    return dict(label=label, d=a["mean"] - r["mean"], ci=1.96 * se, adopted=adopted, n=a["n"])


def forest(sections, name, title, xlim):
    nrows = sum(len(rs) + 1 for _, rs in sections)
    fig, ax = plt.subplots(figsize=(6.3, 0.2 * nrows + 0.9))
    ax.grid(axis="y", visible=False)
    y = 0
    yt, yl = [], []
    for head, rs in sections:
        ax.text(xlim[0] + 0.002, -y, head, ha="left", va="center", fontsize=8.3, weight="bold", color=INK,
                transform=ax.transData, zorder=6, bbox=dict(fc="white", ec="none", pad=1.5))
        y += 1
        for r in rs:
            col = BLUE if r.get("adopted") else GRAYM
            d, ci = r["d"], r["ci"]
            lo, hi = d - ci, d + ci
            clipped = d < xlim[0]
            if clipped:
                ax.annotate("", (xlim[0] + 0.004, -y), (xlim[0] + 0.03, -y),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
                ax.text(xlim[0] + 0.034, -y, f"{d:+.3f} (축 밖)", va="center", fontsize=7, color=INK2)
            else:
                ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [-y, -y], color=col, lw=1.6, solid_capstyle="round")
                ax.plot(d, -y, "o", ms=5.5, color=col, mec="white", mew=1.0, zorder=3)
                ax.text(xlim[1], -y, f"{d:+.3f}", ha="right", va="center", fontsize=7.2,
                        color=INK if r.get("adopted") else INK2)
            yt.append(-y); yl.append(r["label"])
            y += 1
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-y + 0.4, 0.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel("기준 대비 검증 정확도 차이 (%p) — 점: 평균 차이, 선: 95% 신뢰구간")
    ax.set_title(title)
    save(fig, name)


def fig_effects():
    A = "kd_oof_T2"  # ReLU + OOF KD 기준 (12 seed)
    sec_kd = [
        ("① KD 설정  — 기준: KD 없음 (ReLU, 99.597%)", [
            diff_row("in-sample teacher, α0.9 T2", "kd_insample_T2", "nokd_relu"),
            diff_row("in-sample teacher 2개, α0.9 T4", "kd_insample2_T4_a09", "nokd_relu"),
            diff_row("in-sample teacher 2개, α0.5 T4", "kd_insample2_T4_a05", "nokd_relu"),
            diff_row("OOF teacher, α0.9 T1", "kd_oof_T1", "nokd_relu"),
            diff_row("OOF teacher, α0.9 T2  (채택)", "kd_oof_T2", "nokd_relu", adopted=True),
            diff_row("OOF teacher, α0.9 T3", "kd_oof_T3", "nokd_relu"),
            diff_row("OOF teacher, α0.9 T4", "kd_oof_T4", "nokd_relu"),
            diff_row("OOF teacher, α0.5 T1", "kd_oof_a05_T1", "nokd_relu"),
            diff_row("OOF teacher, α0.5 T2", "kd_oof_a05_T2", "nokd_relu"),
            diff_row("OOF teacher, α0.5 T4", "kd_oof_a05_T4", "nokd_relu"),
            diff_row("OOF teacher, α1.0 T2", "kd_oof_a10_T2", "nokd_relu"),
        ]),
        ("② teacher 종류  — 기준: ReLU + OOF KD (99.673%)", [
            diff_row("t_wide OOF teacher", "teacher_wide", A),
            diff_row("t_res + t_wide OOF 평균", "teacher_res_wide", A),
            diff_row("t_res OOF 2세트 평균", "teacher_two_sets", A),
        ]),
        ("③ KD 적용 후 구조 변경  — 기준: ReLU + OOF KD (99.673%)", [
            diff_row("DS-a (depthwise 분리)", "kdarch_ds_a", A),
            diff_row("DS-c", "kdarch_ds_c", A),
            diff_row("DS-k", "kdarch_ds_k", A),
            diff_row("DS-n (5×5 depthwise)", "kdarch_ds_n", A),
            diff_row("DS-a + 2×2 head", "kdarch_ds_a_p2", A),
            diff_row("DS-c + 2×2 head", "kdarch_ds_c_p2", A),
            diff_row("Res-c(R30) + 2×2 head", "kdarch_res_c30_p2", A),
        ]),
    ]
    forest(sec_kd, "fig_effects_kd.png", "검증 실험 효과 크기 ① — KD 설정 · teacher · 구조", (-0.16, 0.16))

    G = "act_gelu"
    soup = S["soup"]
    at = S["at"]

    def soup_row(lab, k):
        s = soup[k]
        return dict(label=lab, d=s["gain"], ci=1.96 * s["gain_se"])

    def at_row(lab, b):
        a, c = at[b], at["0"]
        return dict(label=lab, d=a["mean"] - c["mean"], ci=1.96 * math.sqrt(a["sd"] ** 2 / a["n"] + c["sd"] ** 2 / c["n"]))
    sec_tr = [
        ("④ 학습 설정  — 기준: ReLU + OOF KD (99.673%)", [
            diff_row("lr 0.05", "hp_lr005", A), diff_row("lr 0.2", "hp_lr02", A),
            diff_row("weight decay 1e-4", "hp_wd1e4", A), diff_row("weight decay 1e-3", "hp_wd1e3", A),
            diff_row("batch 64 (lr 0.05)", "hp_bs64", A), diff_row("30 epoch", "hp_ep30", A),
            diff_row("120 epoch", "hp_ep120", A), diff_row("SAM ρ0.05", "hp_sam005", A),
            diff_row("SAM ρ0.15", "hp_sam015", A), diff_row("dropout 0.2", "hp_drop02", A),
            diff_row("선택적 KD α_wrong 0.2", "sel_aw02", A), diff_row("선택적 KD α_wrong 0.5", "sel_aw05", A),
            diff_row("선택적 KD α_wrong 1.0", "sel_aw10", A),
        ]),
        ("⑤ 활성화 함수 · 정밀도  — 기준: ReLU + OOF KD (99.673%)", [
            diff_row("SiLU", "act_silu", A), diff_row("GELU  (채택)", "act_gelu", A, adopted=True),
            diff_row("fp32 학습 (bf16 끔)", "fp32", A),
        ]),
        ("⑥ GELU 이후 추가 기법  — 기준: GELU + OOF KD (99.685%) 또는 짝 비교", [
            diff_row("MaxPool을 R32 뒤로 (7.25M MACs)", "gelu_latepool", G),
            diff_row("첫 conv 5×5", "gelu_k5", G),
            diff_row("구조 재매개변수화 (RepVGG+ACNet)", "gelu_rep_ac", G),
            soup_row("가중치 평균 soup, 분기 8 epoch*", "soup_b8"),
            soup_row("가중치 평균 soup, 분기 15 epoch*", "soup_b15"),
            soup_row("가중치 평균 soup, 분기 15 epoch 재현*", "soup_b15_rep"),
            soup_row("가중치 평균 soup, 분기 15 epoch, 멤버 8*", "soup_b15_m8"),
            soup_row("가중치 평균 soup, 분기 30 epoch*", "soup_b30"),
            soup_row("가중치 평균 soup, 분기 45 epoch*", "soup_b45"),
            soup_row("가중치 평균 soup, 분기 0 epoch*", "soup_b0"),
            at_row("attention transfer β100**", "100"),
            at_row("attention transfer β1000**", "1000"),
        ]),
    ]
    forest(sec_tr, "fig_effects_train.png", "검증 실험 효과 크기 ② — 학습 설정 · 추가 기법", (-0.16, 0.16))


# =============================================================== 7~8. 검증 오답 구성 / 어려운 샘플
@torch.no_grad()
def teacher_val_prob(D):
    p = 0
    for f in range(5):
        ck = torch.load(os.path.join(ROOT, "results", "ckpt", f"t_res_dev_oof5_f{f}.pt"), map_location=dev)
        m = TokenNet(ck["spec"], act=ck["act"]).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        p = p + torch.cat([m(D["x_val"][i:i + 1000]).float().softmax(1) for i in range(0, 5000, 1000)])
    return (p / 5).cpu().numpy()


def fig_val_errors_and_hard():
    D = load_mnist(dev, "dev")
    y = D["y_val"].cpu().numpy()
    lg = torch.cat([torch.load(os.path.join(ROOT, "results", "logits", f"{j}.pt"))["val"].float()
                    for j in ["kd_gelu", "kd_gelu_rep"]], 1)
    pred = lg.argmax(-1).numpy()
    yt = torch.as_tensor(y).long()[:, None, None].expand(-1, lg.shape[1], 1)
    err = (lg.gather(-1, yt).squeeze(-1) < lg.max(-1).values).numpy()   # fp16 logit 동점은 오답으로 세지 않음
    freq = err.sum(1)
    tp = teacher_val_prob(D)
    t_wrong = tp.argmax(1) != y
    # --- 막대: 틀린 seed 수별 이미지 수, teacher 앙상블 정오로 나눔
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ks = np.arange(1, 13)
    a = np.array([((freq == k) & ~t_wrong).sum() for k in ks])
    b = np.array([((freq == k) & t_wrong).sum() for k in ks])
    ax.bar(ks, a, width=0.55, color=BLUE, label="teacher 앙상블은 맞힘", zorder=3)
    ax.bar(ks, b, width=0.55, bottom=a + np.where((a > 0) & (b > 0), 0.12, 0), color=ORANGE,
           label="teacher 앙상블도 틀림", zorder=3)
    for k, tot in zip(ks, a + b):
        if tot:
            ax.text(k, tot + 0.25, str(tot), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(ks)
    ax.set_xlabel("이 이미지를 틀린 seed 수 (12개 중)")
    ax.set_ylabel("검증 이미지 수")
    ax.set_ylim(0, max(a + b) + 2.2)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    E = S["errors"]
    ax.set_title(f"검증 5,000장 중 한 번이라도 틀린 {E['ever_wrong']}장 (GELU + OOF KD, seed 12개)", pad=16)
    save(fig, "fig_val_errors.png")

    # --- 어려운 샘플 그림: 위 = 12개 seed 모두 틀린 검증 이미지, 아래 = 라벨 정제 예시(학습 이미지)
    hard = np.nonzero(freq == 12)[0]
    idx_val = D["x_val"]
    # 라벨 정제 예시: full OOF teacher가 라벨과 다르고 트랙 A 학생이 teacher를 따른 학습 이미지, teacher 확신도 순 상위
    Df = load_mnist(dev, "full")
    T = torch.load(os.path.join(ROOT, "results", "teachers", "t_res_full_oof5.pt"), map_location=dev)["logits"].float()
    tprob2 = T.softmax(1)
    ck = torch.load(os.path.join(ROOT, "results", "ckpt", f"{S['final']['oofkd_gelu']['prereg']['name']}.pt"), map_location=dev)
    m = TokenNet(ck["spec"], act=ck["act"]).to(dev).eval(); m.load_state_dict(ck["state_dict"])
    dis = (T.argmax(1) != Df["y_tr"]).nonzero().flatten()
    with torch.no_grad():
        sp = m(Df["x_tr"][dis]).argmax(1)
    follow = dis[sp == T.argmax(1)[dis]]
    conf = tprob2[follow].max(1).values
    sel = follow[conf.argsort(descending=True)[:9]].cpu().numpy()
    n_cols = 9
    fig, axes = plt.subplots(2, n_cols, figsize=(6.3, 2.75))
    for j in range(n_cols):
        for i in range(2):
            axes[i, j].set_axis_off()
        if j < len(hard):
            k = hard[j]
            maj = np.bincount(pred[k], minlength=10).argmax()
            axes[0, j].imshow(img(idx_val[k]), cmap="gray_r", vmin=0, vmax=1)
            axes[0, j].set_title(f"{y[k]} → {maj}\np={tp[k, y[k]]:.3f}", fontsize=7.5, color=INK,
                                 weight="normal", loc="center")
        k = sel[j]
        axes[1, j].imshow(img(Df["x_tr"][k]), cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].set_title(f"{int(Df['y_tr'][k])} → {int(T[k].argmax())}\np={float(tprob2[k].max()):.2f}",
                             fontsize=7.5, color=INK, weight="normal", loc="center")
    fig.text(0.01, 0.985, "(a) 12개 seed가 모두 틀린 검증 이미지 — 라벨 → student 다수 예측, p: teacher 앙상블이 라벨에 준 확률",
             fontsize=7.5, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.485, "(b) OOF teacher가 라벨과 다르게 본 학습 이미지 — 라벨 → teacher 예측, p: teacher 확신도",
             fontsize=7.5, color=INK, weight="bold", va="top")
    fig.subplots_adjust(top=0.84, bottom=0.02, hspace=0.75, wspace=0.12, left=0.01, right=0.99)
    save(fig, "fig_val_hard.png")


# =============================================================== 9. 트랙 A 테스트 오답
def fig_trackA_errors():
    D = load_mnist(dev, "full")
    tA = S["track_eval"]["A"]
    wrong = tA["wrong_idx"]
    # 24개 최종 모델 중 몇 개가 틀렸나
    cnt = np.zeros(10000, dtype=int)
    preds = {}
    for nm in S["final"]["oofkd_gelu"]["names"]:
        ck = torch.load(os.path.join(ROOT, "results", "ckpt", f"{nm}.pt"), map_location=dev)
        m = TokenNet(ck["spec"], act=ck["act"]).to(dev).eval(); m.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            p = torch.cat([m(D["x_te"][i:i + 2000]).argmax(1) for i in range(0, 10000, 2000)]).cpu().numpy()
        preds[nm] = p
        cnt += p != D["y_te"].cpu().numpy()
    pa = preds[tA["name"]]
    yt = D["y_te"].cpu().numpy()
    order = sorted(wrong, key=lambda k: (-cnt[k], k))
    n_cols = 9
    n_rows = math.ceil(len(order) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.3, 0.92 * n_rows + 0.3))
    for ax in axes.flat:
        ax.set_axis_off()
    for i, k in enumerate(order):
        ax = axes.flat[i]
        ax.imshow(img(D["x_te"][k]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{yt[k]}→{pa[k]}  {cnt[k]}/24", fontsize=7.5, color=INK if cnt[k] >= 12 else INK2,
                     weight="bold" if cnt[k] >= 12 else "normal", loc="center")
    fig.suptitle(f"트랙 A 모델의 테스트 오답 {len(order)}장 — 정답→예측, 24개 모델 중 틀린 모델 수 (12개 이상 굵게)",
                 fontsize=8, color=INK, x=0.01, ha="left", y=1.0)
    fig.subplots_adjust(top=0.9, bottom=0.01, hspace=0.45, wspace=0.08, left=0.01, right=0.99)
    save(fig, "fig_trackA_errors.png")
    return {int(k): int(cnt[k]) for k in order}


if __name__ == "__main__":
    fig_arch()
    fig_oof()
    fig_val_curves()
    fig_final_test()
    fig_effects()
    fig_val_errors_and_hard()
    c = fig_trackA_errors()
    vals = np.array(list(c.values()))
    print("트랙 A 오답의 24모델 오답 수 분포: >=12:", int((vals >= 12).sum()), " <=3:", int((vals <= 3).sum()), " 1(트랙 A만):", int((vals == 1).sum()))
