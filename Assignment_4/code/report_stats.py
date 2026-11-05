"""
보고서용 수치 집계 — 보고서의 모든 숫자는 이 스크립트의 출력(results/report/stats.json)에서 가져온다.
  python report_stats.py            (GPU가 있으면 사용, 약 1분)

집계 내용
  complexity : 최종 모델 층별 파라미터·MACs, thop/ptflops 측정값, teacher 크기
  final      : full split(60k) 최종 학습 레시피별 test 정확도 (seed 분포, 트랙 A 사전 고정 모델, 트랙 B 최고 seed)
  track_eval : 트랙 A/B 체크포인트를 다시 불러 test 평가 (클래스별 정확도, 오답 목록, 혼동 쌍)
  teacher    : fold teacher 정확도, OOF teacher와 학습 라벨의 불일치, 학생이 teacher를 따르는 비율(라벨 정제)
  ablation   : dev split(55k/5k) 검증 실험 그룹별 평균 ± 표준편차(표본, ddof=1)
  soup / at / ensemble_pairs : 추가 실험 세부
  errors     : 검증 오답 구성 분석 (seed 12개, fold teacher 앙상블)
  noise      : seed 노이즈 크기와 채택 기준의 오채택 확률(정규 근사)
표준편차는 모두 표본 표준편차(ddof=1)이고, 정확도 단위는 %이다.
"""
import glob
import json
import math
import os
import re
from collections import Counter

import numpy as np
import torch

from common import ROOT, count_complexity, load_mnist
from models import STUDENTS, TEACHERS, TokenNet

RUNS = os.path.join(ROOT, "results", "runs")
OUT_DIR = os.path.join(ROOT, "results", "report")
os.makedirs(OUT_DIR, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = False          # GPU에서도 순수 fp32로 평가 (CPU 재계산과 같은 값)
torch.backends.cuda.matmul.allow_tf32 = False
S = {}


def load_run(name):
    with open(os.path.join(RUNS, f"{name}.json")) as f:
        return json.load(f)


def runs_of(group):
    """group_s0, group_s1, ... (seed 순서)"""
    fs = glob.glob(os.path.join(RUNS, f"{group}_s*.json"))
    fs = sorted(fs, key=lambda f: int(re.search(r"_s(\d+)\.json$", f).group(1)))
    return [json.load(open(f)) for f in fs]


def stat(v):
    v = np.asarray(v, dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return dict(n=int(len(v)), mean=float(v.mean()), sd=sd, se=sd / math.sqrt(len(v)) if len(v) > 1 else float("nan"),
                min=float(v.min()), max=float(v.max()))


def r2(x):
    return float(np.round(x, 2))


# ---------------------------------------------------------------- 1. 복잡도
def layer_table(spec, act="gelu"):
    m = TokenNet(spec, act=act).eval()
    rows, x = [], torch.zeros(1, 1, 28, 28)
    names = spec.split()
    # TokenNet.net의 각 모듈과 토큰을 대응 (G는 [Pool, Flatten] 2개 모듈)
    mods, ti = [], 0
    for tok in names:
        if tok == "G" or tok.startswith("P"):
            mods.append((tok, [m.net[ti], m.net[ti + 1]])); ti += 2
        else:
            mods.append((tok, [m.net[ti]])); ti += 1
    with torch.no_grad():
        for tok, ms in mods:
            cin_shape = tuple(x.shape[1:])
            conv_macs = 0
            hooks = []

            def hook(mod, inp, out):
                nonlocal conv_macs
                if isinstance(mod, torch.nn.Conv2d):
                    kh, kw = mod.kernel_size
                    conv_macs += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * kh * kw
                elif isinstance(mod, torch.nn.Linear):
                    conv_macs += mod.in_features * mod.out_features
            for mm in ms:
                for sub in mm.modules():
                    hooks.append(sub.register_forward_hook(hook))
            for mm in ms:
                x = mm(x)
            for h in hooks:
                h.remove()
            params = sum(p.numel() for mm in ms for p in mm.parameters())
            rows.append(dict(token=tok, in_shape=cin_shape, out_shape=tuple(x.shape[1:]), params=int(params),
                             conv_linear_macs=int(conv_macs)))
    return rows


final_spec = STUDENTS["res_c"]
cx = count_complexity(TokenNet(final_spec, act="gelu"))
lt = layer_table(final_spec)
S["complexity"] = dict(spec=final_spec, act="gelu", params=cx["max_params"], thop_macs=cx["thop_macs"],
                       ptflops_macs=cx["ptflops_macs"], max_macs=cx["max_macs"], ok=cx["ok"],
                       conv_linear_macs=sum(r["conv_linear_macs"] for r in lt), layers=lt,
                       param_budget_used=cx["max_params"] / 30000, mac_budget_used=cx["max_macs"] / 1e7)
tcx = count_complexity(TokenNet(TEACHERS["t_res"], act="relu"))
S["complexity"]["teacher"] = dict(spec=TEACHERS["t_res"], params=tcx["max_params"], max_macs=tcx["max_macs"],
                                  ratio_params=tcx["max_params"] / cx["max_params"])

# ---------------------------------------------------------------- 2. 최종 결과 (full split)
RECIPES = {
    "base_relu_nokd": ("fin_base_res_c_j", "KD 없음 (ReLU, LS 0.1)"),
    "oofkd_relu": ("fin_oof_res_c_j", "OOF KD (ReLU)"),
    "oofkd_gelu": ("fin_oof_res_c_gelu_j", "OOF KD (GELU) — 최종 레시피"),
}
final = {}
for key, (prefix, label) in RECIPES.items():
    rs = []
    for j in range(4):
        rs += [r for r in runs_of(f"{prefix}{j}")]
    rs = [r for r in rs if r.get("final_test_ema") is not None]
    te = [r["final_test_ema"] for r in rs]
    best = rs[int(np.argmax(te))]
    hist_max = [max(r["hist"]["test_acc_ema"]) for r in rs]
    final[key] = dict(label=label, **stat(te), names=[r["name"] for r in rs], test=[r2(t) for t in te],
                      dist={f"{k:.2f}": v for k, v in sorted(Counter(np.round(te, 2)).items())},
                      prereg=dict(name=rs[0]["name"], test=r2(rs[0]["final_test_ema"])),
                      best=dict(name=best["name"], test=r2(best["final_test_ema"])),
                      raw_mean=float(np.mean([r["final_test"] for r in rs])),
                      epoch_pick_mean=float(np.mean(hist_max)), epoch_pick_max=r2(max(hist_max)),
                      seeds=sorted({r["args"]["seed"] for r in rs}), time_per_model=float(np.mean([r["time"] for r in rs])))
S["final"] = final

# ---------------------------------------------------------------- 3. 트랙 A/B 체크포인트 재평가
D_full = load_mnist(dev, "full")
x_te, y_te = D_full["x_te"], D_full["y_te"]


def load_ckpt(name):
    ck = torch.load(os.path.join(ROOT, "results", "ckpt", f"{name}.pt"), map_location=dev)
    m = TokenNet(ck["spec"], act=ck["act"]).to(dev).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


@torch.no_grad()
def logits_of(m, x, bs=2000):
    return torch.cat([m(x[i:i + bs]).float() for i in range(0, len(x), bs)])


track = {}
for tk, name in [("A", final["oofkd_gelu"]["prereg"]["name"]), ("B", final["oofkd_gelu"]["best"]["name"])]:
    m, ck = load_ckpt(name)
    z = logits_of(m, x_te)
    pred = z.argmax(1)
    wrong = (pred != y_te).nonzero().flatten().cpu().tolist()
    yv, pv = y_te.cpu().numpy(), pred.cpu().numpy()
    per_class = {int(c): float((pv[yv == c] == c).mean() * 100) for c in range(10)}
    pairs = Counter((int(yv[i]), int(pv[i])) for i in wrong)
    ccx = count_complexity(m)
    track[tk] = dict(name=name, test=float((pred == y_te).float().mean() * 100), n_err=len(wrong), wrong_idx=wrong,
                     per_class=per_class, top_pairs=[[f"{a}->{b}", n] for (a, b), n in pairs.most_common(8)],
                     params=ccx["max_params"], thop_macs=ccx["thop_macs"], ptflops_macs=ccx["ptflops_macs"],
                     seed_arg=ck["args"]["seed"], k=ck["k"])
both = set(track["A"]["wrong_idx"]) & set(track["B"]["wrong_idx"])
S["track_eval"] = dict(A=track["A"], B=track["B"], common_errors=len(both))

# 24개 최종 모델이 각 test 이미지를 몇 번 틀리는가 (오답 공유 정도)
cnt = torch.zeros(len(y_te), device=dev)
for nm in final["oofkd_gelu"]["names"]:
    m, _ = load_ckpt(nm)
    cnt += (logits_of(m, x_te).argmax(1) != y_te).float()
cnt = cnt.cpu().numpy()
S["track_eval"]["test_err_freq_24"] = dict(always=int((cnt == 24).sum()), ge_half=int((cnt >= 12).sum()),
                                           ever=int((cnt > 0).sum()), mean_err=float(cnt.sum() / 24))
wA = np.array(track["A"]["wrong_idx"])
S["track_eval"]["trackA_err_share"] = dict(n=len(wA), ge_half=int((cnt[wA] >= 12).sum()), all24=int((cnt[wA] == 24).sum()),
                                           only_A=int((cnt[wA] == 1).sum()))

# ---------------------------------------------------------------- 4. teacher / 라벨 정제
tch = {}
dev_folds = [load_run(f"t_res_dev_oof5_f{f}") for f in range(5)]
full_folds = [load_run(f"t_res_full_oof5_f{f}") for f in range(5)]
tch["dev_fold_val_ema"] = [r2(r["final_val_ema"]) for r in dev_folds]
tch["dev_fold_val_mean"] = float(np.mean([r["final_val_ema"] for r in dev_folds]))
tch["full_fold_test_ema"] = [r2(r["final_test_ema"]) for r in full_folds]
tch["full_fold_test_mean"] = float(np.mean([r["final_test_ema"] for r in full_folds]))
tch["epochs"] = dev_folds[0]["args"]["epochs"]
tch["time_per_fold"] = float(np.mean([r["time"] for r in full_folds]))
tch["in_sample_val_ema"] = r2(load_run("t_res_dev_clean_s0")["final_val_ema"])
y_tr_full = D_full["y_tr"]
T_full = torch.load(os.path.join(ROOT, "results", "teachers", "t_res_full_oof5.pt"), map_location=dev)["logits"].float()
t_arg = T_full.argmax(1)
dis = (t_arg != y_tr_full)
tch["full_oof_agree"] = float((~dis).float().mean() * 100)
tch["full_oof_disagree_n"] = int(dis.sum())
# fold별 held-out이 학습셋 전체를 정확히 한 번씩 덮는지
cover = torch.zeros(60000, dtype=torch.long)
for f in range(5):
    cover[torch.load(os.path.join(ROOT, "results", "teachers", f"t_res_full_oof5_f{f}.pt"))["heldout_idx"].long()] += 1
tch["full_oof_cover_exactly_once"] = bool((cover == 1).all())
T_dev = torch.load(os.path.join(ROOT, "results", "teachers", "t_res_dev_oof5.pt"), map_location=dev)["logits"].float()
D_dev = load_mnist(dev, "dev")
tch["dev_oof_agree"] = float((T_dev[D_dev["idx_tr"]].argmax(1) == D_dev["y_tr"]).float().mean() * 100)
# 최종 학생 24개가 teacher-라벨 불일치 샘플에서 누구를 따르는가
x_tr_full = D_full["x_tr"]
dis_idx = dis.nonzero().flatten()
follow_t, follow_y = [], []
for nm in final["oofkd_gelu"]["names"]:
    m, _ = load_ckpt(nm)
    p = logits_of(m, x_tr_full[dis_idx]).argmax(1)
    follow_t.append(int((p == t_arg[dis_idx]).sum())); follow_y.append(int((p == y_tr_full[dis_idx]).sum()))
tch["students_follow_teacher"] = stat(follow_t)
tch["students_follow_label"] = stat(follow_y)
tch["trackA_follow_teacher"] = follow_t[0]
tch["trackA_follow_label"] = follow_y[0]
# 같은 샘플에서 KD 없는 기준 모델(라벨만 학습)은?
fy, ft = [], []
for nm in final["base_relu_nokd"]["names"]:
    m, _ = load_ckpt(nm)
    p = logits_of(m, x_tr_full[dis_idx]).argmax(1)
    fy.append(int((p == y_tr_full[dis_idx]).sum())); ft.append(int((p == t_arg[dis_idx]).sum()))
tch["nokd_follow_label"] = stat(fy)
tch["nokd_follow_teacher"] = stat(ft)
# in-sample teacher(dev 학습셋 55,000장 전체로 학습)가 자기 학습 이미지의 라벨과 얼마나 일치하나 (= 라벨 암기)
T_in = torch.load(os.path.join(ROOT, "results", "teachers", "t_res_dev_clean_s0.pt"), map_location=dev)["logits"].float()
tch["insample_train_agree"] = float((T_in[D_dev["idx_tr"]].argmax(1) == D_dev["y_tr"]).float().mean() * 100)
S["teacher"] = tch

# ---------------------------------------------------------------- 5. dev split 검증 실험
GROUPS = {
    # key: (job 목록, 설명)
    # --- 구조 탐색 (KD 없음, ReLU, LS 0.1)
    "scr_vgg_a": (["scr1_vgg_a"], "VGG-a"), "scr_vgg_f": (["scr1_vgg_f"], "VGG-f"), "scr_vgg_g": (["scr1_vgg_g"], "VGG-g"),
    "scr_vgg_o": (["scr1_vgg_o"], "VGG-o"), "scr_res_a": (["scr1_res_a"], "Res-a"), "scr_res_b": (["scr1_res_b"], "Res-b"),
    "scr_res_c": (["scr1_res_c"], "Res-c (최종 구조)"), "scr_res_d": (["scr1_res_d"], "Res-d"),
    "scr_sh_a": (["scr1_sh_a"], "가중치 공유-a"), "scr_sh_a3": (["scr1_sh_a3"], "가중치 공유-a3"), "scr_sh_b": (["scr1_sh_b"], "가중치 공유-b"),
    "scr_sh_c": (["scr1_sh_c"], "가중치 공유-c"), "scr_sh_c2": (["scr1_sh_c2"], "가중치 공유-c2"), "scr_ds_a": (["scr1_ds_a"], "DS-a"),
    # --- KD 설정 (ReLU res_c, LS 0)
    "nokd_relu": (["nokd_ls0_res_c"], "KD 없음 (LS 0)"),
    "kd_insample_T2": (["kd_cR_a0.9_T2"], "in-sample teacher, α0.9 T2"),
    "kd_insample2_T4_a09": (["kd_c2_a0.9_T4"], "in-sample teacher 2개 앙상블, α0.9 T4"),
    "kd_insample2_T4_a05": (["kd_c2_a0.5_T4"], "in-sample teacher 2개 앙상블, α0.5 T4"),
    "kd_oof_T1": (["kd_oof_a0.9_T1"], "OOF, α0.9 T1"), "kd_oof_T2": (["kd_oof_a0.9_T2", "kd_oof_a0.9_T2_rep"], "OOF, α0.9 T2 (ReLU 기준)"),
    "kd_oof_T3": (["kd_oof_a0.9_T3"], "OOF, α0.9 T3"), "kd_oof_T4": (["kd_oof_a0.9_T4"], "OOF, α0.9 T4"),
    "kd_oof_a05_T2": (["kd_oof_a0.5_T2"], "OOF, α0.5 T2"), "kd_oof_a10_T2": (["kd_oof_a1.0_T2"], "OOF, α1.0 T2"),
    "kd_oof_a05_T1": (["kd_oof_a0.5_T1"], "OOF, α0.5 T1"), "kd_oof_a05_T4": (["kd_oof_a0.5_T4"], "OOF, α0.5 T4"),
    "kd_oof_T2_first": (["kd_oof_a0.9_T2"], "OOF α0.9 T2 1차"), "kd_oof_T2_rep": (["kd_oof_a0.9_T2_rep"], "OOF α0.9 T2 재현"),
    # --- teacher 종류
    "teacher_wide": (["kd_oofW_a0.9_T2"], "t_wide OOF teacher"), "teacher_res_wide": (["kd_oofRW_a0.9_T2"], "t_res + t_wide OOF 평균"),
    "teacher_two_sets": (["kd_oofAB_a0.9_T2"], "t_res OOF 2세트(다른 fold 분할) 평균"),
    # --- KD 위 구조 변경 (ReLU)
    "kdarch_ds_a": (["kdo_ds_a"], "DS-a"), "kdarch_ds_c": (["kdo_ds_c"], "DS-c"), "kdarch_ds_k": (["kdo_ds_k"], "DS-k"),
    "kdarch_ds_n": (["kdo_ds_n"], "DS-n (5×5)"), "kdarch_ds_a_p2": (["kdo_ds_a_p2"], "DS-a + 2×2 head"),
    "kdarch_ds_c_p2": (["kdo_ds_c_p2"], "DS-c + 2×2 head"), "kdarch_res_c30_p2": (["kdo_res_c30_p2"], "Res-c(R30) + 2×2 head"),
    "kdarch_e_res": (["kdo_e_res"], "앙상블 멤버 e_res"), "kdarch_e_ds2": (["kdo_e_ds2"], "앙상블 멤버 e_ds2"),
    "kdarch_e_vgg": (["kdo_e_vgg"], "앙상블 멤버 e_vgg"), "nokd_ds_a": (["nokd_ls0_ds_a"], "DS-a KD 없음"),
    # --- 학습 하이퍼파라미터 (ReLU + OOF KD)
    "hp_lr005": (["hpk_lr0.05"], "lr 0.05"), "hp_lr02": (["hpk_lr0.2"], "lr 0.2"), "hp_wd1e4": (["hpk_wd1e-4"], "wd 1e-4"),
    "hp_wd1e3": (["hpk_wd1e-3"], "wd 1e-3"), "hp_bs64": (["hpk_bs64"], "batch 64 (lr 0.05)"), "hp_ep30": (["hpk_ep30"], "30 epoch"),
    "hp_ep120": (["hpk_ep120"], "120 epoch"), "hp_sam005": (["hpk_sam0.05"], "SAM ρ0.05"), "hp_sam015": (["hpk_sam0.15"], "SAM ρ0.15"),
    "hp_drop02": (["hpk_drop0.2"], "dropout 0.2"),
    "sel_aw02": (["kd_sel_aw0.2"], "선택적 KD α_wrong 0.2"), "sel_aw05": (["kd_sel_aw0.5"], "선택적 KD α_wrong 0.5"),
    "sel_aw10": (["kd_sel_aw1.0"], "선택적 KD α_wrong 1.0"),
    # --- 활성화/정밀도 (OOF KD)
    "act_silu": (["kd_silu"], "SiLU"), "act_gelu": (["kd_gelu", "kd_gelu_rep"], "GELU (최종)"),
    "act_gelu_first": (["kd_gelu"], "GELU 1차"), "act_gelu_rep": (["kd_gelu_rep"], "GELU 재현"), "fp32": (["kd_fp32"], "fp32 (bf16 끔)"),
    # --- GELU 기준 추가 실험
    "nokd_gelu": (["nokd_gelu"], "GELU, KD 없음"), "gelu_latepool": (["kd_gelu_lp"], "MaxPool을 R32 뒤로"),
    "gelu_k5": (["kd_gelu_k5"], "첫 conv 5×5"), "gelu_rep_ac": (["rep_ac"], "구조 재매개변수화 (RepVGG+ACNet)"),
}
abl = {}
for key, (jobs, label) in GROUPS.items():
    rs = sum([runs_of(j) for j in jobs], [])
    v = [r["final_val_ema"] for r in rs]
    r0 = rs[0]
    abl[key] = dict(label=label, jobs=jobs, spec=r0["spec"], params=r0["complexity"]["max_params"],
                    macs=r0["complexity"]["max_macs"], act=r0["args"].get("act", "relu"), **stat(v))
S["ablation"] = abl

# AT: 한 작업 안에서 β별 6개씩
at_rs = runs_of("at_scr")
S["at"] = {f"{b:g}": stat([r["final_val_ema"] for r in at_rs if r["at_beta"] == b]) for b in sorted({r["at_beta"] for r in at_rs})}
S["at"]["dist_last"] = {f"{b:g}": float(np.mean([r["hist"]["at_dist"][-1] for r in at_rs if r["at_beta"] == b]))
                        for b in sorted({r["at_beta"] for r in at_rs})}

# soup: 그룹별 (멤버 평균, soup, 출력 앙상블)
soup = {}
for name in ["soup_b0", "soup_b8", "soup_b15", "soup_b15_rep", "soup_b15_m8", "soup_b30", "soup_b45"]:
    rs = runs_of(name)
    G = max(r["soup_group"] for r in rs) + 1
    M = len(rs) // G
    mem = [np.mean([rs[k]["final_val_ema"] for k in range(g * M, (g + 1) * M)]) for g in range(G)]
    sp = [rs[g * M]["soup_val_bnrc"] for g in range(G)]
    en = [rs[g * M]["ens_val"] for g in range(G)]
    gain = np.array(sp) - np.array(mem)
    soup[name] = dict(branch_ep=rs[0]["args"]["branch_ep"], G=G, M=M, member=float(np.mean(mem)), soup=float(np.mean(sp)),
                      ens=float(np.mean(en)), gain=float(gain.mean()), gain_se=float(gain.std(ddof=1) / math.sqrt(G)),
                      gains=[float(np.round(g, 3)) for g in gain])
pool = np.concatenate([np.array([rs_["soup_val_bnrc"] for rs_ in []])] + [np.array(soup[n]["gains"]) for n in
                                                                            ["soup_b8", "soup_b15", "soup_b15_rep", "soup_b15_m8", "soup_b30"]])
soup["pooled_b8_b30"] = dict(n_groups=int(len(pool)), gain=float(pool.mean()), se=float(pool.std(ddof=1) / math.sqrt(len(pool))))
S["soup"] = soup

# 예산 내 2-모델 앙상블 (e_res × e_ds2, val logits 교차 조합)
y_val = D_dev["y_val"].cpu().numpy()
la = torch.load(os.path.join(ROOT, "results", "logits", "kdo_e_res.pt"))["val"].float().softmax(-1).numpy()
lb = torch.load(os.path.join(ROOT, "results", "logits", "kdo_e_ds2.pt"))["val"].float().softmax(-1).numpy()
pairs = [((la[:, i] + lb[:, j]) / 2).argmax(-1) for i in range(la.shape[1]) for j in range(lb.shape[1])]
pair_acc = [(p == y_val).mean() * 100 for p in pairs]


class _Ens(torch.nn.Module):
    def __init__(self, *ms):
        super().__init__(); self.ms = torch.nn.ModuleList(ms)

    def forward(self, x):
        return torch.stack([m(x).softmax(1) for m in self.ms]).mean(0)


ecx = count_complexity(_Ens(TokenNet(STUDENTS["e_res"]), TokenNet(STUDENTS["e_ds2"])))
S["ensemble_pairs"] = dict(**stat(pair_acc), params=ecx["max_params"], macs=ecx["max_macs"])

# ---------------------------------------------------------------- 6. 검증 오답 구성 분석
def val_logits(job):
    return torch.load(os.path.join(ROOT, "results", "logits", f"{job}.pt"))["val"].float()


def wrong_mask(L, y):
    """(N,K,10) logit → (N,K) 오답 여부. 저장된 logit이 fp16이라 정답과 다른 클래스가 정확히 같은 값(동점)일 수 있으므로,
    정답 logit이 최댓값보다 엄격히 작을 때만 오답으로 센다 (fp32 학습 기록의 정확도와 일치)"""
    yt = torch.as_tensor(y).long()[:, None, None].expand(-1, L.shape[1], 1)
    return (L.gather(-1, yt).squeeze(-1) < L.max(-1).values).numpy()


lg = torch.cat([val_logits("kd_gelu"), val_logits("kd_gelu_rep")], 1)  # (5000,12,10)
err = wrong_mask(lg, y_val)  # (5000,12)
freq = err.sum(1)
# fold teacher 5개 (dev) 를 val에 적용 — val은 어느 fold teacher의 학습에도 쓰이지 않음
tprob = 0
for f in range(5):
    m, _ = load_ckpt(f"t_res_dev_oof5_f{f}")
    tprob = tprob + logits_of(m, D_dev["x_val"]).softmax(1).cpu()
tprob = (tprob / 5).numpy()
t_wrong = tprob.argmax(1) != y_val
nokd = val_logits("nokd_gelu")
nokd_err = wrong_mask(nokd, y_val)
e6 = [((lg[:, 6 * b:6 * b + 6].softmax(-1).mean(1)).argmax(-1).numpy() == y_val).mean() * 100 for b in range(2)]
e12 = (lg.softmax(-1).mean(1).argmax(-1).numpy() == y_val).mean() * 100
always = freq == 12
S["errors"] = dict(
    n_val=int(len(y_val)), seeds=12, mean_err_per_model=float(err.sum(0).mean()), sd_err_per_model=float(err.sum(0).std(ddof=1)),
    always_wrong=int(always.sum()), ever_wrong=int((freq > 0).sum()), freq_hist={int(k): int(v) for k, v in zip(*np.unique(freq, return_counts=True))},
    teacher_ens_acc=float((~t_wrong).mean() * 100), teacher_ens_err=int(t_wrong.sum()),
    always_wrong_teacher_wrong=int((always & t_wrong).sum()),
    always_wrong_teacher_ptrue_lt_005=int((always & (tprob[np.arange(len(y_val)), y_val] < 0.05)).sum()),
    exp_err_on_teacher_wrong=float(err[t_wrong].sum() / 12), exp_err_on_teacher_right=float(err[~t_wrong].sum() / 12),
    ens6_acc=[float(e) for e in e6], ens12_acc=float(e12), ens12_err=int(round((100 - e12) / 100 * len(y_val))),
    nokd_mean_err_per_model=float(nokd_err.sum(0).mean()),
    nokd_err_on_teacher_wrong=float(nokd_err[t_wrong].sum() / nokd_err.shape[1]),
    nokd_err_on_teacher_right=float(nokd_err[~t_wrong].sum() / nokd_err.shape[1]),
    ens12_teacher_overlap=int(((lg.softmax(-1).mean(1).argmax(-1).numpy() != y_val) & t_wrong).sum()),
    always_wrong_p_label=[float(np.round(tprob[i, y_val[i]], 3)) for i in np.nonzero(always)[0]],
    always_wrong_idx=[int(i) for i in np.nonzero(always)[0]],
)

# ---------------------------------------------------------------- 7. 노이즈와 채택 기준
gv = np.array([r["final_val_ema"] for r in runs_of("kd_gelu") + runs_of("kd_gelu_rep")])
sd_seed = float(gv.std(ddof=1))
se6, se12 = sd_seed / math.sqrt(6), sd_seed / math.sqrt(12)
phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
base = float(gv.mean())
p1 = 1 - phi((99.700 - base) / se6)      # 효과 없는 후보가 1차 선별(≥99.700)을 통과할 확률
p2 = 1 - phi((99.700 - base) / se12)     # 재현(12 seed ≥ 99.700)도 통과할 확률
gt = np.array(final["oofkd_gelu"]["test"])
S["noise"] = dict(val_sd_per_seed=sd_seed, val_se6=se6, val_se12=se12, baseline_val=base,
                  p_screen_null=p1, p_replicate_null=p2, p_false_adopt=p1 * p2,
                  test_sd_per_seed=float(gt.std(ddof=1)), test_sd_images=float(gt.std(ddof=1) * 100),
                  one_val_image=100 / 5000, one_test_image=100 / 10000)

# ---------------------------------------------------------------- 8. 실험 규모
all_runs = [json.load(open(f)) for f in glob.glob(os.path.join(RUNS, "*.json"))]
stud = [r for r in all_runs if "hist" in r and not r["name"].startswith("t_")]
teach = [r for r in all_runs if "hist" in r and r["name"].startswith("t_")]
S["scale"] = dict(student_models=len(stud), teacher_models=len(teach),
                  groups=len({re.sub(r"_s\d+$", "", r["name"]) for r in stud}),
                  gpu_hours_students=float(sum(r["time"] for r in stud) / 3600),
                  gpu_hours_teachers=float(sum(r["time"] for r in teach) / 3600))

with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
    json.dump(S, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 요약 출력
c = S["complexity"]
print(f"[복잡도] params {c['params']:,} ({c['param_budget_used']:.1%}) thop {c['thop_macs']:,} ptflops {c['ptflops_macs']:,} "
      f"({c['mac_budget_used']:.1%}) conv+linear {c['conv_linear_macs']:,} | teacher {c['teacher']['params']:,} "
      f"(×{c['teacher']['ratio_params']:.0f}) MACs {c['teacher']['max_macs']:,}")
for r in c["layers"]:
    print(f"   {r['token']:6s} {str(r['in_shape']):14s} -> {str(r['out_shape']):14s} params {r['params']:6,d} MACs {r['conv_linear_macs']:10,d}")
for k, v in final.items():
    print(f"[최종 {k}] n={v['n']} {v['mean']:.3f} ± {v['sd']:.3f} [{v['min']:.2f}, {v['max']:.2f}] 사전고정 {v['prereg']} 최고 {v['best']} "
          f"| raw {v['raw_mean']:.3f} | epoch 고르기 평균 {v['epoch_pick_mean']:.3f} 최대 {v['epoch_pick_max']} | seeds {v['seeds']}")
for tk in "AB":
    t = S["track_eval"][tk]
    print(f"[트랙 {tk}] {t['name']} test {t['test']:.2f} 오답 {t['n_err']} params {t['params']:,} thop {t['thop_macs']:,} "
          f"ptflops {t['ptflops_macs']:,} 혼동 {t['top_pairs'][:5]}")
print(f"[트랙 A∩B 공통 오답] {S['track_eval']['common_errors']}  24모델 test 오답 빈도 {S['track_eval']['test_err_freq_24']}")
print("[teacher]", json.dumps({k: v for k, v in tch.items()}, ensure_ascii=False))
print("[AT]", json.dumps(S["at"]))
for k, v in soup.items():
    print(f"[soup {k}]", json.dumps(v))
print("[예산 내 앙상블]", json.dumps(S["ensemble_pairs"]))
print("[오답]", json.dumps({k: v for k, v in S["errors"].items() if k != "always_wrong_idx"}, ensure_ascii=False))
print("[노이즈]", json.dumps(S["noise"]))
print("[규모]", json.dumps(S["scale"]))
for k, v in abl.items():
    print(f"   {k:22s} n={v['n']:2d} {v['mean']:.3f} ± {v['sd']:.3f} (se {v['se']:.3f}) params {v['params']:6,d} MACs {v['macs'] / 1e6:5.2f}M {v['act']:5s} {v['label']}")
