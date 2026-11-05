"""
보고서 표(markdown) 생성 — results/report/stats.json → results/report/tables.md
  python report_tables.py
차이(Δ)는 반올림 전 평균으로 계산한 뒤 반올림한다 (그림의 숫자와 같음).
"""
import json
import os

from common import ROOT

S = json.load(open(os.path.join(ROOT, "results", "report", "stats.json")))
A = S["ablation"]
out = []


def pm(k):
    a = A[k]
    return f"{a['mean']:.3f} ± {a['sd']:.3f}"


def d(k, ref):
    return f"{A[k]['mean'] - A[ref]['mean']:+.3f}"


def table(title, header, rows):
    out.append(f"\n### {title}\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")


# ---- 층별 구조
c = S["complexity"]
names = {"c3_16": "Conv 3×3 + BN + GELU", "M": "MaxPool 2×2", "R24": "잔차 블록 (3×3, 3×3, 1×1 지름길)",
         "R32": "잔차 블록 (3×3, 3×3, 1×1 지름길)", "G": "전역 평균 풀링 (GAP)", "L10": "Linear 32→10"}
rows = []
for r in c["layers"]:
    shp = "×".join(map(str, r["out_shape"]))
    rows.append([r["token"], names[r["token"]], shp, f"{r['params']:,}", f"{r['conv_linear_macs']:,}" if r["conv_linear_macs"] else "–"])
rows.append(["", "**합계**", "", f"**{c['params']:,}**", f"**{c['conv_linear_macs']:,}**"])
table("T_layers", ["토큰", "층", "출력 크기", "파라미터", "MACs (conv·linear)"], rows)

# ---- 구조 탐색 (KD 없음)
scr = ["scr_vgg_a", "scr_vgg_f", "scr_vgg_g", "scr_vgg_o", "scr_res_a", "scr_res_b", "scr_res_c", "scr_res_d",
       "scr_ds_a", "scr_sh_a", "scr_sh_a3", "scr_sh_b", "scr_sh_c", "scr_sh_c2"]
rows = [[A[k]["label"], f"`{A[k]['spec']}`", f"{A[k]['params']:,}", f"{A[k]['macs'] / 1e6:.2f}M", pm(k)] for k in scr]
table("T_screen", ["구조", "토큰 표기", "파라미터", "MACs", "검증 정확도 (%)"], rows)

# ---- KD 설정
kd = [("nokd_relu", "KD 없음", "–", "–"), ("kd_insample_T2", "in-sample (55,000장 전체로 학습)", "0.9", "2"),
      ("kd_insample2_T4_a09", "in-sample 2개 평균 (t_res + t_wide)", "0.9", "4"),
      ("kd_insample2_T4_a05", "in-sample 2개 평균 (t_res + t_wide)", "0.5", "4"),
      ("kd_oof_T1", "OOF (5-fold)", "0.9", "1"), ("kd_oof_T2", "**OOF (5-fold)**", "**0.9**", "**2**"),
      ("kd_oof_T3", "OOF (5-fold)", "0.9", "3"), ("kd_oof_T4", "OOF (5-fold)", "0.9", "4"),
      ("kd_oof_a05_T1", "OOF (5-fold)", "0.5", "1"), ("kd_oof_a05_T2", "OOF (5-fold)", "0.5", "2"),
      ("kd_oof_a05_T4", "OOF (5-fold)", "0.5", "4"), ("kd_oof_a10_T2", "OOF (5-fold)", "1.0", "2")]
rows = [[lab, a, t, A[k]["n"], pm(k), d(k, "nokd_relu") if k != "nokd_relu" else "기준"] for k, lab, a, t in kd]
table("T_kd", ["teacher", "α", "T", "seed", "검증 정확도 (%)", "KD 없음 대비 (%p)"], rows)

# ---- teacher 종류
tc = [("kd_oof_T2", "t_res OOF (기준)"), ("teacher_wide", "t_wide OOF (넓은 VGG형, 2,580,202 파라미터)"),
      ("teacher_res_wide", "t_res + t_wide OOF 로짓 평균"), ("teacher_two_sets", "t_res OOF 2세트 (fold 분할 다름) 로짓 평균")]
rows = [[lab, A[k]["n"], pm(k), d(k, "kd_oof_T2") if k != "kd_oof_T2" else "기준"] for k, lab in tc]
table("T_teacher", ["teacher", "seed", "검증 정확도 (%)", "기준 대비 (%p)"], rows)

# ---- KD 적용 후 구조
ka = ["kdarch_ds_a", "kdarch_ds_c", "kdarch_ds_k", "kdarch_ds_n", "kdarch_ds_a_p2", "kdarch_ds_c_p2", "kdarch_res_c30_p2"]
rows = [["Res-c (기준)", f"`{A['kd_oof_T2']['spec']}`", f"{A['kd_oof_T2']['params']:,}", f"{A['kd_oof_T2']['macs'] / 1e6:.2f}M",
         pm("kd_oof_T2"), "기준"]]
rows += [[A[k]["label"], f"`{A[k]['spec']}`", f"{A[k]['params']:,}", f"{A[k]['macs'] / 1e6:.2f}M", pm(k), d(k, "kd_oof_T2")] for k in ka]
table("T_kdarch", ["구조", "토큰 표기", "파라미터", "MACs", "검증 정확도 (%)", "기준 대비 (%p)"], rows)

# ---- 학습 하이퍼파라미터 등 (ReLU + OOF KD 기준)
hp = ["hp_lr005", "hp_lr02", "hp_wd1e4", "hp_wd1e3", "hp_bs64", "hp_ep30", "hp_ep120", "hp_sam005", "hp_sam015",
      "hp_drop02", "sel_aw02", "sel_aw05", "sel_aw10", "act_silu", "act_gelu", "fp32"]
rows = [["기준: lr 0.1, wd 5e-4, batch 128, 60 epoch, ReLU, bf16", A["kd_oof_T2"]["n"], pm("kd_oof_T2"), "기준"]]
rows += [[A[k]["label"] if k != "act_gelu" else "**GELU (채택)**", A[k]["n"], pm(k), d(k, "kd_oof_T2")] for k in hp]
table("T_hp", ["바꾼 설정", "seed", "검증 정확도 (%)", "기준 대비 (%p)"], rows)

# ---- GELU 이후 추가 기법
G = "act_gelu"
rows = [["기준: GELU + OOF KD", "–", A[G]["n"], pm(G), "기준"]]
for k, how in [("nokd_gelu", "KD만 뺌"), ("gelu_latepool", "R32를 14×14에서 계산 (MACs 7.25M)"),
               ("gelu_k5", "첫 conv만 5×5 (29,354 파라미터)"), ("gelu_rep_ac", "학습 때만 1×1·1×3·3×1·항등 가지 추가 후 합침")]:
    rows.append([A[k]["label"], how, A[k]["n"], pm(k), d(k, G)])
table("T_gelu_extra", ["기법", "내용", "seed", "검증 정확도 (%)", "기준 대비 (%p)"], rows)

# ---- AT (같은 작업 안 동시 대조군)
at = S["at"]
rows = []
for b in ["0", "100", "1000"]:
    a = at[b]
    rows.append([f"β = {b}" + (" (대조군)" if b == "0" else ""), a["n"], f"{a['mean']:.3f} ± {a['sd']:.3f}",
                 "대조군" if b == "0" else f"{a['mean'] - at['0']['mean']:+.3f}", f"{at['dist_last'][b]:.4f}"])
table("T_at", ["AT 가중치", "seed", "검증 정확도 (%)", "대조군 대비 (%p)", "teacher 특징 지도와의 거리"], rows)

# ---- soup
rows = []
for k in ["soup_b0", "soup_b8", "soup_b15", "soup_b15_rep", "soup_b15_m8", "soup_b30", "soup_b45"]:
    s = S["soup"][k]
    rows.append([s["branch_ep"], f"{s['G']} × {s['M']}", f"{s['member']:.3f}", f"{s['soup']:.3f}", f"{s['gain']:+.3f} ± {s['gain_se']:.3f}",
                 f"{s['ens']:.3f}"])
p = S["soup"]["pooled_b8_b30"]
table("T_soup", ["분기 epoch", "그룹 × 멤버", "멤버 평균 (%)", "soup (%)", "이득 (%p, ± 표준오차)", "출력 앙상블 (%, 참고)"], rows)
out.append(f"\n(8~30 epoch 분기 그룹 {p['n_groups']}개 합산 이득: {p['gain']:+.3f} ± {p['se']:.3f}%p)\n")

# ---- 최종 결과
F = S["final"]
rows = []
for k, lab in [("base_relu_nokd", "KD 없음 (ReLU, label smoothing 0.1)"), ("oofkd_relu", "OOF KD (ReLU)"),
               ("oofkd_gelu", "**OOF KD (GELU) — 최종**")]:
    f = F[k]
    rows.append([lab, f["n"], f"{f['mean']:.3f} ± {f['sd']:.3f}", f"{f['min']:.2f} – {f['max']:.2f}", f"{f['prereg']['test']:.2f}",
                 f"{f['best']['test']:.2f}"])
table("T_final", ["레시피", "seed", "테스트 정확도 (%)", "최소 – 최대", "사전 고정 모델", "최고 seed"], rows)

# ---- 클래스별 (트랙 A/B)
ta, tb = S["track_eval"]["A"], S["track_eval"]["B"]
rows = [["트랙 A"] + [f"{ta['per_class'][str(i)]:.2f}" for i in range(10)],
        ["트랙 B"] + [f"{tb['per_class'][str(i)]:.2f}" for i in range(10)]]
table("T_class", ["모델"] + [str(i) for i in range(10)], rows)

path = os.path.join(ROOT, "results", "report", "tables.md")
open(path, "w").write("\n".join(out) + "\n")
print("\n".join(out))
