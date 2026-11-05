# 딥러닝응용 과제 2 — CNN 기반 MNIST 분류기 (파라미터 ≤ 30,000, MACs ≤ 10,000,000)

202258096 임준현. 보고서: `202258096_임준현_딥러닝응용_과제2보고서.docx`

## 제출 모델

| 트랙 | 파일 | 테스트 정확도 | 파라미터 | MACs (thop / ptflops) |
|---|---|---|---|---|
| A (최종 학습 전에 지정한 모델) | `final_models/trackA_fin_oof_res_c_gelu_j0_s0.pt` | 99.64% | 29,098 | 4,693,376 / 4,716,874 |
| B (같은 레시피 24개 중 테스트 최고, 참고용) | `final_models/trackB_fin_oof_res_c_gelu_j1_s3.pt` | 99.68% | 29,098 | 4,693,376 / 4,716,874 |

- 구조: `c3_16 c3_16 M R24 M R32 G L10` (Conv16 → Conv16 → MaxPool → 잔차 블록24 → MaxPool → 잔차 블록32 → GAP → Linear10), GELU
- 학습: 5-fold out-of-fold(OOF) teacher의 soft label로 지식 증류 (α 0.9, T 2), 증강 없음, 학습 60,000장
- 체크포인트 내용: `spec`(모델 토큰 문자열), `act`, `state_dict`(EMA 가중치, 제출 모델), `raw_state_dict`, `args`(학습 인자)

## 설치

```bash
pip install -r code/requirements.txt     # torch, numpy, matplotlib, thop, ptflops
```

## 빠른 확인 (CPU 가능)

```bash
cd code
python evaluate.py ../final_models/trackA_fin_oof_res_c_gelu_j0_s0.pt ../final_models/trackB_fin_oof_res_c_gelu_j1_s3.pt
python models.py res_c                       # 구조별 파라미터·MACs (thop, ptflops)
python final_report.py "^fin_oof_res_c_gelu_j" # 24개 모델 평균 99.636 ± 0.023, 트랙 A·B
python report_tables.py                      # results/report/stats.json → tables.md
```

## 재현 (CUDA GPU 필요)

압축을 새 폴더에 풀어 실행한다. 동봉된 기록을 덮어쓰지 않도록 `repro_` 이름을 쓴다.

```bash
cd code
# 1) OOF teacher 5개 (각 약 3.2분) → OOF logit 합치기  (동봉된 results/teachers/t_res_full_oof5.pt를 쓰면 생략 가능)
for f in 0 1 2 3 4; do
  python train.py --model t_res --split full --nfolds 5 --fold $f --epochs 40 --amp --opt sgd \
    --lr 0.1 --wd 5e-4 --ls 0 --ema 0.999 --save-logits --seed $f --name repro_t_res_full_oof5_f$f
done
P=repro_t_res_full_oof5
python combine_oof.py --name $P --parts ${P}_f0,${P}_f1,${P}_f2,${P}_f3,${P}_f4

# 2) 최종 student 6개 동시 학습 (약 5분 30초). 트랙 A에 해당하는 모델 = _s0
python train_packed.py --model res_c --act gelu --split full --K 6 --opt sgd --epochs 60 \
  --lr 0.1 --wd 5e-4 --ls 0 --kd ../results/teachers/t_res_full_oof5.pt --kd-alpha 0.9 --kd-T 2 \
  --seed 11 --name repro_gelu_j0
#    1)을 실행했다면 --kd ../results/teachers/repro_t_res_full_oof5.pt 로 바꾼다
#    → results/runs/repro_gelu_j0_s{0..5}.json, results/ckpt/repro_gelu_j0_s{0..5}.pt
python final_report.py "^repro_gelu_j0"
```

- 원래 최종 학습 명령(`jobs/queue.txt`의 `fin_oof_res_c_gelu_j0..j3`)에는 `--test-every-epoch`가 붙어 있다. 매 epoch 테스트 정확도를 **기록만** 하는 옵션이며 학습에는 영향이 없다. 빼고 실행하면 테스트는 학습 종료 후 한 번만 계산된다.
- GPU 연산의 비결정성 때문에 다시 학습하면 결과가 조금 다를 수 있다.
- `report_stats.py`, `make_figures.py`(보고서 수치·그림)는 학습 중 생기는 `results/ckpt`, `results/logits`, fold별·in-sample teacher logit 파일(`results/teachers/t_res_full_oof5_f*.pt`, `t_res_dev_clean_s0.pt`)이 필요하다. 용량 때문에 압축파일에는 넣지 않았다.

## 실험 기록

- `jobs/queue.txt`: 실행 큐(`qdaemon.sh`)로 돌린 명령 104개(학습 102개, fold logit 합치기 2개). 실제 실행 순서와 시각은 `results/logs/_daemon.log`에 있다.
- 큐 밖에서 직접 실행한 명령 3개:
  - `python combine_oof.py t_res_dev_oof5 t_res_dev_oof5_f0 t_res_dev_oof5_f1 t_res_dev_oof5_f2 t_res_dev_oof5_f3 t_res_dev_oof5_f4`
  - `python combine_oof.py t_wide_dev_oof5 t_wide_dev_oof5_f0 t_wide_dev_oof5_f1 t_wide_dev_oof5_f2 t_wide_dev_oof5_f3 t_wide_dev_oof5_f4`
  - `python make_at.py --teacher t_res_dev_oof5` (attention transfer 실험 `at_scr` 전)
- `qdaemon.sh`는 기록용이다. `../../.venv` 경로를 가정하며, `results/runs`에 완료 기록이 모두 있으므로 그대로 실행하면 아무 작업도 하지 않는다.

## 테스트셋 격리 (과제 조건 2)

- `common.load_mnist()`는 학습/테스트 배열을 따로 반환하며, 학습 루프는 학습 배열에서만 미니배치를 뽑는다.
- 개발 단계(`--split dev`: 학습 55,000 / 검증 5,000)의 학습은 테스트 정확도를 계산하지 않는다 (`results/runs/`의 dev 기록 527개 모두 테스트 값 없음).
- OOF fold 분할, teacher soft label, attention 지도는 모두 학습 이미지에 대해서만 만든다.
- 최종 모델은 항상 마지막 epoch(60)의 EMA 가중치이며, 테스트 정확도로 epoch을 고르지 않았다.

## 학습 데이터 증강 (과제 조건 3)

제출 모델과 이 압축파일에 실린 실험에서는 student·teacher 모두 증강을 쓰지 않았다. 초기 버전의 학습 코드(teacher용 `train.py`, student용 `train_packed.py`)에는 기본값이 꺼진 증강 옵션이 있었고, 그 시기의 기록 275개(student 258개, teacher 17개)의 `args`에는 `"aug": false`가 남아 있다(teacher 기록에는 쓰이지 않은 증강 인자 `aug_rot` 등도 남아 있음). `aug: true`인 기록은 없으며, 증강 코드는 이후 삭제하였다. 그 뒤의 기록(최종 GELU 모델 24개를 포함한 student 312개, teacher 5개)에는 이 항목이 없다.

## 파일 설명 (`code/`)

| 파일 | 역할 |
|---|---|
| `common.py` | 데이터 로딩(학습·검증·테스트 분리), EMA, 복잡도 측정(thop·ptflops 중 큰 값) |
| `models.py` | 토큰 문자열로 CNN을 조립하는 빌더(`TokenNet`), student·teacher 구조 목록 |
| `packed.py` | K개 독립 모델을 grouped conv로 묶어 동시 학습, 단일 모델 추출(`extract`), 재매개변수화 합치기(`to_plain`). `python packed.py`로 동등성 검사 |
| `train.py` | teacher(및 단일 모델) 학습, OOF fold 분할(`--nfolds/--fold`), teacher logit 저장(`--save-logits`) |
| `combine_oof.py` | fold teacher logit을 OOF soft label 하나로 합침 |
| `train_packed.py` | student 학습: KD(`--kd`), EMA, SAM, soup(`--soup`), attention transfer(`--at`), 재매개변수화(`--rep`) |
| `make_at.py` | attention transfer용 OOF teacher attention 지도 생성 |
| `evaluate.py` | 체크포인트의 파라미터·MACs와 테스트 정확도 (CPU 가능) |
| `final_report.py`, `summarize.py` | 최종 결과(트랙 A/B)·실험 그룹별 결과 집계 |
| `analyze_logits.py`, `ensemble_pairs.py`, `teacher_val.py` | 분석 보조 |
| `report_stats.py`, `report_tables.py`, `make_figures.py` | 보고서 수치(`results/report/stats.json`)·표·그림 생성 |
| `qdaemon.sh` | `jobs/queue.txt`를 GPU 하나에서 순서대로 실행 |
| `requirements.txt` | 필요한 패키지 목록 |

## 결과 폴더 (`results/`)

- `runs/*.json`: 모델별 설정(`args`), epoch별 정확도(`hist`), 최종 정확도(`final_val_ema`, `final_test_ema` 등)
- `logs/*.log`: 학습 로그, `_daemon.log`: 실행 순서와 시각
- `teachers/t_res_full_oof5.pt`, `t_res_dev_oof5.pt`: OOF teacher logit (60,000 × 10, 학습 이미지)
- `report/stats.json`, `report/tables.md`: 보고서의 수치와 표
- `figures/*.png`: 보고서 그림

## 환경

WSL2 (Ubuntu 24.04), NVIDIA RTX 3070 Ti, Python 3.11, PyTorch 2.14 (CUDA 13.0), thop 0.1.1, ptflops 0.7.5, numpy, matplotlib
