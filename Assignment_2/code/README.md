# 딥러닝기초 과제 2 — MNIST Optimizer / Scheduler / Ensemble 비교

202258096 임준현

강의자료 `pr_14_torch_mnist_v3` 코드를 기반으로 (1) optimizer 비교, (2) LR scheduler 비교,
(3) K-Fold 앙상블을 수행하는 코드입니다.

## 환경

- Python 3.12, PyTorch 2.x (CUDA 권장, CPU도 동작), numpy, scikit-learn, matplotlib
- 데이터: 같은 폴더의 `mnist.npz` (포함되어 있음)

```bash
pip install torch numpy scikit-learn matplotlib
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `data_loader.py` | MNIST 로드, z-정규화, train/val 분할(시드 고정) |
| `model.py` | MLP(784-256-256-10, BatchNorm+Dropout) |
| `engine.py` | 공용 학습/평가 루프 (epoch/batch/plateau scheduler 모드 지원) |
| `configs.py` | 공통 하이퍼파라미터, optimizer/scheduler 정의, 최종 선택 구성(BEST) |
| `run_experiments.py` | Part 1·2 실험 러너 (JSON/CSV/MD 표 + 곡선 생성) |
| `plotting.py` | 비교 곡선 시각화 (단독 재실행 가능) |
| `train.py` | **Part 3 학습**: K-Fold(k=5) 앙상블 + 시드 앙상블(full-train×5) |
| `test.py` | **Part 3 평가**: 테스트셋에서 단일/K-Fold/시드 앙상블, hard/soft voting 비교 |
| `inference.py` | **Part 3 추론**: 단일 샘플 앙상블 추론 |

## 실행 순서

```bash
# (1) Optimizer 비교
python run_experiments.py --part 1 --stage sweep   # optimizer별 LR 탐색 (5 epochs)
python run_experiments.py --part 1 --stage main    # 최적 LR로 본 비교 (15 epochs × 3 seeds)

# (2) Scheduler 비교 — (1)에서 선택된 구성을 인자로 전달
python run_experiments.py --part 2 --optimizer <우승optimizer> --lr <우승lr>

# (3) 앙상블 — configs.BEST에 (2)까지의 우승 구성이 기본값으로 저장되어 있음
python train.py        # 5-fold 앙상블 + full-train×5(시드 앙상블) 학습 → models/*.pth
python test.py         # 테스트셋 10,000장 평가 (단일/K-Fold/시드 앙상블 비교)
python inference.py --sample_idx 11   # 단일 샘플 추론
```

- 동작 확인용 축소 실행: `python run_experiments.py --part 1 --stage sweep --quick`,
  `python train.py --epochs 2`
- 학습된 모델(`models/*.pth`)이 포함되어 있으므로 `test.py`/`inference.py`는
  재학습 없이 바로 실행 가능합니다.

## 출력물

- `../results/part1_sweep/`, `../results/part1/`, `../results/part2/` — run별 JSON,
  `summary.csv`, `summary.md`(보고서용 표)
- `../results/part3/` — 학습 이력, `test_results.json`
- `../figures/` — 비교 곡선, LR 스케줄, 앙상블 결과 그래프 PNG
- `models/` — `mlp_fold{0..4}.pth`, `mlp_full_seed{0..4}.pth`
