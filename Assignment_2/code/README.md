# 딥러닝기초 과제 2 — MNIST 분류 모델 최적화 실험 코드

MNIST 분류 MLP에 대해 (1) Optimizer 비교, (2) LR Scheduler 비교, (3) K-Fold 앙상블을
수행하는 코드입니다. 강의자료 `pr_14_torch_mnist_v3` 코드를 기반으로 작성했습니다.

## 환경

- Python 3.12, PyTorch 2.x (CUDA), scikit-learn, numpy, matplotlib
- `mnist.npz` 가 이 폴더에 있어야 합니다 (강의자료 동일 파일).

```bash
pip install torch scikit-learn numpy matplotlib
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `data_loader.py` | MNIST 로드/표준화, train/val 분할 |
| `model.py` | MLP 784-256-256-10 (BatchNorm, Dropout 0.2) |
| `engine.py` | 학습 루프, optimizer/scheduler 팩토리, seed 고정 |
| `train.py` | 단일 설정 학습 + 테스트 평가 + JSON 로그 |
| `test.py` | 저장된 단일 모델 테스트 평가 |
| `train_ensemble.py` | K-Fold 앙상블 학습 (과제 3) |
| `test_ensemble.py` | 앙상블 평가 (Hard/Soft Voting) |
| `inference.py` | 단일 샘플 앙상블 추론 + 시각화 |
| `run_experiments.py` | 과제 (1)(2)(3) 전체 실험 자동 실행 |
| `plot_results.py` | 보고서용 그림/표 생성 |

## 실행 방법

### 전체 실험 재현 (권장)

```bash
python run_experiments.py   # Stage 1~4 전체 (GPU 기준 약 20~30분)
python plot_results.py      # 그림/표 생성
```

결과는 `../results/` 아래에 저장됩니다
(`logs/` 학습 로그 JSON, `models/` 모델 가중치, `figures/` 그림, `decisions.json` 단계별 선택 결과).

### 개별 실행 예시

```bash
# (1) 단일 optimizer 학습
python train.py --optimizer adam --scheduler none --lr 1e-3 --batch_size 128 --epochs 20

# (2) scheduler 적용 학습
python train.py --optimizer adam --scheduler cosine --lr 1e-3 --epochs 20

# (3) 5-Fold 앙상블 학습 → 평가 → 추론
python train_ensemble.py --optimizer adam --scheduler cosine --lr 1e-3 --k_splits 5 --epochs 20
python test_ensemble.py  --model_prefix ../results/models/ensemble/mlp_mnist --k_splits 5
python inference.py      --model_prefix ../results/models/ensemble/mlp_mnist --k_splits 5 --sample_idx 11

# 저장된 단일 모델 평가
python test.py --model_path ../results/models/ensemble/mlp_mnist_fold0.pth
```

GPU가 없으면 `--device cpu` 를 추가하세요.
