# 딥러닝기초 과제 3 — MNIST 원본 크기(28×28) CNN 학습·평가·추론

강의자료 `pr_week_04_torch_mnist_cnn`의 `train.py`, `test.py`, `inference.py`, `model_v3.py`를
MNIST 원본 크기(28×28) 입력으로 동작하도록 수정한 코드입니다.

## 환경

- Python 3.11, PyTorch 2.x (CUDA), numpy, matplotlib, thop
- `mnist.npz` 가 이 폴더에 있어야 합니다 (강의자료 동일 파일).

## 파일 구성

| 파일 | 역할 | 강의자료 대비 주요 변경 |
|---|---|---|
| `model_v3.py` | CNN (Conv-BN-ReLU-Pool ×2 → FC-BN-ReLU-Dropout → FC) | `CNN(in_size)`로 입력 크기 인자화, fc1 입력 64×7×7=3136, reshape·flatten에 배치 차원 명시 |
| `data_loader.py` | MNIST 로드/표준화, train/val 분할 | 기본값 원본 28×28, 분할 seed 고정, 7×7 다운샘플 방식(`stride`/`avg`) 선택 |
| `train.py` | 학습 + 가중치/학습 곡선 저장 | `--use_small` 플래그화, `--small_method`, `--seed`, `--log_path` |
| `test.py` | 테스트셋 평가, 혼동행렬, 오분류 시각화 | MLP → CNN, 모델 경로, `--save_dir` |
| `inference.py` | 단일 샘플 추론 + 시각화 | MLP → CNN, 모델 경로, softmax 확신도 출력 |
| `run_experiments.py` | 28×28 / 7×7(stride) / 7×7(avg) × seed 3개 전체 실험 | 신규 |
| `plot_results.py` | 보고서용 그림/표 생성 | 신규 |
| `extra_checks.py` | 보고서 '알아두기' 항목의 보조 측정 | 신규 |
| `extra_experiment.py` | 추가 실험(보고서 11장): 학습 설정 개선(A), 구조 개선 `CNNPlus`(B) | 신규 |

## 실행 방법

```bash
# 개별 실행 (기본: 원본 28×28 입력)
python train.py --device cuda                       # -> cnn_mnist_28.pth
python test.py --device cuda
python inference.py --device cuda --sample_idx 12
python model_v3.py                                  # 파라미터 수 / MACs

# 7×7 다운사이징 입력 비교
python train.py --use_small --device cuda --save_path cnn_mnist_7.pth
python test.py  --use_small --device cuda --model_path cnn_mnist_7.pth

# 7×7을 4×4 블록 평균으로 만든 입력 (추가 비교)
python train.py --use_small --small_method avg --device cuda --save_path cnn_mnist_7avg.pth
python test.py  --use_small --small_method avg --device cuda --model_path cnn_mnist_7avg.pth

# 전체 실험 재현 (GPU 기준 약 5분) + 그림/표 생성
python run_experiments.py            # 일부만: --configs 7avg
python plot_results.py
python extra_checks.py > ../results/logs/extra_checks.txt
python extra_experiment.py            # 추가 실험 A, B × seed 3개 (약 5분), --plot: 그림만
```

결과는 `../results/` 아래에 저장됩니다
(`logs/` 학습·테스트 로그, `models/` 가중치, `figures/` 그림, `tables/summary.md` 요약 표).
