import os
import argparse
import torch
from data_loader import load_mnist
from model_v2 import MLP
import matplotlib.pyplot as plt

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.close("all")

#%% 함수 정의

def show_sample(x_test_raw, y_pred, y_true, idx: int = 0):

    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx] if isinstance(y_pred, (list, tuple)) else y_pred}, True: {y_true[idx]}")
    plt.axis('off')
    plt.show()


def infer_single(model, x_sample, device=None):

    model.eval()
    # 텐서로 변환 및 디바이스 이동
    if not isinstance(x_sample, torch.Tensor):
        tensor = torch.FloatTensor(x_sample)
    else:
        tensor = x_sample.float()
    if device is not None:
        tensor = tensor.to(device)
    # 배치 차원 추가
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return pred


def infer_ensemble(model_prefix, k_splits, x_sample, in_dim, device=None):

    preds_per_fold = []

    for fold in range(k_splits):
        model_path = f"{model_prefix}_fold{fold}.pth"
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_path} 파일을 찾을 수 없습니다.")

        # 모델 초기화 및 가중치 로드
        model = MLP(in_dim)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # 단일 Fold 예측
        pred = infer_single(model, x_sample, device=device)
        preds_per_fold.append(pred)

    # 다수결 투표: 가장 많이 등장한 클래스 (동률 시 가장 작은 레이블)
    vote_counts = {}
    for v in preds_per_fold:
        vote_counts[v] = vote_counts.get(v, 0) + 1
    max_count = max(vote_counts.values())
    winners = [cls for cls, cnt in vote_counts.items() if cnt == max_count]
    ensemble_pred = min(winners)

    return preds_per_fold, ensemble_pred


#%% main 함수

def main(args):
    # 1) 데이터 로드
    _, _, _, in_dim, x_test_raw, y_test = load_mnist( use_small=args.use_small,valDB_portion=0.0)

    # 2) 단일 샘플 전처리
    sample = x_test_raw[args.sample_idx]
    if args.use_small:
        # 28×28 -> 7×7 다운샘플링 (4배 스트라이드)
        sample_proc = torch.FloatTensor(sample[::4, ::4].reshape(-1))
    else:
        sample_proc = torch.FloatTensor(sample.reshape(-1))

    # 3) 장치 설정
    device = torch.device(args.device) if args.device else torch.device('cuda')

    # 4) 단일 모델 inference
    # 모델 초기화 및 가중치 로드
    model_single = MLP(in_dim)
    state_dict = torch.load(args.model_path, map_location=device)
    model_single.load_state_dict(state_dict)
    model_single.to(device)

    single_pred = infer_single(model_single, sample_proc, device=device)
    print(f"Sample Index: {args.sample_idx}")
    print(f"Prediction: {single_pred}, True Label: {y_test[args.sample_idx]}\n")

    # 5) 앙상블 inference (Fold별 모델이 있을 때만 수행)
    if args.k_splits > 1:
        preds_per_fold, ensemble_pred = infer_ensemble(
            model_prefix=args.model_prefix,
            k_splits=args.k_splits,
            x_sample=sample_proc,
            in_dim=in_dim,
            device=device
        )
        print(f"=== Ensemble Inference (Fold {args.k_splits}) ===")
        for fold, p in enumerate(preds_per_fold):
            print(f"Fold {fold} Prediction: {p}")
        print(f"Ensemble Prediction (majority vote): {ensemble_pred}, True Label: {y_test[args.sample_idx]}\n")
    else:
        preds_per_fold, ensemble_pred = [], None

    # 6) 시각화
    # 단일 모델 예측 혹은 앙상블 예측을 타이틀로 보여줌
    if args.k_splits > 1:
        # 앙상블 예측을 사용
        show_sample(x_test_raw, [ensemble_pred], y_test, idx=args.sample_idx)
    else:
        # 단일 모델 예측을 사용
        show_sample(x_test_raw, single_pred, y_test, idx=args.sample_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser( )
    parser.add_argument("--device",       type=str,  default="cuda")
    parser.add_argument("--model_path",   type=str,  default="mlp_mnist.pth")
    parser.add_argument("--model_prefix", type=str,  default="mlp_mnist.pth")
    parser.add_argument("--k_splits",     type=int,  default=1)
    parser.add_argument("--sample_idx",   type=int,  default=12)
    parser.add_argument("--use_small",    type=bool, default=False)
    args, _ = parser.parse_known_args()

    main(args)
