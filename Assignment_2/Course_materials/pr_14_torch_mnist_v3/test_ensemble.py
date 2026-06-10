import os
import argparse
import torch
from data_loader import load_mnist
from model_v2 import MLP
from torch.utils.data import DataLoader

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
plt.close("all")
#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int = 0):
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}, True: {y_true[idx]}")
    plt.axis('off')
    plt.show()
#%%
def test_model(model, test_loader, device=None):

    model.eval()
    correct = 0
    total   = len(test_loader.dataset)
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            if device is not None:
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred   = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().tolist())

    acc = 100.0 * correct / total
    return acc, all_preds

#%%
def ensemble_test(all_fold_preds, num_classes=10):

    k_splits = len(all_fold_preds)
    num_test = len(all_fold_preds[0])
    ensemble_preds = []

    for i in range(num_test):
        # 샘플 i에 대한 Fold별 예측 모음
        votes = [all_fold_preds[f][i] for f in range(k_splits)]
        # 투표 집계
        vote_counts = {}
        for v in votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1

        # 최다 득표 클래스(동률 시 가장 작은 레이블 선택)
        max_count = max(vote_counts.values())
        winners = [cls for cls, cnt in vote_counts.items() if cnt == max_count]
        ensemble_pred = min(winners)
        ensemble_preds.append(ensemble_pred)

    return ensemble_preds

#%%
def evaluate_accuracy(preds, y_true):

    correct = sum([1 for p, t in zip(preds, y_true) if p == t])
    total   = len(y_true)
    return 100.0 * correct / total

#%%
def main(args):
    # 1) DataLoader 준비
    # load_mnist 반환값: train_ds, val_ds, test_ds, in_dim, x_test_raw, y_test
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist( use_small=args.use_small, valDB_portion=0.0 )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2) 장치 설정
    device = torch.device(args.device) if args.device else torch.device('cuda')

    # 3) Fold별 단독 테스트 및 예측값 수집
    all_fold_preds = []
    fold_accuracies = []

    for fold in range(args.k_splits):
        model_path = f"{args.model}_fold{fold}.pth"


        # 모델 초기화 및 가중치 로드
        model = MLP(in_dim)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # 단일 모델 테스트
        acc, preds = test_model(model, test_loader, device=device)
        print(f"Fold {fold} Test Accuracy: {acc:.2f}%")

        fold_accuracies.append(acc)
        all_fold_preds.append(preds)

    # 4) 앙상블 다수결 투표 수행
    ensemble_preds = ensemble_test(all_fold_preds)
    ensemble_acc = evaluate_accuracy(ensemble_preds, y_test)
    print(f"\n## Ensemble Test Accuracy: {ensemble_acc:.2f}%")

    # 5) 샘플 시각화
    show_sample(x_test_raw, ensemble_preds, y_test, idx=args.sample_idx)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--model",  type=str,   default="mlp_mnist.pth"  )
    parser.add_argument("--k_splits",      type=int,   default=4)
    parser.add_argument("--sample_idx",    type=int,   default=11)
    parser.add_argument("--use_small",     type=bool,  default=False)
    args, _ = parser.parse_known_args()  # Unknown args 무시

    main(args)
