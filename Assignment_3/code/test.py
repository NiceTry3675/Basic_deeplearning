import os
import json
import argparse
import numpy as np
import torch
from data_loader import load_mnist
from model_v3       import CNN
from torch.utils.data import  DataLoader

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
import matplotlib.pyplot as plt
plt.close("all")
#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int=0, save_path=None):
    """
    x_test_raw: 원본(28×28) 이미지 배열, shape=(N,28,28)
    y_pred, y_true: 1차원 리스트/배열
    """
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}, True: {y_true[idx]}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def show_misclassified(x_test_raw, y_pred, y_true, n: int=16, save_path=None):
    """오분류 샘플을 격자로 시각화"""
    wrong = np.where(np.asarray(y_pred) != np.asarray(y_true))[0][:n]
    cols = 8
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(x_test_raw[i], cmap='gray')
        ax.set_title(f"#{i}\nP:{y_pred[i]} T:{y_true[i]}", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def confusion_matrix(y_pred, y_true, num_classes: int=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
#%%
def test(model, test_loader, device=None):
    model.eval()
    correct = 0
    total   = len(test_loader.dataset)
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            if device:
                data, target = data.to(device), target.to(device)

            output = model(data)
            pred   = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # squeeze()는 배치 크기가 1이면 스칼라가 되어 extend가 실패하므로 view(-1) 사용
            all_preds.extend(pred.view(-1).cpu().tolist())

    acc = 100.0 * correct / total
    return acc, all_preds


#%%
def main(args):
    # DataLoader 준비 (학습 때와 동일한 전처리/입력 크기)
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist( use_small=args.use_small, valDB_portion = 0.1, small_method=args.small_method )
    test_loader  = DataLoader(test_ds,   batch_size=args.batch_size,   shuffle=False)
    in_size = int(in_dim ** 0.5)   # 784 -> 28, 49 -> 7

    # 모델 초기화 및 가중치 로드
    model = CNN(in_size)

    device = torch.device(args.device) if args.device else torch.device('cpu')
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 테스트 수행
    test_acc, y_pred = test(model, test_loader, device=device)
    n_wrong = int((np.asarray(y_pred) != y_test).sum())
    print(f"Input: {in_size}x{in_size}, Model: {args.model_path}")
    print(f"Test Accuracy: {test_acc:.2f}% ({len(y_test) - n_wrong}/{len(y_test)}, wrong={n_wrong})")

    cm = confusion_matrix(y_pred, y_test)
    class_acc = 100.0 * cm.diagonal() / cm.sum(axis=1)
    print("Per-class Accuracy: " + ", ".join(f"{c}:{a:.2f}" for c, a in enumerate(class_acc)))

    out = args.save_dir
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"test_result_{in_size}.json"), "w") as f:
            json.dump({"model_path": args.model_path, "in_size": in_size, "test_acc": test_acc,
                       "n_wrong": n_wrong, "class_acc": class_acc.tolist(),
                       "confusion_matrix": cm.tolist(), "y_pred": y_pred}, f)

    show_sample(x_test_raw, y_pred, y_test, idx=args.sample_idx,
                save_path=os.path.join(out, f"test_sample_{in_size}.png") if out else None)
    show_misclassified(x_test_raw, y_pred, y_test,
                       save_path=os.path.join(out, f"test_misclassified_{in_size}.png") if out else None)


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CNN on MNIST")
    parser.add_argument("--batch_size", type=int, default=64,                        help="mini-batch size for testing")
    parser.add_argument("--device", type=str, default=None,                        help="device for inference (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--model_path", type=str, default="cnn_mnist_28.pth",          help="file path of the saved model to load")
    parser.add_argument("--sample_idx", type=int, default=11,                        help="index of the test sample to visualize")
    parser.add_argument("--use_small",  action="store_true",  help="use 7x7 down-sampled input (default: original 28x28)")
    parser.add_argument("--small_method", type=str, default="stride", choices=["stride", "avg"], help="7x7 down-sampling: stride(::4, lecture) or avg(4x4 mean)")
    parser.add_argument("--save_dir",   type=str, default=None,  help="save figures/results here instead of plt.show()")
    args, _ = parser.parse_known_args()  # Unknown args ignored (e.g., --wdir)

    if args.save_dir:
        matplotlib.use("Agg")
    main(args)
