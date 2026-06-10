import os
import argparse
import torch
from data_loader import load_mnist

from model import MLP
import matplotlib.pyplot as plt
#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int=0):
    """
    x_test_raw: 원본(28×28) 이미지 배열, shape=(N,28,28)
    y_pred, y_true: 1차원 리스트/배열
    """
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred}, True: {y_true[idx]}")
    plt.axis('off')
    plt.show()

#%%
plt.close("all")
# plt.close("all")
# from inference import show_sample

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# def test(model, test_loader, device=None):
#     model.eval()
#     correct = 0
#     total   = len(test_loader.dataset)
#     all_preds = []

#     with torch.no_grad():
#         for data, target in test_loader:
#             if device:
#                 data, target = data.to(device), target.to(device)

#             output = model(data)
#             pred   = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             all_preds.extend(pred.squeeze().cpu().tolist())

#     acc = 100.0 * correct / total
#     return acc, all_preds


def infer_single(model, x_sample, device=None):
    """
    하나의 샘플 입력에 대해 예측을 수행하는 함수
    x_sample: numpy 배열 또는 torch tensor 형태의 단일 샘플 (C=1, H, W) 또는 벡터로 전처리된 입력
    """
    model.eval()
    # 텐서로 변환 및 디바이스 이동
    if isinstance(x_sample, torch.Tensor):
        tensor = x_sample.float()
    else:
        tensor = torch.FloatTensor(x_sample)
    if device:
        tensor = tensor.to(device)
    # 배치 차원 추가
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return pred


#%%
def main(args):
    # DataLoader 준비
    train_loader, test_loader, in_dim, x_test_raw, y_test = load_mnist(
        use_small=args.use_small,
        batch_size=1,
        shuffle=False
    )

    # 모델 초기화 및 가중치 로드
    model = MLP(in_dim)
    device = torch.device(args.device) if args.device else torch.device('cuda')
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # # 전체 테스트 수행
    # test_acc, y_pred = test(model, test_loader, device=device)
    # print(f"Test Accuracy: {test_acc:.2f}%")

    # 샘플 시각화 (옵션에 따라 건너뜀)
    # if not args.no_display:
    # show_sample(x_test_raw, y_pred, y_test, idx=args.sample_idx)

    # 단일 샘플 inference 예시
    sample = x_test_raw[args.sample_idx]
    # 다운사이징 여부에 따라 형태 맞추기
    if args.use_small:
        sample_proc = torch.FloatTensor(sample[::4, ::4].reshape(-1))
    else:
        sample_proc = torch.FloatTensor(sample.reshape(-1))
    pred_label = infer_single(model, sample_proc, device=device)
    print(f"Single Sample Inference -> Pred: {pred_label}, True: {y_test[args.sample_idx]}")

    show_sample(x_test_raw, pred_label, y_test, idx=args.sample_idx)
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MLP on MNIST with single-sample inference")
    # parser.add_argument("--use_small", action="store_true",                        help="use 7x7 down-sampled input for testing")
    # parser.add_argument("--batch_size", type=int, default=64,                        help="mini-batch size for testing")
    parser.add_argument("--device", type=str, default="cuda",                        help="device for inference (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--model_path", type=str, default="mlp_mnist.pth",                        help="file path of the saved model to load")
    parser.add_argument("--sample_idx", type=int, default=12,                        help="index of the test sample to visualize and infer")
    parser.add_argument("--use_small",   type=bool,  default=False,  help="use 7x7 down-sampled input")
    parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    args, _ = parser.parse_known_args()  # Unknown args ignored (e.g., --wdir)

    main(args)
