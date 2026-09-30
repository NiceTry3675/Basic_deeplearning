import os
import argparse
import torch
from data_loader import load_mnist, downsample
# from torch.utils.data import  DataLoader
from model_v3 import CNN
import matplotlib
import matplotlib.pyplot as plt
#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int=0, save_path=None):
    """
    x_test_raw: 원본(28×28) 이미지 배열, shape=(N,28,28)
    y_pred, y_true: 1차원 리스트/배열
    """
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred}, True: {y_true[idx]}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

#%%
plt.close("all")

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def infer_single(model, x_sample, device=None):
    """
    하나의 샘플 입력에 대해 예측을 수행하는 함수
    x_sample: numpy 배열 또는 torch tensor 형태의 단일 샘플 (C=1, H, W) 또는 벡터로 전처리된 입력
    """
    model.eval()   # BatchNorm: running 통계 사용(배치 1에서도 동작), Dropout: 비활성
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
        prob = torch.softmax(output, dim=1).squeeze(0)
        pred = prob.argmax().item()
    return pred, prob[pred].item()


#%%
def main(args):
    # DataLoader 준비
    _, _, _, in_dim, x_test_raw, y_test = load_mnist(
        use_small=args.use_small,
        valDB_portion = 0.1
    )
    # test_loader  = DataLoader(test_ds,   batch_size=args.batch_size,   shuffle=False)

    in_size = int(in_dim ** 0.5)   # 784 -> 28, 49 -> 7

    # 모델 초기화 및 가중치 로드
    model = CNN(in_size)
    device = torch.device(args.device) if args.device else torch.device('cpu')
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
        sample_proc = torch.FloatTensor(downsample(sample, args.small_method).reshape(-1))
    else:
        sample_proc = torch.FloatTensor(sample.reshape(-1))
    pred_label, conf = infer_single(model, sample_proc, device=device)
    print(f"Input: {in_size}x{in_size}, Model: {args.model_path}")
    print(f"Single Sample Inference [idx={args.sample_idx}] -> Pred: {pred_label} (conf {conf*100:.2f}%), True: {y_test[args.sample_idx]}")

    save_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"inference_{in_size}_idx{args.sample_idx}.png")
    show_sample(x_test_raw, pred_label, y_test, idx=args.sample_idx, save_path=save_path)
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN single-sample inference on MNIST")
    # parser.add_argument("--use_small", action="store_true",                        help="use 7x7 down-sampled input for testing")
    # parser.add_argument("--batch_size", type=int, default=64,                        help="mini-batch size for testing")
    parser.add_argument("--device", type=str, default=None,                        help="device for inference (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--model_path", type=str, default="cnn_mnist_28.pth",                        help="file path of the saved model to load")
    parser.add_argument("--sample_idx", type=int, default=12,                        help="index of the test sample to visualize and infer")
    parser.add_argument("--use_small",  action="store_true",  help="use 7x7 down-sampled input (default: original 28x28)")
    parser.add_argument("--small_method", type=str, default="stride", choices=["stride", "avg"], help="7x7 down-sampling: stride(::4, lecture) or avg(4x4 mean)")
    parser.add_argument("--save_dir",   type=str, default=None,  help="save figure here instead of plt.show()")
    # parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    args, _ = parser.parse_known_args()  # Unknown args ignored (e.g., --wdir)

    if args.save_dir:
        matplotlib.use("Agg")
    main(args)
