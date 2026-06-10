import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from data_loader import load_mnist
from model import MLP
plt.close("all")
# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#%%
def show_sample(x_test_raw, y_pred, y_true, idx: int=0):
    """
    x_test_raw: 원본(28×28) 이미지 배열, shape=(N,28,28)
    y_pred, y_true: 1차원 리스트/배열 또는 스칼라 값
    """
    plt.figure()
    plt.imshow(x_test_raw[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx] if isinstance(y_pred, (list, np.ndarray)) else y_pred}, True: {y_true[idx]}")
    plt.axis('off')
    plt.show()

#%%
def infer_single(model, x_sample, device=None):
    """
    하나의 샘플 입력에 대해 예측을 수행하는 함수
    x_sample: 벡터로 전처리된 1차원 배열 또는 torch tensor
    """
    model.eval()
    # 텐서로 변환 및 디바이스 이동
    tensor = x_sample if isinstance(x_sample, torch.Tensor) else torch.FloatTensor(x_sample)
    if device:
        tensor = tensor.to(device)
    # 배치 차원 추가
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return pred

#%%
def load_custom_image(path, use_small):
    """
    사용자 그림 이미지를 불러와 모델 입력 형태로 전처리
    1. 흑백으로 읽고 28x28로 리사이즈
    2. numpy 배열로 변환 후 정규화
    3. use_small 여부에 따라 down-sample 및 flatten
    """
    img = Image.open(path).convert('L')  # 흑백
    img = ImageOps.invert(img)
    # ANTIALIAS is deprecated in newer Pillow versions; use LANCZOS or Resampling
    # For Pillow>=10.0.0, use Image.Resampling.LANCZOS
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    img = img.resize((28, 28), resample_filter)
    arr = np.array(img, dtype=np.float32)
    # 원본 학습 정규화 사용 (train set mean/std)
    mean = np.mean(arr)
    std  = np.std(arr) if np.std(arr) > 0 else 1.0
    arr = (arr - mean) / std
    if use_small:
        arr = arr[::4, ::4]
    return arr.reshape(-1)

#%%
def main(args):
    # DataLoader 준비 (테스트 용이지만 train_loader 불필요)
    _, test_loader, in_dim, x_test_raw, y_test = load_mnist(
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

    # 커스텀 이미지를 통한 inference
    # if args.input_image:
    sample_proc = load_custom_image(args.input_image, args.use_small)
    pred_label = infer_single(model, sample_proc, device=device)
    print(f"Custom Image Inference -> Pred: {pred_label}")

    img_vis = Image.open(args.input_image).convert('L')  # 흑백
    plt.figure()
    plt.imshow(img_vis, cmap='gray')
    plt.title(f"Input Image: {args.input_image}")
    plt.axis('off')
    plt.show()
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MLP on MNIST with single-sample or custom-image inference"
    )
    # parser.add_argument("--use_small", action="store_true",                        help="use 7x7 down-sampled input for inference")
    parser.add_argument("--device", type=str, default="cuda",                        help="device for inference (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--model_path", type=str, default="mlp_mnist.pth",                        help="file path of the saved model to load")
    parser.add_argument("--sample_idx", type=int, default=12,                        help="index of the test sample to visualize and infer")
    parser.add_argument("--input_image", type=str, default='test_num3.jpg',                        help="path to custom digit image for inference")
    parser.add_argument("--use_small",   type=bool,  default=False,  help="use 7x7 down-sampled input")
    parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    
    args, _ = parser.parse_known_args()

    main(args)
