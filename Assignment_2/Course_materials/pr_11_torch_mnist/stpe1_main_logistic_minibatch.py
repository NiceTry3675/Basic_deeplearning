import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
plt.close("all")
#%% 데이터 로드 및 전처리
mnist = np.load('mnist.npz')
x_train = (mnist['x_train'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
y_train = mnist['y_train']
x_test  = (mnist['x_test']  - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
y_test  = mnist['y_test']

# one-hot 변환
def to_onehot(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train_oh = to_onehot(y_train, 10)
y_test_oh  = to_onehot(y_test, 10)

#%% 다운사이징(선택) 또는 전체 픽셀 사용
use_small = False
if use_small:
    # 7×7로 다운사이징
    x_train_in = x_train[:, ::4, ::4].reshape(-1, 7*7)
    x_test_in  = x_test[:,  ::4, ::4].reshape(-1, 7*7)
    in_dim = 7*7
else:
    # 원본 28×28 사용
    x_train_in = x_train.reshape(-1, 28*28)
    x_test_in  = x_test.reshape(-1, 28*28)
    in_dim = 28*28

print("Input dim:", in_dim)
print("Train:", x_train_in.shape, y_train_oh.shape)
print("Test: ", x_test_in.shape,  y_test_oh.shape)

#%% 모델 파라미터 초기화
w = np.random.randn(in_dim, 10) * 0.01
b = np.zeros(10,)

#%% 함수 정의
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def hypothesis(w, x, b):
    return softmax(x.dot(w) + b)

def cross_entropy(y_true, y_pred):
    idx = np.argmax(y_true, axis=1)
    p   = y_pred[np.arange(len(idx)), idx]
    return -np.mean(np.log(p + 1e-8))

#%% 하이퍼파라메터
epochs     = 10
alpha         = 0.01
batch_size = 64
n_samples  = x_train_in.shape[0]

# 학습 기록용
loss_history = []

#%% 미니배치 SGD
for epoch in range(1, epochs+1):
    # 데이터 셔플
    perm = np.random.permutation(n_samples)
    Xs   = x_train_in[perm]
    Ys   = y_train_oh[perm]

    epoch_loss = 0.0
    for i in range(0, n_samples, batch_size):
        xb = Xs[i:i+batch_size]
        yb = Ys[i:i+batch_size]

        # 순전파
        preds = hypothesis(w, xb, b)
        loss  = cross_entropy(yb, preds)
        epoch_loss += loss * xb.shape[0]

        # 역전파
        grad_w = xb.T.dot(preds - yb) / xb.shape[0]
        grad_b = np.sum(preds - yb, axis=0) / xb.shape[0]

        # 파라미터 업데이트
        w = w - alpha * grad_w
        b = b - alpha * grad_b

    epoch_loss /= n_samples
    loss_history.append(epoch_loss)

    if epoch % 1 == 0 or epoch == 1:
        print(f"[Epoch {epoch:2d}] Loss: {epoch_loss:.6f}")

#%% 학습 손실 시각화
plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

#%% 테스트 정확도
y_pred_test = hypothesis(w, x_test_in, b)
y_pred_lbl  = np.argmax(y_pred_test, axis=1)
acc = accuracy_score(y_test, y_pred_lbl)
print(f"Test Accuracy: {acc:.4f}")

#%% 샘플 시각화
plt.figure()
idx = 20
orig_img = x_test[idx]
plt.imshow(orig_img, cmap='gray')
plt.title(f"Pred: {y_pred_lbl[idx]}, True: {y_test[idx]}")
plt.axis('off')
plt.show()
