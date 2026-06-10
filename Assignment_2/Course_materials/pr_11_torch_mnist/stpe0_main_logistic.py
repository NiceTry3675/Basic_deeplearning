import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
#%%
# 'mnist.npz' 파일 다운로드!
mnist = np.load('mnist.npz')

# 정규화 코드 추가 : x_norm = (x - u) / sigma
x_train = (mnist['x_train'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
y_train = mnist['y_train']
x_test = (mnist['x_test'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
y_test = mnist['y_test']
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#%%
def softmax(x):
    exp_x = np.exp(x)
    for i in range(len(x)):
        exp_x[i] /= np.sum(exp_x[i])
    return exp_x
#%%
def hypothesis(w, x, b):
    return softmax(x.dot(w) + b)
#%%
def cross_entropy(y_true, y_pred):
    # one-hot vector -> label
    # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] -> 0
    # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] -> 5
    y_true = np.argmax(y_true, axis=-1)

    # 레이블에 해당하는 y_pred만을 가져옴
    y_pred = y_pred[np.arange(y_true.shape[0]), y_true]

    return -np.mean(np.log(y_pred + 1e-8))
#%%
# label을 onehot vector로 변환
# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# 9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
def to_onehot(labels, num_classes):
    return np.eye(num_classes)[labels]
#%%
# onehot vector의 형태로 변환함
y_train_onehot = to_onehot(y_train, 10) # 클래스는 0~9까지 10개
y_test_onehot = to_onehot(y_test, 10) # 클래스는 0~9까지 10개
print(y_train_onehot.shape, y_test_onehot.shape)
#%%
# (28, 28) 크기의 이미지를 (7, 7) 크기의 이미지로 down-sizing (계산 효율을 위해서)
# (7, 7) 크기를 (49,) 크기로 reshape
use_small = False
if use_small:
    x_train_small = x_train[:, ::4, ::4].reshape(-1, 7*7)
    x_test_small = x_test[:, ::4, ::4].reshape(-1, 7*7)
    print(x_train_small.shape, x_test_small.shape)
    w = np.random.rand(7*7, 10)
    b = np.random.rand(10,)
    print(w.shape, b.shape)
#%%
else:    
    x_train_small = x_train.reshape(-1, 28*28)
    x_test_small = x_test.reshape(-1, 28*28)
    
    w = np.random.rand(28*28, 10)
    b = np.random.rand(10,)
    print(w.shape, b.shape)
#%%
# 하이퍼파라메타 설정!
epoch =100
alpha =0.01

# 총 학습데이터 숫자 (60000)
sample_num = x_train_small.shape[0]

total_loss = []
for i in range(epoch):
    h = hypothesis(w, x_train_small, b)
    loss = cross_entropy(y_train_onehot, h)

    # 경사하강법으로 파라미터 업데이트
    grad_w = x_train_small.T.dot(h - y_train_onehot) / sample_num  # w에 대한 기울기
    grad_b = np.sum(h - y_train_onehot, axis=0) / sample_num       # b에 대한 기울기
    
    # 파라미터 업데이트
    w = w - alpha * grad_w
    b = b - alpha * grad_b
    # if i % 10 == 0:
    print(f"[Epoch : {i:3d}] Loss : {loss:.10f}")
    total_loss.append(loss)
        
#%%    
total_loss = np.array(total_loss)
plt.figure()
plt.plot(10.0 * np.log(total_loss / (np.max(total_loss + 1e-5))))
plt.show()
#%%
from sklearn.metrics import accuracy_score

y_pred_onehot = hypothesis(w, x_test_small, b)
y_pred = np.argmax(y_pred_onehot, axis=-1)

acc = accuracy_score(y_test, y_pred)
print(acc)
#%%
import matplotlib.pyplot as plt

# 테스트 샘플 선택
sample_index = 20  
x_sample = x_test_small[sample_index]  
y_true_label = y_test[sample_index]    

# 선택한 테스트 샘플 추론
y_pred_onehot = hypothesis(w, x_sample.reshape(1, -1), b)  
y_pred_label = np.argmax(y_pred_onehot, axis=-1)[0]       

# 테스트 샘플 및 결과 가시화
plt.figure()
plt.imshow(x_test[sample_index], cmap='gray')  # Use the original 28x28 image for visualization
plt.title(f"Predicted: {y_pred_label}, True: {y_true_label}")
plt.axis('off')
plt.show()



