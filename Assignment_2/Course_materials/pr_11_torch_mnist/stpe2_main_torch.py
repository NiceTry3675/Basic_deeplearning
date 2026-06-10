import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
mnist = np.load('mnist.npz')

x_train = (mnist['x_train'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
y_train = mnist['y_train']
x_test = (mnist['x_test'] - np.mean(mnist['x_train'])) / np.std(mnist['x_train'])
y_test = mnist['y_test']
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

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
# print("Train:", x_train_in.shape, y_train_oh.shape)
# print("Test: ", x_test_in.shape,  y_test_oh.shape)
#%%
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.fc3 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
#%%    
from torch.optim import SGD

model = MLP(in_dim)
criterion = nn.CrossEntropyLoss()

opti = SGD(model.parameters(), lr=1e-2)


from torch.utils.data import TensorDataset, DataLoader

# TensorDataset을 사용하여 데이터셋 생성
train_dataset = TensorDataset(torch.FloatTensor(x_train_in), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(x_test_in), torch.LongTensor(y_test))

# DataLoader를 사용하여 미니배치 생성
batch_size = 64
shuffle = True
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epochs = 10

losses = []
for epoch in range(epochs):
    correct = 0
    for data, target in train_dataloader:

        data = data.view(-1, in_dim)

        output = model(data)
        loss = criterion(output, target)

        opti.zero_grad()
        loss.backward()
        opti.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100.0 * correct / len(train_dataloader.dataset)
    print(f"[Epoch {epoch+1:2d}] Loss: {loss:.6f}, Accuracy: {acc:.2f}%")


#%% 테스트 및 예측 저장
model.eval()
correct = 0
y_pred_lbl = []
with torch.no_grad():
    logits = model(torch.FloatTensor(x_test_in))
    y_pred = logits.argmax(dim=1).numpy()

test_acc = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy: {test_acc:.2f}%")

#%% 샘플 시각화
plt.figure()
idx      = 20
orig_img = x_test[idx]             # 원본 28×28 이미지
plt.imshow(orig_img, cmap='gray')
plt.title(f"Pred: {y_pred[idx]}, True: {y_test[idx]}")
plt.axis('off')
plt.show()

