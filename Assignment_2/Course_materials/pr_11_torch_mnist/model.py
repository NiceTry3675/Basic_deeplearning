import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,     256)
        self.fc3 = nn.Linear(256,     10)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3
        # return self.fc3(x2)

#%%
if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 8
    in_dim = 784  # 예: 28x28 이미지를 펼친 크기
    x = torch.randn(batch_size, in_dim)
    model = MLP(in_dim)
    output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    # torch.Size([8, 784]) -> torch.Size([8, 10])