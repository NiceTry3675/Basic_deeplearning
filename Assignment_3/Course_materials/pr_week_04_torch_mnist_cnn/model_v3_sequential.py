import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 첫 번째 Convolutional Layer
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        self.layers2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 1, 256),   
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 10),
            )
                

    def forward(self, x):
        x = x.view(-1, 1, 7, 7)        
        # Conv layers
        x = self.layers1(x)
        x = self.layers2(x)

        # Flatten
        x = x.view(-1, 64 * 1 * 1)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 64
    in_dim = 7*7
    x = torch.randn(batch_size, in_dim)
    
    model = CNN()
    output = model(x)

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # 모델 파라미터 및 연산량(MACs) 계산 pip install thop
    from thop import profile
    macs, params = profile(model, inputs=(x, ))
    print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    # torch.Size([8, 784]) -> torch.Size([8, 10])
