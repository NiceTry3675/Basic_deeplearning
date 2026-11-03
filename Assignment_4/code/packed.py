"""
Packed 학습: 같은 구조의 독립 모델 K개를 grouped conv로 하나의 네트워크에 묶음.
- 입력 (B, K, 28, 28): 모델 k는 채널 k의 이미지만 봄 → 모델마다 다른 미니배치/초기화 = 완전히 독립 학습
- 모든 층이 채널 방향으로 K개 블록으로 분리 (groups=K), BN도 채널별이라 독립
- extract(k) 로 일반 TokenNet 가중치로 변환 (제출용 단일 모델)
"""
import torch
import torch.nn as nn

from models import TokenNet, act_layer


class PConvBNAct(nn.Sequential):
    def __init__(self, K, cin, cout, k=3, s=1, depthwise=False, act="relu"):
        g = K * cin if depthwise else K
        layers = [nn.Conv2d(K * cin, K * cout, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(K * cout)]
        if act:
            layers.append(act_layer(act)())
        super().__init__(*layers)


class PResBlock(nn.Module):
    def __init__(self, K, cin, cout, act="relu"):
        super().__init__()
        self.c1 = PConvBNAct(K, cin, cout, 3, act=act)
        self.c2 = PConvBNAct(K, cout, cout, 3, act=None)
        self.sc = nn.Identity() if cin == cout else PConvBNAct(K, cin, cout, 1, act=None)
        self.act = act_layer(act)()

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.sc(x))


class PInvRes(nn.Module):
    def __init__(self, K, cin, cout, e, act="relu"):
        super().__init__()
        h = cin * e
        layers = []
        if e != 1:
            layers.append(PConvBNAct(K, cin, h, 1, act=act))
        layers += [PConvBNAct(K, h, h, 3, depthwise=True, act=act), PConvBNAct(K, h, cout, 1, act=None)]
        self.f = nn.Sequential(*layers)
        self.skip = cin == cout

    def forward(self, x):
        y = self.f(x)
        return x + y if self.skip else y


class PSharedRes(nn.Module):
    def __init__(self, K, c, n, act="relu"):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(K * c, K * c, 3, 1, 1, groups=K, bias=False)
        self.conv2 = nn.Conv2d(K * c, K * c, 3, 1, 1, groups=K, bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(K * c) for _ in range(n)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(K * c) for _ in range(n)])
        self.act = act_layer(act)()

    def forward(self, x):
        for i in range(self.n):
            x = self.act(self.bn2[i](self.conv2(self.act(self.bn1[i](self.conv1(x))))) + x)
        return x


class PSE(nn.Module):
    def __init__(self, K, c, r=4):
        super().__init__()
        h = max(4, c // r)
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(K * c, K * h, 1, groups=K), nn.ReLU(inplace=True),
                               nn.Conv2d(K * h, K * c, 1, groups=K), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class PackedTokenNet(nn.Module):
    """TokenNet과 동일한 모듈 이름/순서를 유지 → state_dict 키가 1:1 대응"""

    def __init__(self, spec, K, act="relu"):
        super().__init__()
        self.spec, self.K, self.act = spec, K, act
        layers, c, ps = [], 1, 1  # ps: 분류 직전 pooling 크기 (G=1, P{n}=n)
        for tok in spec.split():
            if tok.startswith("c") and "_" in tok:
                k, cout = map(int, tok[1:].split("_"))
                layers.append(PConvBNAct(K, c, cout, k, 1, act=act)); c = cout
            elif tok.startswith("s") and "_" in tok:
                k, cout = map(int, tok[1:].split("_"))
                layers.append(PConvBNAct(K, c, cout, k, 2, act=act)); c = cout
            elif tok == "M":
                layers.append(nn.MaxPool2d(2))
            elif tok == "A":
                layers.append(nn.AvgPool2d(2))
            elif tok.startswith("dw"):
                layers.append(PConvBNAct(K, c, c, int(tok[2:]), depthwise=True, act=act))
            elif tok.startswith("ds"):
                k, cout = map(int, tok[2:].split("_"))
                layers.append(PConvBNAct(K, c, c, k, depthwise=True, act=act))
                layers.append(PConvBNAct(K, c, cout, 1, act=act)); c = cout
            elif tok.startswith("RS"):
                cc, n = map(int, tok[2:].split("x")); layers.append(PSharedRes(K, c, n, act))
            elif tok.startswith("R"):
                cout = int(tok[1:]); layers.append(PResBlock(K, c, cout, act)); c = cout
            elif tok.startswith("I"):
                e, cout = map(int, tok[1:].split("_")); layers.append(PInvRes(K, c, cout, e, act)); c = cout
            elif tok.startswith("SE"):
                layers.append(PSE(K, c, int(tok[2:])))
            elif tok == "G":
                layers += [nn.AdaptiveAvgPool2d(1), nn.Identity()]  # TokenNet: [AvgPool, Flatten]
            elif tok.startswith("P"):
                ps = int(tok[1:]); layers += [nn.AdaptiveAvgPool2d(ps), nn.Identity()]  # TokenNet: [Pool, Flatten]
            elif tok.startswith("D"):
                layers.append(nn.Dropout(float(tok[1:])))
            elif tok.startswith("L"):
                # n×n 특징을 펼쳐 Linear에 넣는 것 = kernel n인 grouped conv (extract 시 reshape로 1:1 대응)
                n = int(tok[1:]); layers.append(nn.Conv2d(K * c, K * n, ps, groups=K)); c = n
            else:
                raise ValueError(f"packed: unsupported token {tok}")
        self.net = nn.Sequential(*layers)
        self.n_out = c

    def forward(self, x):
        # x: (B, K, H, W) 또는 (B, 1, H, W) → 모든 모델에 같은 입력
        if x.shape[1] == 1 and self.K > 1:
            x = x.expand(-1, self.K, -1, -1)
        y = self.net(x)
        return y.view(y.shape[0], self.K, self.n_out)

    @torch.no_grad()
    def extract(self, k):
        """k번째 모델을 일반 TokenNet으로 추출"""
        m = TokenNet(self.spec, act=self.act)
        sd_p = self.state_dict()
        sd = {}
        for key, ref in m.state_dict().items():
            v = sd_p[key]
            if v.ndim == 0:  # num_batches_tracked
                sd[key] = v.clone()
            else:
                sd[key] = v.chunk(self.K, 0)[k].reshape(ref.shape).clone()
        m.load_state_dict(sd)
        return m


if __name__ == "__main__":
    # 동작 검증: packed 출력 == 추출한 단일 모델 출력
    from models import STUDENTS
    torch.manual_seed(0)
    for name in ["vgg_a", "res_a", "ds_a", "mb_b", "c3_16 SE4 M c3_32 M c1_32 G D0.1 L10", "c3_16 M c3_24 RS24x3 M c3_32 G L10",
                 "c3_16 ds3_32 M ds3_48 ds3_64 M ds3_96 ds3_96 P2 L10", "c3_8 M c3_16 M c3_16 P3 D0.1 L10"]:
        spec = STUDENTS.get(name, name)
        pm = PackedTokenNet(spec, K=4).eval()
        # BN running stats 랜덤화
        for mod in pm.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(); mod.running_var.uniform_(0.5, 2); mod.weight.data.normal_()
        x = torch.randn(3, 4, 28, 28)
        yp = pm(x)
        errs = [(pm.extract(k).eval()(x[:, k:k + 1]) - yp[:, k]).abs().max().item() for k in range(4)]
        print(f"{name:40s} max|diff| = {max(errs):.2e}")
