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


def fuse_bn(w, bn):
    """conv 가중치 w 뒤에 eval 모드 BN → 등가 (w', b')"""
    s = bn.weight / (bn.running_var + bn.eps).sqrt()
    return w * s.view(-1, *([1] * (w.ndim - 1))), bn.bias - bn.running_mean * s


class PRepConvBNAct(nn.Module):
    """학습 전용 구조 재매개변수화 (RepVGG + ACNet): 3x3 conv+BN 하나 대신
    3x3 / 1x1 / 1x3 / 3x1 가지(각자 BN) + 항등 BN(cin==cout일 때)을 더한 뒤 활성화.
    모든 가지가 선형(conv→eval BN)이라 학습 후 3x3 conv 하나 + bias로 정확히 합쳐짐 → 배포 모델은 기존과 동일 구조·크기."""
    SHAPES = {"k33": (3, 3), "k11": (1, 1), "k13": (1, 3), "k31": (3, 1)}

    def __init__(self, K, cin, cout, act, branches):
        super().__init__()
        self.K, self.cin, self.cout = K, cin, cout
        self.b, self.bn = nn.ModuleDict(), nn.ModuleDict()
        for name in ["k33"] + [b for b in branches if b in self.SHAPES and b != "k33"]:
            kh, kw = self.SHAPES[name]
            self.b[name] = nn.Conv2d(K * cin, K * cout, (kh, kw), 1, (kh // 2, kw // 2), groups=K, bias=False)
            self.bn[name] = nn.BatchNorm2d(K * cout)
        self.id_bn = nn.BatchNorm2d(K * cout) if ("id" in branches and cin == cout) else None
        self.act = act_layer(act)() if act else nn.Identity()

    def forward(self, x):
        y = sum(self.bn[k](c(x)) for k, c in self.b.items())
        if self.id_bn is not None:
            y = y + self.id_bn(x)
        return self.act(y)

    @torch.no_grad()
    def fused(self):
        dev = self.b["k33"].weight.device
        W = torch.zeros(self.K * self.cout, self.cin, 3, 3, device=dev)
        B = torch.zeros(self.K * self.cout, device=dev)
        for k, c in self.b.items():
            w, b = fuse_bn(c.weight, self.bn[k])
            kh, kw = w.shape[2:]
            W[:, :, 1 - kh // 2:2 + kh // 2, 1 - kw // 2:2 + kw // 2] += w
            B += b
        if self.id_bn is not None:  # grouped conv에서 출력 채널 o는 자기 그룹의 (o % cin)번 입력 채널을 그대로 통과
            eye = torch.zeros_like(W)
            o = torch.arange(self.K * self.cout, device=dev)
            eye[o, o % self.cin, 1, 1] = 1.0
            w, b = fuse_bn(eye, self.id_bn)
            W += w
            B += b
        return W, B


def make_conv(K, cin, cout, act, rep):
    return PRepConvBNAct(K, cin, cout, act, rep) if rep else PConvBNAct(K, cin, cout, 3, act=act)


class PResBlock(nn.Module):
    def __init__(self, K, cin, cout, act="relu", rep=None):
        super().__init__()
        self.c1 = make_conv(K, cin, cout, act, rep)
        self.c2 = make_conv(K, cout, cout, None, rep)
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

    def __init__(self, spec, K, act="relu", rep=None):
        """rep: 학습 전용 재매개변수화 가지 목록 (예: ["k11","k13","k31","id"]) — stride 1 3x3 conv에만 적용, to_plain()으로 합침"""
        super().__init__()
        self.spec, self.K, self.act, self.rep = spec, K, act, rep
        layers, c, ps = [], 1, 1  # ps: 분류 직전 pooling 크기 (G=1, P{n}=n)
        for tok in spec.split():
            if tok.startswith("c") and "_" in tok:
                k, cout = map(int, tok[1:].split("_"))
                if rep and k == 3:
                    layers.append(PRepConvBNAct(K, c, cout, act, rep))
                else:
                    layers.append(PConvBNAct(K, c, cout, k, 1, act=act))
                c = cout
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
                cout = int(tok[1:]); layers.append(PResBlock(K, c, cout, act, rep)); c = cout
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
    def to_plain(self):
        """재매개변수화 가지를 합쳐 일반 PackedTokenNet으로 변환 (rep=None이면 그대로 복사)
        합친 3x3 conv: weight=W, 뒤 BN: weight=1, bias=B, running_mean=0, running_var=1-eps → BN이 항등+bias가 됨"""
        dev = next(self.parameters()).device
        plain = PackedTokenNet(self.spec, self.K, act=self.act).to(dev)
        sd_r, sd_p = self.state_dict(), plain.state_dict()
        for key, v in sd_p.items():  # 재매개변수화되지 않은 모듈(shortcut, 분류기 등)은 키가 같음
            if key in sd_r and sd_r[key].shape == v.shape:
                v.copy_(sd_r[key])
        for name, mod in self.named_modules():
            if isinstance(mod, PRepConvBNAct):
                W, B = mod.fused()
                tgt = plain.get_submodule(name)
                conv, bn = tgt[0], tgt[1]
                conv.weight.copy_(W)
                bn.weight.fill_(1.0); bn.bias.copy_(B); bn.running_mean.zero_(); bn.running_var.fill_(1.0 - bn.eps)
        plain.train(self.training)
        return plain

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


if __name__ == "__main__" and "--rep-test" in __import__("sys").argv:
    # 재매개변수화 검증: 학습 그래프(eval) == 합친 일반 모델 == 추출한 TokenNet, 배포 모델 크기 동일
    from models import STUDENTS
    from common import count_complexity
    torch.manual_seed(0)
    for branches in [["k11", "k13", "k31", "id"], ["k11", "id"]]:
        rm = PackedTokenNet(STUDENTS["res_c"], K=4, act="gelu", rep=branches)
        for mod in rm.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.2)
        rm.train()
        with torch.no_grad():
            for _ in range(5):
                rm(torch.randn(64, 4, 28, 28))
        rm.eval()
        pm = rm.to_plain().eval()
        x = torch.randn(32, 4, 28, 28)
        with torch.no_grad():
            d1 = (rm(x) - pm(x)).abs().max().item()
            d2 = max((pm.extract(k).eval()(x[:, k:k + 1]) - rm(x)[:, k]).abs().max().item() for k in range(4))
        cx = count_complexity(pm.extract(0))
        ntr = sum(p.numel() for p in rm.parameters()) // 4
        same_keys = list(pm.extract(0).state_dict()) == list(TokenNet(STUDENTS["res_c"], act="gelu").state_dict())
        print(f"rep={branches}: |train-plain|={d1:.2e} |extract-train|={d2:.2e} 학습그래프 params/모델={ntr:,} "
              f"배포 params={cx['max_params']:,} MACs={cx['max_macs']:,} ok={cx['ok']} 키동일={same_keys}")
    raise SystemExit

if __name__ == "__main__":
    # 동작 검증: packed 출력 == 추출한 단일 모델 출력
    from models import STUDENTS
    torch.manual_seed(0)
    for name in ["vgg_a", "res_a", "ds_a", "mb_b", "c3_16 SE4 M c3_32 M c1_32 G D0.1 L10", "c3_16 M c3_24 RS24x3 M c3_32 G L10",
                 "c3_16 ds3_32 M ds3_48 ds3_64 M ds3_96 ds3_96 P2 L10", "c3_8 M c3_16 M c3_16 P3 D0.1 L10", "res_c"]:
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
