"""
토큰 문자열로 CNN을 조립하는 빌더 (구조 탐색용)

토큰 (공백 구분):
  c{k}_{C}     : Conv kxk (stride 1, same pad) + BN + Act      예) c3_16
  s{k}_{C}     : Conv kxk (stride 2) + BN + Act                예) s3_32
  M            : MaxPool 2x2
  A            : AvgPool 2x2
  dw{k}        : Depthwise Conv kxk + BN + Act (채널 유지)
  ds{k}_{C}    : Depthwise kxk + BN + Act → Pointwise 1x1 C + BN + Act
  RS{C}x{n}    : 가중치 공유 residual block (같은 conv를 n회 반복, BN은 반복별)
  R{C}         : Residual basic block (3x3-3x3, 채널 변경 시 1x1 shortcut)
  I{e}_{C}     : Inverted residual (1x1 expand×e → dw3x3 → 1x1 project, 같은 shape이면 skip)
  SE{r}        : Squeeze-Excitation (축소비 r)
  G            : Global Average Pooling (+flatten)
  P{n}         : n×n Adaptive Average Pooling (+flatten) — 공간 정보를 남기는 head (분류층 입력 C·n·n)
  F            : Flatten
  D{p}         : Dropout p
  L{n}         : Linear n (마지막 분류기)
"""
import torch
import torch.nn as nn


def act_layer(name):
    return {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "hswish": nn.Hardswish}[name]


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act="relu"):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(act_layer(act)())
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, act="relu"):
        super().__init__()
        self.c1 = ConvBNAct(cin, cout, 3, act=act)
        self.c2 = ConvBNAct(cout, cout, 3, act=None)
        self.sc = nn.Identity() if cin == cout else ConvBNAct(cin, cout, 1, act=None)
        self.act = act_layer(act)()

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.sc(x))


class InvRes(nn.Module):
    def __init__(self, cin, cout, e, act="relu"):
        super().__init__()
        h = cin * e
        layers = []
        if e != 1:
            layers.append(ConvBNAct(cin, h, 1, act=act))
        layers += [ConvBNAct(h, h, 3, groups=h, act=act), ConvBNAct(h, cout, 1, act=None)]
        self.f = nn.Sequential(*layers)
        self.skip = cin == cout

    def forward(self, x):
        y = self.f(x)
        return x + y if self.skip else y


class SharedRes(nn.Module):
    """가중치 공유 residual block: conv 가중치는 n회 재사용, BN은 반복마다 별도 (파라미터↑ 없이 깊이↑)"""

    def __init__(self, c, n, act="relu"):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(c) for _ in range(n)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(c) for _ in range(n)])
        self.act = act_layer(act)()

    def forward(self, x):
        for i in range(self.n):
            x = self.act(self.bn2[i](self.conv2(self.act(self.bn1[i](self.conv1(x))))) + x)
        return x


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        h = max(4, c // r)
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, h, 1), nn.ReLU(inplace=True),
                               nn.Conv2d(h, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class TokenNet(nn.Module):
    def __init__(self, spec: str, act="relu", in_ch=1):
        super().__init__()
        self.spec = spec
        layers = []
        c = in_ch
        flat = None
        for tok in spec.split():
            if tok.startswith("c") and "_" in tok:
                k, cout = map(int, tok[1:].split("_"))
                layers.append(ConvBNAct(c, cout, k, 1, act=act)); c = cout
            elif tok.startswith("s") and "_" in tok:
                k, cout = map(int, tok[1:].split("_"))
                layers.append(ConvBNAct(c, cout, k, 2, act=act)); c = cout
            elif tok == "M":
                layers.append(nn.MaxPool2d(2))
            elif tok == "A":
                layers.append(nn.AvgPool2d(2))
            elif tok.startswith("dw"):
                k = int(tok[2:])
                layers.append(ConvBNAct(c, c, k, groups=c, act=act))
            elif tok.startswith("ds"):
                k, cout = map(int, tok[2:].split("_"))
                layers.append(ConvBNAct(c, c, k, groups=c, act=act))
                layers.append(ConvBNAct(c, cout, 1, act=act)); c = cout
            elif tok.startswith("RS"):
                cc, n = map(int, tok[2:].split("x"))
                assert cc == c, "RS 블록은 입력 채널과 같아야 함"
                layers.append(SharedRes(c, n, act))
            elif tok.startswith("R"):
                cout = int(tok[1:])
                layers.append(ResBlock(c, cout, act)); c = cout
            elif tok.startswith("I"):
                e, cout = map(int, tok[1:].split("_"))
                layers.append(InvRes(c, cout, e, act)); c = cout
            elif tok.startswith("SE"):
                layers.append(SE(c, int(tok[2:])))
            elif tok == "G":
                layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]; flat = c
            elif tok.startswith("P"):
                n = int(tok[1:])
                layers += [nn.AdaptiveAvgPool2d(n), nn.Flatten()]; c = c * n * n; flat = c
            elif tok == "F":
                layers.append(nn.Flatten()); flat = "auto"
            elif tok.startswith("D"):
                layers.append(nn.Dropout(float(tok[1:])))
            elif tok.startswith("L"):
                n = int(tok[1:])
                if flat == "auto":
                    layers.append(nn.LazyLinear(n))
                else:
                    layers.append(nn.Linear(c, n))
                c = n
            else:
                raise ValueError(f"unknown token {tok}")
        self.net = nn.Sequential(*layers)
        if flat == "auto":  # LazyLinear 초기화
            with torch.no_grad():
                self.net(torch.zeros(1, in_ch, 28, 28))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- 후보 구조 모음
STUDENTS = {
    # --- plain VGG 계열
    "vgg_a": "c3_16 c3_24 M c3_32 c3_48 M c1_64 G L10",
    "vgg_f": "c3_12 c3_20 M c3_28 c3_32 M c3_40 G L10",
    "vgg_g": "c3_16 c3_16 M c3_24 c3_32 M c3_32 c3_24 G L10",
    "vgg_o": "c3_24 c3_24 M c3_32 c3_40 M c1_64 G L10",
    # --- residual 계열
    "res_a": "c3_16 M R24 M R32 c1_64 G L10",
    "res_b": "c3_12 c3_16 M R24 M R32 G L10",
    "res_c": "c3_16 c3_16 M R24 M R32 G L10",
    "res_d": "c3_16 R16 M R32 M c3_32 G L10",
    "res_c_d2": "c3_16 c3_16 M R24 M R32 G D0.2 L10",
    "res_c_lp": "c3_16 c3_16 M R24 R32 M G L10",  # 두 번째 MaxPool을 R32 뒤로 (R32가 14×14에서 동작)
    "res_c_k5": "c5_16 c3_16 M R24 M R32 G L10",  # 첫 conv만 5×5
    "res_a_d2": "c3_16 M R24 M R32 c1_64 G D0.2 L10",
    # --- depthwise separable 계열 (같은 파라미터로 더 넓은 채널)
    "ds_a": "c3_16 ds3_32 M ds3_48 ds3_64 M ds3_96 ds3_96 G L10",
    "ds_c": "c3_16 ds3_32 M ds3_64 ds3_64 M ds3_96 ds3_96 G L10",
    "ds_h": "c3_16 c3_24 M ds3_48 ds3_64 ds3_64 M ds3_80 ds3_80 G L10",
    "ds_i": "c3_24 c3_24 M ds3_48 ds3_64 M ds3_96 ds3_96 G L10",
    "ds_k": "c3_16 ds3_32 M ds3_48 ds3_64 M ds3_104 ds3_104 G L10",
    "ds_n": "c3_16 ds5_32 M ds5_48 ds5_64 M ds5_96 ds5_96 G L10",
    "ds_o": "c3_16 ds3_32 ds3_32 M ds3_48 ds3_64 M ds3_96 ds3_96 G L10",
    # --- 2×2 공간 head (GAP 대신 P2)
    "ds_a_p2": "c3_16 ds3_32 M ds3_48 ds3_64 M ds3_96 ds3_96 P2 L10",
    "ds_c_p2": "c3_16 ds3_32 M ds3_64 ds3_64 M ds3_96 ds3_96 P2 L10",
    "res_c30_p2": "c3_16 c3_16 M R24 M R30 P2 L10",
    # --- 예산 내 2-모델 앙상블 멤버 후보 (~15K, 두 개 합쳐 30K 이하)
    "e_res": "c3_12 c3_12 M R16 M R24 G L10",
    "e_ds2": "c3_16 ds3_24 M ds3_40 ds3_48 M ds3_64 ds3_72 G L10",
    "e_vgg": "c5_12 M c3_20 c3_24 M c3_32 G L10",
    # --- 가중치 공유 residual 계열 (파라미터는 그대로, 남는 MACs로 깊이 확보)
    "sh_a": "c3_16 M c3_24 RS24x3 M c3_48 G L10",
    "sh_a3": "c3_16 M c3_24 RS24x3 M c3_48 c1_64 G L10",
    "sh_b": "c3_12 M c3_28 RS28x3 M c3_32 G L10",
    "sh_c": "c3_12 M c3_24 M c3_32 RS32x4 G L10",
    "sh_c2": "c3_12 M c3_24 M c3_32 RS32x6 G L10",
    "sh_e": "c3_12 M c3_20 RS20x3 M c3_28 RS28x3 G L10",
    "sh_f": "c3_16 c3_16 M c3_24 RS24x3 M c3_40 G L10",
    # --- inverted residual (MobileNetV2) 계열
    "mb_b": "c3_16 I2_16 M I4_24 I3_24 M I3_40 I3_40 c1_64 G L10",
}

TEACHERS = {
    "t_vgg": "c3_64 c3_64 M c3_128 c3_128 M c3_256 c3_256 M G D0.3 L10",
    "t_res": "c3_64 R64 R64 M R128 R128 M R256 R256 G D0.3 L10",
    "t_wide": "c5_96 c3_96 M c3_192 c3_192 M c3_384 c3_384 G D0.4 L10",
}


def build(name_or_spec: str, act="relu"):
    spec = STUDENTS.get(name_or_spec) or TEACHERS.get(name_or_spec) or name_or_spec
    return TokenNet(spec, act=act)


if __name__ == "__main__":
    import sys
    from common import count_complexity
    names = sys.argv[1:] or list(STUDENTS)
    print(f"{'name':10s} {'params':>8s} {'thop_MACs':>11s} {'ptfl_MACs':>11s} ok  spec")
    for n in names:
        m = build(n)
        r = count_complexity(m)
        print(f"{n:10s} {r['max_params']:8d} {r.get('thop_macs', -1):11,d} {r.get('ptflops_macs', -1):11,d} "
              f"{'O' if r['ok'] else 'X'}   {m.spec}")
