"""
보고서의 '알아두기' 항목에서 인용하는 수치를 재현하는 보조 측정 스크립트
  python extra_checks.py > ../results/logs/extra_checks.txt
(1 epoch 학습·시간 측정이 포함되어 GPU 기준 약 1분)
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import load_mnist, downsample
from model_v3 import CNN

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def section(title):
    print(f"\n===== {title} =====")

#%% 1) argparse type=bool
section("argparse type=bool")
p = argparse.ArgumentParser()
p.add_argument("--flag", type=bool, default=False)
for v in ["False", "0", ""]:
    print(f'type=bool, --flag "{v}" ->', p.parse_args(["--flag", v]).flag)
p = argparse.ArgumentParser()
p.add_argument("--use_small", action=argparse.BooleanOptionalAction, default=False)
print("BooleanOptionalAction: [] ->", p.parse_args([]).use_small,
      "| --use_small ->", p.parse_args(["--use_small"]).use_small,
      "| --no-use_small ->", p.parse_args(["--no-use_small"]).use_small)

#%% 2) 표준화 통계, 다운샘플 후 남는 획 픽셀 수
section("standardization / remaining stroke pixels")
raw = np.load("mnist.npz")
xtr, xte = raw["x_train"].astype(np.float64), raw["x_test"].astype(np.float64)
print(f"train mean/std = {xtr.mean():.2f} / {xtr.std():.2f}")
print(f"test  mean/std = {xte.mean():.2f} / {xte.std():.2f}")
print(f"train[:54000] mean/std = {xtr[:54000].mean():.2f} / {xtr[:54000].std():.2f}")
for name, x in (("28x28", xtr), ("7x7 stride", downsample(xtr, "stride")), ("7x7 avg", downsample(xtr, "avg"))):
    print(f"non-zero pixels per image ({name}): {(x > 0).sum(axis=(1, 2)).mean():.1f} / {x.shape[1] * x.shape[2]}")

#%% 3) 파라미터·버퍼, thop과 BatchNorm
section("parameters / buffers / thop")
m = CNN(28)
print("parameter tensors:", len(list(m.parameters())), "| state_dict keys:", len(m.state_dict()))
print("buffers:", [n for n, _ in m.named_buffers()])
print("buffer elements:", sum(b.numel() for b in m.buffers()))
from thop import profile
for size in (28, 7):
    mm = CNN(size).eval()
    macs, _ = profile(mm, inputs=(torch.randn(1, size * size),), verbose=False)
    s1, s2 = size, size // 2
    layer = 1 * 32 * 9 * s1 * s1 + 32 * 64 * 9 * s2 * s2 + mm.flat_dim * 256 + 256 * 10
    bn = 4 * (32 * s1 * s1 + 64 * s2 * s2 + 256)
    print(f"{size}x{size}: thop={int(macs):,}  conv+fc={layer:,}  diff={int(macs) - layer:,}  4*BN elements={bn:,}")

#%% 4) BatchNorm: train 모드 배치 1, no_grad와 running 통계
section("train/eval mode")
m = CNN(28)
try:
    m.train()(torch.randn(1, 784))
except ValueError as e:
    print("train mode, batch 1 -> ValueError:", e)
print("eval mode, batch 1 ->", tuple(m.eval()(torch.randn(1, 784)).shape))
m.train()
before = m.bn1.running_mean.clone()
with torch.no_grad():
    m(torch.randn(8, 784))
print("train mode + no_grad changes bn1.running_mean:", not torch.equal(before, m.bn1.running_mean))

#%% 5) state_dict 로드: 구조가 다르면 오류
section("state_dict")
try:
    CNN(7).load_state_dict(torch.load("../results/models/cnn_28_seed0.pth", map_location="cpu"))
except RuntimeError as e:
    print("load 28x28 weights into CNN(7) ->", str(e).splitlines()[1].strip())

#%% 6) 확신도(softmax)와 오답
section("confidence of wrong predictions (28x28, seed 0)")
_, _, test_ds, _, _, y_test = load_mnist(use_small=False)
m = CNN(28).to(dev)
m.load_state_dict(torch.load("../results/models/cnn_28_seed0.pth", map_location=dev))
m.eval()
with torch.no_grad():
    prob = torch.cat([torch.softmax(m(x.to(dev)), 1).cpu() for x, _ in DataLoader(test_ds, batch_size=1000)])
conf, pred = prob.max(1)
wrong = pred.numpy() != y_test
print(f"wrong={wrong.sum()}, wrong with confidence >= 90%: {(conf.numpy()[wrong] >= 0.9).sum()}")

#%% 7) 1 epoch 후 학습 정확도 분해 (누적 vs eval 재평가)
section("epoch-1 train accuracy decomposition (seed 0)")
def accuracy(model, loader, train_mode):
    model.train(train_mode)
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
    return 100.0 * correct / len(loader.dataset)

for name, kw in (("28x28", dict(use_small=False)), ("7x7 stride", dict(use_small=True))):
    torch.manual_seed(0)
    tr, va, _, in_dim, _, _ = load_mnist(valDB_portion=0.1, seed=0, **kw)
    model = CNN(int(in_dim ** 0.5)).to(dev)
    opt, crit = torch.optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
    loader = DataLoader(tr, batch_size=64, shuffle=True)
    model.train()
    correct = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        crit(out, y).backward()
        opt.step()
        correct += (out.argmax(1) == y).sum().item()
    eval_loader = DataLoader(tr, batch_size=1000)
    print(f"{name}: running(train, during epoch)={100.0 * correct / len(tr):.2f}  "
          f"re-eval train set(eval mode)={accuracy(model, eval_loader, False):.2f}  "
          f"re-eval train set(train mode)={accuracy(model, eval_loader, True):.2f}  "
          f"val(eval mode)={accuracy(model, DataLoader(va, batch_size=1000), False):.2f}")

#%% 8) 스텝 시간: 배치 크기에 따른 연산량 차이
section("training step time (CUDA events)")
if dev.type == "cuda":
    for bs in (64, 1024):
        for size in (28, 7):
            model = CNN(size).to(dev).train()
            opt, crit = torch.optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
            x = torch.randn(bs, size * size, device=dev)
            y = torch.randint(0, 10, (bs,), device=dev)
            for _ in range(20):
                opt.zero_grad(); crit(model(x), y).backward(); opt.step()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200):
                opt.zero_grad(); crit(model(x), y).backward(); opt.step()
            end.record(); torch.cuda.synchronize()
            print(f"batch {bs:4d}, {size}x{size}: {start.elapsed_time(end) / 200:.2f} ms/step")
    tr, _, _, _, _, _ = load_mnist(valDB_portion=0.1)
    t0 = time.perf_counter()
    for x, y in DataLoader(tr, batch_size=64, shuffle=True):
        x, y = x.to(dev), y.to(dev)
    torch.cuda.synchronize()
    print(f"one epoch of data loading + transfer only (28x28, batch 64): {time.perf_counter() - t0:.2f} s")
print("\nperf_counter monotonic:", time.get_clock_info("perf_counter").monotonic,
      "| time.time adjustable:", time.get_clock_info("time").adjustable)
