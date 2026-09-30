"""
과제 3 전체 실험 실행
- 원본 28×28 입력(주 실험)과 7×7 다운사이징 입력(비교군)을 각각 3개 seed로 학습 → 테스트
- 추가 비교: 7×7을 4×4 블록 평균으로 만든 입력(7avg) — 해상도 vs 다운샘플링 방식의 효과 분리
- 일부만 다시 실행: python run_experiments.py --configs 7avg
- 하이퍼파라미터는 강의자료 train.py 기본값 그대로 (SGD lr=0.01, batch 64, 10 epochs)
- 대표 모델(seed 0)로 test.py 시각화와 inference.py 단일 샘플 추론 수행
"""
import os
import sys
import argparse
import subprocess

PY      = sys.executable
RESULTS = os.path.join("..", "results")
LOGS    = os.path.join(RESULTS, "logs")
MODELS  = os.path.join(RESULTS, "models")
FIGS    = os.path.join(RESULTS, "figures")
SEEDS   = [0, 1, 2]
DEVICE  = "cuda"
# tag: (in_size, 추가 인자, 단일 샘플 추론 여부)
CONFIGS = {
    "28":   (28, [], True),
    "7":    (7,  ["--use_small"], True),
    "7avg": (7,  ["--use_small", "--small_method", "avg"], False),
}

def run(cmd, log_file):
    print(">>", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    print(out)
    with open(log_file, "w") as f:
        f.write(out)

def main(configs):
    for d in (LOGS, MODELS, FIGS):
        os.makedirs(d, exist_ok=True)

    for name in configs:
        size, flag, do_infer = CONFIGS[name]
        for seed in SEEDS:
            tag   = f"cnn_{name}_seed{seed}"
            model = os.path.join(MODELS, f"{tag}.pth")
            run([PY, "train.py", *flag, "--device", DEVICE, "--seed", str(seed),
                 "--save_path", model, "--log_path", os.path.join(LOGS, f"train_{tag}.json")],
                os.path.join(LOGS, f"train_{tag}.txt"))
            test_dir = os.path.join(LOGS, f"test_{tag}")
            run([PY, "test.py", *flag, "--device", DEVICE, "--model_path", model, "--save_dir", test_dir],
                os.path.join(LOGS, f"test_{tag}.txt"))

        if not do_infer:
            continue
        # 대표 모델(seed 0) 단일 샘플 추론
        model = os.path.join(MODELS, f"cnn_{name}_seed0.pth")
        for idx in (0, 12, 247):
            run([PY, "inference.py", *flag, "--device", DEVICE, "--model_path", model,
                 "--sample_idx", str(idx), "--save_dir", FIGS],
                os.path.join(LOGS, f"inference_{size}_idx{idx}.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    main(parser.parse_args().configs)
