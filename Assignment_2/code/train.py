"""Part 3: 앙상블 학습 (강의자료 pr_14 train_optimizer_scheduler_ensemble.py 기반).

Part 1/2에서 선택된 optimizer/scheduler(configs.BEST)로 두 가지 앙상블을 학습한다.
- K-Fold(k=5) 앙상블: 분할 모델 k개 → models/mlp_fold{0..4}.pth (강의 방식)
- 시드 앙상블: 전체 60,000장 × 서로 다른 시드 n개 → models/mlp_full_seed{0..4}.pth
  (seed0 모델이 단일 모델 baseline을 겸함)

사용법:
    python train.py                 # BEST 구성으로 전체 학습
    python train.py --epochs 2      # 빠른 동작 확인
"""
import argparse
import os

from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
import torch

from configs import BEST, COMMON, OPTIMIZERS, make_scheduler
from data_loader import load_mnist
from engine import train_model
from model import MLP
from utils import get_device, save_json, set_seed

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CODE_DIR, "models")
RESULTS_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "results", "part3"))


def train_one(model, train_loader, val_loader, args, device):
    criterion = CrossEntropyLoss()
    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), args.lr)
    scheduler, mode = make_scheduler(args.scheduler, optimizer, args.lr,
                                     args.epochs, len(train_loader))
    return train_model(model, train_loader, val_loader, criterion, optimizer,
                       scheduler=scheduler, scheduler_mode=mode,
                       epochs=args.epochs, device=device)


def main(args):
    device = get_device(args.device)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, _, _, in_dim, _, _ = load_mnist(valDB_portion=0.0)
    print(f"구성: optimizer={args.optimizer}, lr={args.lr:g}, scheduler={args.scheduler}, "
          f"k={args.k_splits}, epochs={args.epochs}, batch_size={args.batch_size}")

    # ----- K-Fold 앙상블 모델 학습 -----
    kfold = KFold(n_splits=args.k_splits, shuffle=True, random_state=42)
    fold_summary = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_ds)))):
        set_seed(args.seed + fold)
        train_loader = DataLoader(Subset(train_ds, train_idx.tolist()),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(train_ds, val_idx.tolist()),
                                batch_size=256, shuffle=False)

        print(f"\n===== Fold {fold} (train {len(train_idx)} / val {len(val_idx)}) =====",
              flush=True)
        model = MLP(in_dim).to(device)
        history = train_one(model, train_loader, val_loader, args, device)

        model_path = os.path.join(MODEL_DIR, f"mlp_fold{fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        save_json({"name": f"fold{fold}", "config": vars(args), "history": history},
                  os.path.join(RESULTS_DIR, f"train_fold{fold}.json"))
        fold_summary.append({"fold": fold,
                             "final_val_acc": history[-1]["val_acc"],
                             "best_val_acc": max(h["val_acc"] for h in history)})

    # ----- 시드 앙상블: 전체 데이터 × 서로 다른 시드 -----
    for i in range(args.n_full):
        set_seed(args.seed + 100 + i)
        print(f"\n===== Full-train 모델 {i} (train {len(train_ds)}, seed {args.seed + 100 + i}) =====",
              flush=True)
        full_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        model = MLP(in_dim).to(device)
        history = train_one(model, full_loader, None, args, device)
        model_path = os.path.join(MODEL_DIR, f"mlp_full_seed{i}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        save_json({"name": f"full_seed{i}", "config": vars(args), "history": history},
                  os.path.join(RESULTS_DIR, f"train_full_seed{i}.json"))

    save_json({"config": vars(args), "folds": fold_summary},
              os.path.join(RESULTS_DIR, "train_summary.json"))
    print("\n----- Fold별 검증 정확도 -----")
    for s in fold_summary:
        print(f"Fold {s['fold']}: final {s['final_val_acc']:.2f}% / best {s['best_val_acc']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold 앙상블 학습")
    parser.add_argument("--k_splits", type=int, default=5)
    parser.add_argument("--n_full", type=int, default=5,
                        help="시드 앙상블용 full-train 모델 수")
    parser.add_argument("--epochs", type=int, default=COMMON["epochs"])
    parser.add_argument("--batch_size", type=int, default=COMMON["batch_size"])
    parser.add_argument("--optimizer", type=str, default=BEST["optimizer"])
    parser.add_argument("--lr", type=float, default=BEST["lr"])
    parser.add_argument("--scheduler", type=str, default=BEST["scheduler"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
