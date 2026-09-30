import argparse
import json
import time
import torch
from torch.optim    import SGD
from torch.nn       import CrossEntropyLoss
from torch.utils.data import  DataLoader
from data_loader import load_mnist
#from model_v2       import MLP
from model_v3       import CNN
#%%
def train(model, train_loader, val_loader, criterion, optimizer, epochs: int=10, device=None):
    # model.train()
    total_train_samples = len(train_loader.dataset)
    total_val_samples   = len(val_loader.dataset)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        train_loss_sum = 0.0
        train_correct  = 0

        for data, target in train_loader:
            if device is not None:
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = train_loss_sum / len(train_loader)
        train_acc  = 100.0 * train_correct / total_train_samples

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        with torch.no_grad():
            for data, target in val_loader:
                if device is not None:
                    data, target = data.to(device), target.to(device)

                output = model(data)
                loss   = criterion(output, target)
                val_loss_sum += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc  = 100.0 * val_correct / total_val_samples

        # Print metrics for each epoch
        print(f"[Epoch {epoch:2d}] "
              f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
        for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[k].append(v)
    return history
#%%
def main(args):
    torch.manual_seed(args.seed)

    # 1) DataLoader 준비
    train_ds, val_ds, _, in_dim, x_test_raw, y_test = load_mnist(use_small=args.use_small, valDB_portion = 0.1, seed=args.seed,
                                                                  small_method=args.small_method)
    in_size = int(in_dim ** 0.5)   # 784 -> 28, 49 -> 7
    print(f"Input: {in_size}x{in_size} (in_dim={in_dim}), train={len(train_ds)}, val={len(val_ds)}")
    
    train_loader = DataLoader(train_ds,  batch_size=args.batch_size,   shuffle=not args.no_shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,   shuffle=False)
    
    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else None
    model  = CNN(in_size)
    #model = MLP(in_dim)
    if device is not None:
        model.to(device)
        
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr)
    
    # 3) 학습
    start = time.perf_counter()
    history = train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=args.epochs,
                device=device
            )

    elapsed = time.perf_counter() - start
    print(f"Training time: {elapsed:.1f}s")

    # 4) 학습된 모델 저장
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    # 5) 학습 곡선 기록 저장 (보고서 그래프용)
    if args.log_path:
        log = {"args": vars(args), "in_size": in_size, "train_time_sec": elapsed, **history}
        with open(args.log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Log saved to {args.log_path}")
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mnist_model on MNIST")
    parser.add_argument("--epochs",      type=int,   default=10,    help="number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=64,    help="mini-batch size for training")
    parser.add_argument("--lr",          type=float, default=1e-2,  help="learning rate for optimizer")
    # type=bool은 "--use_small False"도 bool("False")=True가 되므로 플래그(store_true)로 변경
    # 기본값: 원본 28x28 입력 사용
    parser.add_argument("--use_small",   action="store_true",  help="use 7x7 down-sampled input (default: original 28x28)")
    parser.add_argument("--small_method", type=str, default="stride", choices=["stride", "avg"], help="7x7 down-sampling: stride(::4, lecture) or avg(4x4 mean)")
    parser.add_argument("--no_shuffle",  action="store_true",  help="disable data shuffling")
    parser.add_argument("--device",      type=str,   default=None,  help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--seed",        type=int,   default=0,     help="random seed (weight init, shuffling, train/val split)")
    parser.add_argument("--save_path",   type=str,   default="cnn_mnist_28.pth", help="file path to save trained model")
    parser.add_argument("--log_path",    type=str,   default=None,  help="json file path to save training history")
    args = parser.parse_args()

    main(args)

