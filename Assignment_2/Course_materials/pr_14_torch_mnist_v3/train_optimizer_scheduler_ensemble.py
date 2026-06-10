import argparse
import torch
from torch.optim import SGD, Adagrad, RMSprop, Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, CyclicLR
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from data_loader import load_mnist
from model_v2 import MLP

#%% Training Function (수정 없음)
def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs: int=10, device=None):

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_samples = len(train_loader.sampler)
        total_val_samples   = len(val_loader.sampler)
                
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

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch:2d}] LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")

#%% Main - KFold Ensemble
def main(args):
    # 1) 전체 Train 데이터셋만 로드 (valDB_portion=0.0)
    train_ds, _, _, in_dim, x_test_raw, y_test = load_mnist( use_small=args.use_small,valDB_portion=0.0)

    # 2) KFold 준비
    k_num = 4
    kfold = KFold(n_splits=k_num, shuffle=True, random_state=42)

    device = torch.device(args.device) if args.device else torch.device('cuda')
    criterion = CrossEntropyLoss()

    # 3) Fold별로 모델 학습 및 저장
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_ds)):
        print(f"\n========== Fold {fold+1}/{k_num} ==========")

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler   = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader   = DataLoader(train_ds, batch_size=args.batch_size, sampler=val_subsampler)

        model = MLP(in_dim)
        if device is not None:
            model.to(device)
            
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-3)
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.75) 

        train( model, train_loader, val_loader, criterion, optimizer,
            scheduler=scheduler, epochs=args.epochs, device=device )

        model_path = f"{args.save_path}_fold{fold}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model for fold {fold} saved to {model_path}")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KFold Ensemble Train MLP on MNIST")
    parser.add_argument("--epochs",      type=int,   default=10,    help="number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=64,    help="mini-batch size for training")
    parser.add_argument("--lr",          type=float, default=1e-2,  help="learning rate for optimizer")
    parser.add_argument("--use_small",   type=bool,  default=False,  help="use 7x7 down-sampled input")
    parser.add_argument("--no_shuffle",  type=bool,  default=False, help="disable data shuffling")
    parser.add_argument("--device",      type=str,   default="cuda",  help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--save_path",   type=str,   default="mlp_mnist.pth", help="file path to save trained model")
    args = parser.parse_args()

    main(args)
