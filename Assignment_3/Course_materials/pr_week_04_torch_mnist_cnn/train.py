import argparse
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
#%%
def main(args):
    # 1) DataLoader 준비
    train_ds, val_ds, _, in_dim, x_test_raw, y_test = load_mnist(use_small=args.use_small, valDB_portion = 0.1)
    
    train_loader = DataLoader(train_ds,  batch_size=args.batch_size,   shuffle=not args.no_shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,   shuffle=False)
    
    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else None
    model  = CNN()
    #model = MLP(in_dim)
    if device is not None:
        model.to(device)
        
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr)
    
    # 3) 학습
    train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=args.epochs,
                device=device
            )

    # 4) 학습된 모델 저장
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mnist_model on MNIST")
    parser.add_argument("--epochs",      type=int,   default=10,    help="number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=64,    help="mini-batch size for training")
    parser.add_argument("--lr",          type=float, default=1e-2,  help="learning rate for optimizer")
    parser.add_argument("--use_small",   type=bool,  default=True,  help="use 7x7 down-sampled input")
    parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    parser.add_argument("--device",      type=str,   default=None,  help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--save_path",   type=str,   default="cnn_mnist_v0.pth", help="file path to save trained model")
    args = parser.parse_args()

    main(args)

