import argparse
import torch
from torch.optim    import SGD
from torch.nn       import CrossEntropyLoss

from data_loader import load_mnist
from model       import MLP
#%%
def train(model, train_loader, criterion, optimizer, epochs: int=10, device=None):
    model.train()
    total_samples = len(train_loader.dataset)

    for epoch in range(1, epochs+1):
        correct = 0
        for data, target in train_loader:
            if device is not None:
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        acc = 100.0 * correct / total_samples
        print(f"[Epoch {epoch:2d}] Loss: {loss.item():.6f}, Accuracy: {acc:.2f}%")
#%%
def main(args):
    # 1) DataLoader 준비
    train_loader, test_loader, in_dim, x_test_raw, y_test = load_mnist(
        use_small=args.use_small,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle
    )

    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else torch.device('cuda')
    model     = MLP(in_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr)

    # 3) 학습
    train(
        model,
        train_loader,
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
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs",      type=int,   default=10,    help="number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=64,    help="mini-batch size for training")
    parser.add_argument("--lr",          type=float, default=1e-2,  help="learning rate for optimizer")
    parser.add_argument("--use_small",   type=bool,  default=False,  help="use 7x7 down-sampled input")
    parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    parser.add_argument("--device",      type=str,   default="cuda",  help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--save_path",   type=str,   default="mlp_mnist.pth", help="file path to save trained model")
    args = parser.parse_args()

    main(args)

