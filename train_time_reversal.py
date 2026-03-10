"""
Train an MLP to classify single-scatter waveforms as forward vs. time-reversed.

Usage:
    python train_time_reversal.py [--epochs 50] [--batch_size 256] [--lr 1e-3]
"""
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class TimeReversalDataset(Dataset):
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            waveforms = f["waveforms"][:].astype(np.float32)  # (N, C, T)

        waveforms = waveforms.squeeze(1)  # (N, T)

        # Keep only middle 650 time samples
        T = waveforms.shape[1]
        start = (T - 650) // 2
        waveforms = waveforms[:, start:start + 650]

        # Normalize each waveform to unit max
        maxes = np.abs(waveforms).max(axis=1, keepdims=True).clip(min=1e-8)
        waveforms = waveforms / maxes

        n = waveforms.shape[0]
        forward = waveforms
        reversed_ = waveforms[:, ::-1].copy()

        self.x = np.concatenate([forward, reversed_], axis=0)
        self.y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)

        perm = np.random.permutation(len(self.y))
        self.x = self.x[perm]
        self.y = self.y[perm]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple = (256, 128, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = TimeReversalDataset(args.data)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Dataset: {n_total} samples ({n_train} train / {n_val} val)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    input_dim = dataset.x.shape[1]
    model = MLP(input_dim, hidden_dims=(256, 128, 64), dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = (logits > 0).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += x_batch.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                preds = (logits > 0).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += x_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f} | "
                f"Best Val Acc {best_val_acc:.4f}"
            )

    print(f"\nFinal results after {args.epochs} epochs:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val   Accuracy: {val_acc:.4f}")
    print(f"  Best  Val Acc:  {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on forward vs. time-reversed SS waveforms")
    parser.add_argument("--data", type=str, default="data/tritium_ss_single_node.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_every", type=int, default=5)
    args = parser.parse_args()
    train(args)
