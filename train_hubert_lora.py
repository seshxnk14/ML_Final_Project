import argparse
import math
import os

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from losses import SupConLoss
from mod1_hubert_lora import DEFAULT_HUBERT, HubertLoRAEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str, required=True)
    parser.add_argument("--h5", type=str, default=None,
                        help="Path to train_hubert.h5; defaults to <traindir>/train_hubert.h5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.06)
    parser.add_argument("--method", type=str, default="scl", choices=["scl", "ssl"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default=DEFAULT_HUBERT)
    parser.add_argument("--noise_std", type=float, default=0.005)
    parser.add_argument("--time_mask_pct", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Where to save the checkpoint; "
                             "defaults to <traindir>/../model_hubert/")
    return parser.parse_args()


def random_time_mask(x: torch.Tensor, max_pct: float = 0.1) -> torch.Tensor:
    """Zero a random time span; mask length is always < T."""
    if max_pct <= 0:
        return x
    out = x.clone()
    B, T = out.shape
    if T < 3:
        return out
    max_len = int(T * max_pct)
    max_len = max(1, min(max_len, T - 1))
    for b in range(B):
        L = torch.randint(1, max_len + 1, (1,)).item()
        s = torch.randint(0, T - L + 1, (1,)).item()
        out[b, s:s + L] = 0.0
    return out


def add_noise(x: torch.Tensor, std: float = 0.005) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def make_two_views(x: torch.Tensor, args):
    v1 = add_noise(random_time_mask(x, args.time_mask_pct), args.noise_std)
    v2 = add_noise(random_time_mask(x, args.time_mask_pct), args.noise_std)
    return v1, v2


def adjust_lr(optimizer, init_lr: float, epoch: int, tot_epochs: int):
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / tot_epochs))
    for g in optimizer.param_groups:
        g["lr"] = cur_lr


def main():
    args = parse_args()
    h5_path = args.h5 or os.path.join(args.traindir, "train_hubert.h5")
    print(f"loading dataset from {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        X = hf["data"][:]
        Y = hf["label"][:]
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    train_ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y.squeeze(), dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    encoder = HubertLoRAEncoder(
        method=args.method,
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(args.device)

    print(
        f"trainable params: {encoder.num_trainable_parameters():,} "
        f"/ {encoder.num_total_parameters():,}"
    )

    loss_fn = SupConLoss(temperature=args.tau, device=args.device)
    optim = torch.optim.AdamW(
        encoder.trainable_parameters(), lr=args.lr, weight_decay=args.wd
    )

    ckpt_dir = args.ckpt_dir or os.path.join(args.traindir, "../model_hubert/")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")

    encoder.train()
    for epoch in range(1, args.epochs + 1):
        adjust_lr(optim, args.lr, epoch, args.epochs + 1)
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            x1, x2 = make_two_views(x, args)

            optim.zero_grad()
            _, p1 = encoder(x1)
            _, p2 = encoder(x2)

            if args.method == "ssl":
                loss = loss_fn(p1, p2)
            else:
                loss = loss_fn(p1, p2, y)

            loss.backward()
            optim.step()
            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))
        print(f"epoch {epoch}: loss={tr_loss:.4f}")

    torch.save({"encoder": encoder.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
