"""
experiments/train_clothing1m.py
================================
Reproduces Table 15 / Table S28 — Clothing1M real-world noise.

Expected result: AAB-SGN  75.23 ± 0.58%  (p < 0.001 vs SGN)
Mode:            two_stage  (KL ≈ 0.34 → SGN raises to 1.82, Table 4)

Usage
-----
    python experiments/train_clothing1m.py \\
        --data_root /path/to/clothing1m --seed 42

Download Clothing1M: https://github.com/Cysu/noisy_label
Expected directory:
    data_root/images/
    data_root/noisy_train_key_list.txt
    data_root/clean_val_key_list.txt

Config (Table S2, Supplementary):
    ResNet-50 pretrained on ImageNet, Adam lr=0.001, StepLR ×0.1 @5ep, 10 epochs, batch=64.
Hardware: 8×A100, ~12h. Single GPU: ~35h.
"""

import argparse, logging, os, sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aab_sgn import AABSGNTrainer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


class Clothing1MDataset(Dataset):
    """Clothing1M loader — expects key-list files from Cysu/noisy_label."""

    def __init__(self, root, key_file, transform=None, return_index=False):
        self.root = root
        self.transform = transform
        self.return_index = return_index
        self.samples = []
        with open(os.path.join(root, key_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img, label, idx) if self.return_index else (img, label)


def get_loaders(root, batch_size):
    mean = (0.6959, 0.6537, 0.6371)
    std  = (0.3113, 0.3192, 0.3214)
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_ds = Clothing1MDataset(root, "noisy_train_key_list.txt", train_tf, return_index=True)
    val_ds   = Clothing1MDataset(root, "clean_val_key_list.txt",   val_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
        DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=8, pin_memory=True),
        len(train_ds),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--pi_min",     type=float, default=0.10)
    p.add_argument("--save_dir",   type=str,   default="./checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, n_train = get_loaders(args.data_root, args.batch_size)

    # ResNet-50 pretrained on ImageNet (Table S2)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 14)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = AABSGNTrainer(
        model=model, optimizer=optimizer, device=device,
        pi_min=args.pi_min, kl_threshold=1.5, use_adaptive_kl=True,
        num_train_samples=n_train, scheduler=scheduler, log_interval=200,
    )

    # Expected: KL ≈ 0.34 → two_stage  (SGN raises to 1.82 during training, Table 4)
    report = trainer.run_kl_diagnostic(train_loader, n_samples=500)
    log.info(f"[Diagnostic] {report}")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(train_loader, epoch)
        acc = trainer.evaluate(val_loader)["acc"]
        log.info(f"Epoch {epoch:2d} | Val Acc={acc*100:.2f}%")
        if acc > best:
            best = acc
            torch.save(model.state_dict(),
                       f"{args.save_dir}/clothing1m_seed{args.seed}.pt")

    log.info(f"FINAL: {best*100:.2f}%  (expected: 75.23 ± 0.58%)")


if __name__ == "__main__":
    main()
