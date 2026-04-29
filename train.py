"""

Changes :
  1. FSAM instead of SAM          — better geometry-aware perturbation
  2. Cosine LR decay              — replaces flat LR; free convergence improvement
  3. EMA weight tracking          — shadow weights used at eval time
  4. Improved model head          — GELU + extra Dropout for better regularisation
  5. Augmentation                 — via noise + SpecAugment
  6. Saves EMA state in checkpoint 

Usage:
    python train.py --epochs 20 --batch_size 8 --lr 1e-5
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ASTFeatureExtractor
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.dataset import ASTDataset
from src.model   import CustomAST, EMA
from src.sam     import FSAM


def icbhi_score(all_preds, all_labels):
    cm = confusion_matrix(all_labels, all_preds)
    se = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp = cm[0, 0]            / (np.sum(cm[0,  :]) + 1e-8)
    return se, sp, (se + sp) / 2


def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {DEVICE}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"📥 Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data file not found: {args.data_path}. Run preprocess.py first."
        )

    data = np.load(args.data_path)
    X_train, y_train, d_train = data["X_train"], data["y_train"], data["device_train"]
    X_test,  y_test,  d_test  = data["X_test"],  data["y_test"],  data["device_test"]

    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    # ── Weighted sampler ────────────────────────────
    counts         = np.bincount(y_train)
    sample_weights = [1.0 / counts[y] for y in y_train]
    sampler        = WeightedRandomSampler(sample_weights, len(y_train))

    # augment=True for train, augment=False for test
    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("🧠 Preparing model")
    model = CustomAST(num_classes=4, freeze_layers=8).to(DEVICE)

    # ── Loss ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer: FSAM wrapping AdamW ────────────────────────────────────────
    optimizer = FSAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=0.05,
        fisher_beta=1e-2,
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ── Cosine LR decay over the full training run ────────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer.base_optimizer,
        T_max=args.epochs,
        eta_min=1e-7,
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=0.999)

    # ── Resume from last epoch checkpoint if one exists ───────────────────────
    start_epoch = 1
    best_score  = 0.0
    best_se     = 0.0
    resume_path = os.path.join(args.checkpoint_dir, "resume.pth")

    if os.path.exists(resume_path):
        print(f"🔄 Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.base_optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_score  = ckpt["best_score"]
        best_se     = ckpt["best_se"]
        print(f"   Resumed at epoch {start_epoch}  |  best score so far: {best_score*100:.2f}%")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"🚀 Training — epochs {start_epoch} → {args.epochs}")
    print("   Paper SOTA → Se: 68.31%  Sp: 67.89%  Score: 68.10%")
    print("=" * 58)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for inputs, labels, _ in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # FSAM first step
            logits = model(inputs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # FSAM second step
            criterion(model(inputs), labels).backward()
            optimizer.second_step(zero_grad=True)

            ema.update(model)
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # ── Eval with EMA weights ─────────────────────────────────────────────
        ema.apply_shadow(model)
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                preds  = model(inputs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        ema.restore(model)

        se, sp, score = icbhi_score(all_preds, all_labels)
        lr_now        = optimizer.base_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | lr={lr_now:.1e} | "
            f"loss={running_loss/len(train_loader):.4f} | "
            f"Se={se*100:.2f}%  Sp={sp*100:.2f}%  Score={score*100:.2f}%"
        )

        # Save best (score first, then se as tiebreaker)
        if score > best_score or (score >= best_score - 0.005 and se > best_se):
            best_score = score
            best_se    = se
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema":   ema.state_dict(),
                    "score": best_score,
                    "se":    best_se,
                    "sp":    sp,
                },
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )
            print(f"   💾 Saved best  → Score={best_score*100:.2f}%  Se={best_se*100:.2f}%")

        # Save resume checkpoint (overwrites each epoch — always reflects latest state)
        torch.save(
            {
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.base_optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "ema":        ema.state_dict(),
                "best_score": best_score,
                "best_se":    best_se,
            },
            resume_path,
        )

    print(f"\n🏆 Best Score: {best_score*100:.2f}%   Se: {best_se*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir", type=str,   default="./checkpoints")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-5)
    args = parser.parse_args()
    train(args)