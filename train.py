import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader          # ← CHANGED: removed WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ASTFeatureExtractor
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from src.dataset import ASTDataset
from src.model   import CustomAST, EMA
from src.sam     import FSAM


# ── Metrics ───────────────────────────────────────────────────────────────────

def icbhi_score(all_preds, all_labels):
    cm = confusion_matrix(all_labels, all_preds)
    se = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp = cm[0, 0]            / (np.sum(cm[0,  :]) + 1e-8)
    return se, sp, (se + sp) / 2


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):

    SCORE_FLOOR = 0.6200   # ← CHANGED: gate on balanced score, not Se-only

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"⚙️  Device: {DEVICE}")
    if num_gpus > 1:
        print(f"⚙️  Multi-GPU: {num_gpus} GPUs detected")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"📥 Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"{args.data_path} not found. Run preprocess.py first.")

    data = np.load(args.data_path)
    X_train, y_train, d_train = data["X_train"], data["y_train"], data["device_train"]
    X_test,  y_test,  d_test  = data["X_test"],  data["y_test"],  data["device_test"]

    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    # ← CHANGED: removed WeightedRandomSampler — class-weighted loss handles imbalance instead
    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("🧠 Preparing model")
    model = CustomAST(num_classes=4, freeze_layers=12).to(DEVICE)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # ── Loss  ─────────────────────────────────────────────────────────────────
    # ← CHANGED: class-weighted CE + label smoothing replaces plain CE + sampler
    counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * len(counts)  # normalise
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = FSAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=0.05,
        fisher_beta=1e-2,
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ── LR Schedule ───────────────────────────────────────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer.base_optimizer,
        T_max=args.epochs,
        eta_min=1e-7,
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(base_model, decay=0.999)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_score   = 0.0          # ← CHANGED: track best overall score
    resume_path  = os.path.join(args.checkpoint_dir, "resume.pth")

    if os.path.exists(resume_path):
        print(f"🔄 Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        base_model.load_state_dict(ckpt["model"])
        optimizer.base_optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_score  = ckpt.get("best_score", 0.0)   # ← CHANGED
        print(f"   Resumed at epoch {start_epoch}  |  best Score so far: {best_score*100:.2f}%")

    # ── Loop ──────────────────────────────────────────────────────────────────
    print(f"🚀 Training — epochs {start_epoch} → {args.epochs}")
    print(f"   Goal  : beat Score=68.10%  (Se=68.31%  Sp=67.89%)")
    print(f"   SOTA  : Se=68.31%  Sp=67.89%  Score=68.10%")
    print("=" * 60)

    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    if start_epoch == 1 or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("Epoch,Train_Loss,Val_Se,Val_Sp,Val_Score\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Two-Phase unfreezing ──────────────────────────────────────────────
        if epoch == 7:
            print("\n🔓 Phase 2: Unfreezing upper 4 AST layers for fine-tuning...")
            for i, layer in enumerate(base_model.ast.audio_spectrogram_transformer.encoder.layer):
                if i >= 8:
                    for param in layer.parameters():
                        param.requires_grad = True

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for inputs, labels, _ in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            criterion(model(inputs), labels).backward()
            optimizer.second_step(zero_grad=True)

            ema.update(base_model)
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # ── Eval with EMA weights ─────────────────────────────────────────────
        ema.apply_shadow(base_model)
        model.eval()
        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                preds  = model(inputs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.numpy())

        ema.restore(base_model)

        se, sp, score = icbhi_score(all_preds, all_labels_list)
        lr_now        = optimizer.base_optimizer.param_groups[0]["lr"]

        beat_sota = "✅" if score > 0.6810 else "  "
        print(
            f"Epoch {epoch:02d} | lr={lr_now:.1e} | loss={running_loss/len(train_loader):.4f} | "
            f"Se={se*100:.2f}%  Sp={sp*100:.2f}%  Score={score*100:.2f}%{beat_sota}"
        )

        with open(log_path, "a") as f:
            f.write(f"{epoch},{running_loss/len(train_loader):.4f},{se:.4f},{sp:.4f},{score:.4f}\n")

        # ← CHANGED: save when balanced score improves past floor — no Sp-separate gate
        if score >= SCORE_FLOOR and score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model": base_model.state_dict(),
                    "ema":   ema.state_dict(),
                    "se":    se,
                    "sp":    sp,
                    "score": score,
                },
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )
            print(f"   💾 New best  Se={se*100:.2f}%  Sp={sp*100:.2f}%  Score={score*100:.2f}%")

        # Resume checkpoint — always overwrite
        torch.save(
            {
                "epoch":      epoch,
                "model":      base_model.state_dict(),
                "optimizer":  optimizer.base_optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "ema":        ema.state_dict(),
                "best_score": best_score,           # ← CHANGED
            },
            resume_path,
        )

    print(f"\n🏆 Best Score: {best_score*100:.2f}%")
    print(f"   Paper SOTA → Se: 68.31%  Sp: 67.89%  Score: 68.10%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir", type=str,   default="./checkpoints")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-5)
    args = parser.parse_args()
    train(args)