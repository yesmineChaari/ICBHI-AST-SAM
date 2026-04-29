"""
Changes :
  - Loads EMA shadow weights from checkpoint when available

"""

import os
import gc
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import ASTDataset
from src.model   import CustomAST, EMA


def evaluate(args):
    gc.collect()
    torch.cuda.empty_cache()

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]
    print(f"⚙️  Device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"📦 Loading data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    data      = np.load(args.data_path)
    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    test_loader = DataLoader(
        ASTDataset(data["X_test"], data["y_test"], data["device_test"],
                   processor, train=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"📦 Loading model: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. Run train.py first."
        )

    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model      = CustomAST(num_classes=4, freeze_layers=8).to(DEVICE)

    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        # Load EMA shadow weights — better generalisation than raw weights
        ema = EMA(model, decay=0.999)
        ema.load_state_dict(checkpoint["ema"])
        ema.apply_shadow(model)
        print("   ✅ Loaded EMA shadow weights")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("   ✅ Loaded model weights")
    else:
        # Original checkpoint format (plain state dict)
        model.load_state_dict(checkpoint)
        print("   ✅ Loaded raw state dict")

    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    print("🔍 Evaluating …")
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    preds = model(inputs).argmax(dim=1)
            else:
                preds = model(inputs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # ── Metrics ───────────────────────────────────────────────────────────────
    cm  = confusion_matrix(all_targets, all_preds)
    se  = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp  = cm[0, 0]            / (np.sum(cm[0,  :]) + 1e-8)
    score = (se + sp) / 2

    print(f"\n📊 Metrics:")
    print(f"   Sensitivity (Se) : {se*100:.2f}%")
    print(f"   Specificity (Sp) : {sp*100:.2f}%")
    print(f"   Score            : {score*100:.2f}%")
    print(f"\n   Paper SOTA → Se: 68.31%  Sp: 67.89%  Score: 68.10%")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"label": "Number of Samples"},
        )
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label",      fontsize=12, fontweight="bold")
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)

        metrics_text = (
            f"Se: {se*100:.2f}%  |  Sp: {sp*100:.2f}%  |  Score: {score*100:.2f}%"
        )
        plt.figtext(
            0.5, 0.02, metrics_text, ha="center", fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black",
                      boxstyle="round,pad=0.5"),
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        save_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args)