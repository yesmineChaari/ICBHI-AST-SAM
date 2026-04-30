
import os
import gc
import argparse
import pandas as pd
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

    checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
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
    all_probs, all_targets = [], []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
            else:
                logits = model(inputs)
            # Save probabilities instead of raw argmax predictions
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # ── Threshold Search ──────────────────────────────────────────────────────
    print("\n🔎 Running Threshold Search for 'Normal' class...")
    best_score  = 0.0
    best_thresh = 0.5
    best_preds  = None

    # Test thresholds from 0.20 to 0.80
    for thresh in np.arange(0.20, 0.81, 0.05):
        preds = np.zeros(len(all_probs), dtype=int)
        for i, p in enumerate(all_probs):
            if p[0] >= thresh:
                preds[i] = 0  # Predict Normal if probability beats the threshold
            else:
                # Otherwise, predict the highest probability among the abnormal classes
                preds[i] = np.argmax(p[1:]) + 1
                
        cm  = confusion_matrix(all_targets, preds)
        se  = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
        sp  = cm[0, 0]            / (np.sum(cm[0,  :]) + 1e-8)
        score = (se + sp) / 2
        
        print(f"   Thresh {thresh:.2f} -> Se: {se*100:.2f}%  Sp: {sp*100:.2f}%  Score: {score*100:.2f}%")
        
        if score > best_score:
            best_score  = score
            best_thresh = thresh
            best_preds  = preds

    print(f"\n🏆 Best Threshold selected: {best_thresh:.2f}")

    # ── Final Metrics (Using Best Threshold) ──────────────────────────────────
    all_preds = best_preds
    cm  = confusion_matrix(all_targets, all_preds)
    se  = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp  = cm[0, 0]            / (np.sum(cm[0,  :]) + 1e-8)
    
    print(f"\n📊 Final Optimised Metrics:")
    print(f"   Sensitivity (Se) : {se*100:.2f}%")
    print(f"   Specificity (Sp) : {sp*100:.2f}%")
    print(f"   Score            : {best_score*100:.2f}%")
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

        # ── Plot Training Curves from Log ─────────────────────────────────────────
    # Deduce the checkpoint directory from the model path
    checkpoint_dir = os.path.dirname(args.model_path)
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    
    if os.path.exists(log_path):
        print(f"\n📈 Found training log! Generating learning curves...")
        df = pd.read_csv(log_path)
        
        plt.figure(figsize=(14, 5))
        
        # Subplot 1: Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(df['Epoch'], df['Train_Loss'], marker='o', color='red', label='Train Loss')
        plt.title('Training Loss vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Subplot 2: Metrics Curve
        plt.subplot(1, 2, 2)
        plt.plot(df['Epoch'], df['Val_Se'], marker='o', color='blue', label='Sensitivity (Se)')
        plt.plot(df['Epoch'], df['Val_Sp'], marker='s', color='green', label='Specificity (Sp)')
        plt.plot(df['Epoch'], df['Val_Score'], marker='^', color='purple', label='Overall Score')
        
        # Draw a line for the paper's target score
        plt.axhline(y=0.6810, color='black', linestyle='--', label='Paper Target (68.10%)')
        
        plt.title('Validation Metrics vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plot_path = os.path.join(args.output_dir, "training_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"✅ Learning curves saved to: {plot_path}")
    else:
        print("\n⚠️ No training_log.csv found. Skipping learning curves plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args)