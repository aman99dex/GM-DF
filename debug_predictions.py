"""
Debug script to analyze model predictions and diagnose AUC=0.5 issue
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import GMDFConfig
from model.gm_df import build_model
from data.transforms import get_val_transforms
from data.dataset import MultiDomainDataset
from torch.utils.data import DataLoader


def debug_predictions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    config = GMDFConfig()
    model = build_model(config=config, device=device, verbose=False)
    
    # Try to load checkpoint
    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint found, using random weights")
    
    model.eval()
    
    # Load a small test batch
    domain_paths = {
        "FaceForensics": "datasets/FaceForensics",
        "StableDiffusion": "datasets/StableDiffusion",
    }
    
    # Check which exist
    valid_domains = {}
    for d, p in domain_paths.items():
        if Path(p).exists():
            valid_domains[d] = p
            print(f"Found domain: {d}")
    
    if not valid_domains:
        print("ERROR: No valid domains found!")
        return
    
    dataset = MultiDomainDataset(
        domain_paths=valid_domains,
        transform=get_val_transforms(),
        split="val",
        split_ratio=0.0,  # Use all data
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get predictions
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:  # Only check first 10 batches
                break
            
            images, labels, domain_ids = batch
            images = images.to(device)
            domain_ids = domain_ids.to(device)
            
            outputs = model(images, domain_ids)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits.squeeze(-1))
            
            all_logits.extend(logits.squeeze(-1).cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Show first batch details
            if i == 0:
                print(f"\n=== First Batch Sample ===")
                print(f"Labels:  {labels[:8].tolist()}")
                print(f"Logits:  {logits.squeeze(-1)[:8].tolist()}")
                print(f"Probs:   {[f'{p:.3f}' for p in probs[:8].tolist()]}")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    print(f"\n=== Statistics ===")
    print(f"Total samples: {len(all_preds)}")
    print(f"Labels: {np.sum(all_labels==0)} real, {np.sum(all_labels==1)} fake")
    print(f"\nLogits range: [{all_logits.min():.4f}, {all_logits.max():.4f}]")
    print(f"Logits mean: {all_logits.mean():.4f}, std: {all_logits.std():.4f}")
    print(f"\nPreds range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    print(f"Preds mean: {all_preds.mean():.4f}, std: {all_preds.std():.4f}")
    
    # Check if predictions are collapsed
    if all_preds.std() < 0.01:
        print("\n⚠️ PROBLEM: Predictions have near-zero variance!")
        print("   Model is outputting the same prediction for all samples.")
        print("   This explains the AUC=0.5")
    
    # Check label distribution vs prediction
    print(f"\n=== Prediction vs Label ===")
    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    print(f"Mean pred for REAL: {all_preds[real_mask].mean():.4f}")
    print(f"Mean pred for FAKE: {all_preds[fake_mask].mean():.4f}")
    
    # Calculate AUC manually
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds)
        print(f"\nCalculated AUC: {auc:.4f}")
    except:
        print("\nCould not calculate AUC (likely constant predictions)")


if __name__ == "__main__":
    debug_predictions()
