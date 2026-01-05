
import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from config import GMDFConfig
from model.gm_df import build_model
from data.transforms import get_val_transforms
from data.dataset import MultiDomainDataset
from torch.utils.data import DataLoader

def test_model(args):
    print("\n" + "="*60)
    print("GM-DF: Model Evaluation")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Config and Model
    config = GMDFConfig()
    model = build_model(config=config, device=device, verbose=False)
    
    # 2. Load Checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"[*] Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Setup Test Data
    print(f"[*] Testing on domains: {args.domains}")
    domain_paths = {
        domain: os.path.join(args.data_root, domain)
        for domain in args.domains
    }
    
    # Check if domains exist
    for d, p in domain_paths.items():
        if not os.path.exists(p):
            print(f"Error: Domain path {p} does not exist.")
            return

    test_dataset = MultiDomainDataset(
        domain_paths=domain_paths,
        transform=get_val_transforms(),
        split="val", # Use entire provided folder structure as validation/test
        # Note: If you want to use the 'test' split specifically, ensure dataset.py supports it.
        # Currently dataset.py splits train/val. For external testing, we usually manually separate folders.
        # Here we just reuse 'val' logic which loads the data. 
    )
    
    # For testing, we might want to use ALL data in the folder, not just the 20% split.
    # Hack: We can override the split ratio to 0.0 to make 'val' take 100% of data if split="val".
    # But dataset.py logic is: if split="train" takes [:ratio], if "val" takes [ratio:].
    # So if we want 100% for testing, we should use split="val" and ratio=0.0.
    test_dataset.split_ratio = 0.0 
    # Reload samples with new ratio logic requires re-init or re-calling load.
    # Easier: Just re-instantiate with split_ratio=0.0 and split="val"
    
    test_dataset = MultiDomainDataset(
        domain_paths=domain_paths,
        transform=get_val_transforms(),
        split="val",
        split_ratio=0.0 # This ensures "val" gets 100% of the data (from index 0 to end)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"[*] Loaded {len(test_dataset)} samples.")

    # 4. Evaluation Loop
    all_preds = []
    all_labels = []
    
    print("[*] Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels, domain_ids = batch
            images = images.to(device)
            labels = labels.to(device)
            domain_ids = domain_ids.to(device)
            # Clamp domain_ids to valid range for model (max 5 for trained model)
            domain_ids = torch.clamp(domain_ids, 0, 5)
            
            # Forward pass
            outputs = model(images, domain_ids, labels) # labels needed for some internal logic? 
            # Actually labels are optional in forward, but if provided, losses are calc'd.
            # We just need logits.
            
            logits = outputs["logits"]
            probs = torch.sigmoid(logits.squeeze(-1))
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    try:
        auc = roc_auc_score(all_labels, all_preds)
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    except:
        auc = 0.5
        eer = 0.5

    print("\n" + "="*30)
    print("TEST RESULTS")
    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"EER:      {eer:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--domains", type=str, nargs="+", required=True)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    test_model(args)
