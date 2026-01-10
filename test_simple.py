"""
Test script for simplified CLIP-based deepfake detector.
Calculates AUC, EER, and per-domain metrics.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader
import argparse
import clip

from data.dataset import MultiDomainDataset
from data.transforms import get_val_transforms


class SimpleCLIPClassifier(nn.Module):
    """Simple CLIP-based binary classifier for deepfake detection."""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        # Load CLIP
        self.clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
        self.clip_model.float()
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Simple classifier head (must match training)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        features = features.float()
        logits = self.classifier(features)
        return logits.squeeze(-1)


def calculate_eer(labels, preds):
    """Calculate Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def test_model(args):
    print("\n" + "="*60)
    print("GM-DF: Simplified Model Evaluation")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    model = SimpleCLIPClassifier(device=device)
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    print(f"[*] Loading checkpoint from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Setup test data
    print(f"[*] Testing on domains: {args.domains}")
    domain_paths = {
        domain: str(Path(args.data_root) / domain)
        for domain in args.domains
    }
    
    # Filter valid domains
    valid_domains = {}
    for d, p in domain_paths.items():
        if Path(p).exists():
            valid_domains[d] = p
        else:
            print(f"[!] Warning: Domain path {p} does not exist, skipping.")
    
    if not valid_domains:
        print("Error: No valid domain paths found.")
        return
    
    # Create domain name to ID mapping based on order
    domain_names = list(valid_domains.keys())
    domain_to_id = {name: idx for idx, name in enumerate(domain_names)}
    
    test_dataset = MultiDomainDataset(
        domain_paths=valid_domains,
        transform=get_val_transforms(),
        split="val",
        split_ratio=0.0,  # Use 100% as test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"[*] Loaded {len(test_dataset)} test samples.")
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_domains = []
    
    print("[*] Running evaluation...")
    with torch.no_grad():
        for images, labels, domain_ids in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_domains.extend(domain_ids.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_domains = np.array(all_domains)
    
    # Overall metrics
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    eer, eer_threshold = calculate_eer(all_labels, all_preds)
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"  Samples:   {len(all_labels)}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUC:       {auc:.4f} ({auc*100:.2f}%)")
    print(f"  EER:       {eer:.4f} ({eer*100:.2f}%)")
    print(f"  EER Thresh: {eer_threshold:.4f}")
    print("="*60)
    
    # Per-domain metrics
    print("\n📊 PER-DOMAIN METRICS:")
    print("-" * 50)
    
    for dom_id, dom_name in enumerate(domain_names):
        mask = all_domains == dom_id
        if mask.sum() == 0:
            continue
        
        dom_preds = all_preds[mask]
        dom_labels = all_labels[mask]
        
        try:
            dom_auc = roc_auc_score(dom_labels, dom_preds)
            dom_eer, _ = calculate_eer(dom_labels, dom_preds)
        except:
            dom_auc = 0.5
            dom_eer = 0.5
        
        dom_acc = accuracy_score(dom_labels, (dom_preds > 0.5).astype(int))
        
        print(f"  {dom_name:20s}  AUC={dom_auc:.4f}  EER={dom_eer:.4f}  Acc={dom_acc:.4f}  (n={mask.sum()})")
    
    print("="*60 + "\n")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write("GM-DF Test Results\n")
            f.write("="*40 + "\n\n")
            f.write(f"Overall: AUC={auc:.4f}, EER={eer:.4f}, Acc={acc:.4f}\n\n")
            f.write("Per-Domain:\n")
            for dom_id, dom_name in enumerate(domain_names):
                mask = all_domains == dom_id
                if mask.sum() == 0:
                    continue
                dom_preds = all_preds[mask]
                dom_labels = all_labels[mask]
                try:
                    dom_auc = roc_auc_score(dom_labels, dom_preds)
                    dom_eer, _ = calculate_eer(dom_labels, dom_preds)
                except:
                    dom_auc, dom_eer = 0.5, 0.5
                dom_acc = accuracy_score(dom_labels, (dom_preds > 0.5).astype(int))
                f.write(f"  {dom_name}: AUC={dom_auc:.4f}, EER={dom_eer:.4f}, Acc={dom_acc:.4f}\n")
        print(f"[*] Results saved to {args.output_file}")
    
    return {"auc": auc, "eer": eer, "acc": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test simplified CLIP deepfake detector")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--domains", type=str, nargs="+", 
                        default=["FaceForensics", "Celeb-DF-v1", "Celeb-DF-v2", "WildDeepfake", "StableDiffusion"])
    parser.add_argument("--model_path", type=str, default="checkpoints/simple_best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_file", type=str, default=None)
    
    args = parser.parse_args()
    test_model(args)
