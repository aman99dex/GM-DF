"""
GM-DF Model Evaluation with Cross-Domain Testing
Supports per-domain metrics and held-out domain evaluation
"""

import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from collections import defaultdict

from config import GMDFConfig
from model.gm_df import build_model
from data.transforms import get_val_transforms
from data.dataset import MultiDomainDataset
from torch.utils.data import DataLoader


def compute_metrics(labels, preds):
    """Compute AUC, EER, and accuracy from labels and predictions."""
    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    try:
        auc = roc_auc_score(labels, preds)
        fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    except ValueError:
        auc = 0.5
        eer = 0.5
    return {"acc": acc, "auc": auc, "eer": eer}


def test_model(args):
    print("\n" + "="*60)
    print("GM-DF: Model Evaluation with Cross-Domain Analysis")
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
    valid_domains = {}
    for d, p in domain_paths.items():
        if not os.path.exists(p):
            print(f"[!] Warning: Domain path {p} does not exist, skipping.")
        else:
            valid_domains[d] = p
    
    if not valid_domains:
        print("Error: No valid domain paths found.")
        return
    
    # Create mapping from domain name to ID
    domain_name_to_id = {name: idx for idx, name in enumerate(args.domains)}
    
    test_dataset = MultiDomainDataset(
        domain_paths=valid_domains,
        transform=get_val_transforms(),
        split="val",
        split_ratio=0.0  # Use 100% of data for testing
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"[*] Loaded {len(test_dataset)} samples.")

    # 4. Evaluation Loop with per-domain tracking
    domain_preds = defaultdict(list)
    domain_labels = defaultdict(list)
    all_preds = []
    all_labels = []
    
    print("[*] Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels, domain_ids = batch
            images = images.to(device)
            labels = labels.to(device)
            domain_ids = domain_ids.to(device)
            
            # Clamp domain_ids to valid range
            domain_ids = torch.clamp(domain_ids, 0, config.num_domains - 1)
            
            # Forward pass (no labels to avoid loss computation overhead)
            outputs = model(images, domain_ids)
            
            logits = outputs["logits"]
            probs = torch.sigmoid(logits.squeeze(-1))
            
            # Store overall predictions
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store per-domain predictions
            for i, (prob, label, dom_id) in enumerate(zip(
                probs.cpu().numpy(), 
                labels.cpu().numpy(), 
                domain_ids.cpu().numpy()
            )):
                domain_preds[int(dom_id)].append(prob)
                domain_labels[int(dom_id)].append(label)

    # 5. Compute Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    overall_metrics = compute_metrics(all_labels, all_preds)
    
    # Per-domain metrics
    domain_metrics = {}
    for dom_id in sorted(domain_preds.keys()):
        preds = np.array(domain_preds[dom_id])
        labels = np.array(domain_labels[dom_id])
        if len(preds) > 0:
            domain_metrics[dom_id] = compute_metrics(labels, preds)
            domain_metrics[dom_id]["n_samples"] = len(preds)

    # 6. Print Results
    print("\n" + "="*60)
    print("CROSS-DOMAIN EVALUATION RESULTS")
    print("="*60)
    
    # Determine seen vs unseen domains
    train_domains = set(args.train_domains) if args.train_domains else set()
    
    if train_domains:
        print("\n📊 SEEN DOMAINS (trained on):")
        print("-" * 40)
        for dom_id, metrics in domain_metrics.items():
            dom_name = args.domains[dom_id] if dom_id < len(args.domains) else f"Domain_{dom_id}"
            if dom_name in train_domains:
                print(f"  {dom_name:20s}  AUC={metrics['auc']:.4f}  EER={metrics['eer']:.4f}  Acc={metrics['acc']:.4f}  (n={metrics['n_samples']})")
        
        print("\n🔮 UNSEEN DOMAINS (held-out):")
        print("-" * 40)
        for dom_id, metrics in domain_metrics.items():
            dom_name = args.domains[dom_id] if dom_id < len(args.domains) else f"Domain_{dom_id}"
            if dom_name not in train_domains:
                print(f"  {dom_name:20s}  AUC={metrics['auc']:.4f}  EER={metrics['eer']:.4f}  Acc={metrics['acc']:.4f}  (n={metrics['n_samples']})")
    else:
        print("\n📊 PER-DOMAIN METRICS:")
        print("-" * 40)
        for dom_id, metrics in domain_metrics.items():
            dom_name = args.domains[dom_id] if dom_id < len(args.domains) else f"Domain_{dom_id}"
            print(f"  {dom_name:20s}  AUC={metrics['auc']:.4f}  EER={metrics['eer']:.4f}  Acc={metrics['acc']:.4f}  (n={metrics['n_samples']})")
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"  Accuracy: {overall_metrics['acc']:.4f}")
    print(f"  AUC:      {overall_metrics['auc']:.4f}")
    print(f"  EER:      {overall_metrics['eer']:.4f}")
    print(f"  Samples:  {len(all_labels)}")
    print("="*60 + "\n")
    
    # 7. Save results to file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write("GM-DF Cross-Domain Evaluation Results\n")
            f.write("="*40 + "\n\n")
            f.write("Per-Domain Metrics:\n")
            for dom_id, metrics in domain_metrics.items():
                dom_name = args.domains[dom_id] if dom_id < len(args.domains) else f"Domain_{dom_id}"
                f.write(f"  {dom_name}: AUC={metrics['auc']:.4f}, EER={metrics['eer']:.4f}, Acc={metrics['acc']:.4f}\n")
            f.write(f"\nOverall: AUC={overall_metrics['auc']:.4f}, EER={overall_metrics['eer']:.4f}, Acc={overall_metrics['acc']:.4f}\n")
        print(f"[*] Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GM-DF Model Evaluation with Cross-Domain Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing domain folders")
    parser.add_argument("--domains", type=str, nargs="+", required=True,
                        help="Domains to evaluate on")
    parser.add_argument("--train_domains", type=str, nargs="+", default=None,
                        help="Domains used during training (for seen/unseen split)")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional file to save results")
    args = parser.parse_args()
    test_model(args)
