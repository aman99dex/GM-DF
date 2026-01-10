"""
Simplified training script for debugging AUC issues.
Uses standard training (no MAML) with just classification loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
import clip

from data.dataset import MultiDomainDataset
from data.transforms import get_train_transforms, get_val_transforms


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
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize with reasonable gains
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images):
        # Get CLIP features
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        features = features.float()
        
        # Classify
        logits = self.classifier(features)
        return logits.squeeze(-1)


def train_simple():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Hyperparameters
    batch_size = 64
    epochs = 30
    lr = 1e-3  # Higher LR for classifier-only training
    
    # Load data
    domain_paths = {
        "FaceForensics": "datasets/FaceForensics",
        "Celeb-DF-v1": "datasets/Celeb-DF-v1",
        "Celeb-DF-v2": "datasets/Celeb-DF-v2",
        "WildDeepfake": "datasets/WildDeepfake",
        "StableDiffusion": "datasets/StableDiffusion",
    }
    
    # Filter to existing domains
    valid_domains = {k: v for k, v in domain_paths.items() if Path(v).exists()}
    print(f"Using domains: {list(valid_domains.keys())}")
    
    train_dataset = MultiDomainDataset(
        domain_paths=valid_domains,
        transform=get_train_transforms(),
        split="train",
        split_ratio=0.8,
    )
    
    val_dataset = MultiDomainDataset(
        domain_paths=valid_domains,
        transform=get_val_transforms(),
        split="val",
        split_ratio=0.8,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = SimpleCLIPClassifier(device=device).to(device)
    print(f"Classifier params: {sum(p.numel() for p in model.classifier.parameters()):,}")
    
    # Optimizer - only train classifier
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss with class weights (if imbalanced)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            
            pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total)
        
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        print(f"\nEpoch {epoch+1}: Val AUC={auc:.4f}, Val Acc={acc:.4f}")
        print(f"  Preds range: [{all_preds.min():.4f}, {all_preds.max():.4f}], std={all_preds.std():.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "checkpoints/simple_best.pt")
            print(f"  [*] New best AUC: {auc:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Training complete! Best AUC: {best_auc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    train_simple()
