"""
GM-DF MAML Training Loop (MDEO - Meta-Domain-Embedding Optimizer)
Implements Equations 13-15 from the paper.

Features:
- Automatic GPU detection (CUDA, MPS, CPU)
- Optimized DataLoader with pin_memory and prefetching
- Mixed precision training (AMP) for NVIDIA GPUs
- Non-blocking data transfers

Training structure:
1. Inner Loop (Meta-Train): Update θ_E (expert params) with L_cls on Domain A
2. Outer Loop (Meta-Test): Update θ_O (base params) with L_total on Domain B
"""

import os
import copy
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import GMDFConfig
from model.gm_df import GMDF_Detector, build_model
from data.dataset import MultiDomainDataset, create_meta_dataloaders

# Import device utilities
try:
    from utils import (
        get_optimal_device,
        get_dataloader_kwargs,
        to_device,
        get_amp_context,
        get_grad_scaler,
        print_device_summary,
        DeviceInfo,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


class MAMLTrainer:
    """
    MAML-based trainer for GM-DF (MDEO - Equations 13-15)
    
    Features:
    - Automatic GPU detection (CUDA > MPS > CPU)
    - Mixed precision training (AMP) when supported
    - Optimized data loading
    
    Inner loop: θ'_E ← θ_E - β·∇_{θ_E}(L_cls)
    Outer loop: θ'_O ← θ_O - δ·∇_{θ_O}(L_total)
    """
    
    def __init__(
        self,
        model: GMDF_Detector,
        config: GMDFConfig,
        device_info: Optional['DeviceInfo'] = None,
        device: Optional[str] = None,
        log_dir: str = "./logs",
        use_amp: bool = True,
    ):
        """
        Args:
            model: GMDF_Detector model
            config: Model configuration
            device_info: DeviceInfo from get_optimal_device() (preferred)
            device: Device string fallback if device_info not provided
            log_dir: TensorBoard log directory
            use_amp: Enable automatic mixed precision (if supported)
        """
        # Setup device
        if device_info is not None:
            self.device_info = device_info
            self.device = device_info.device
        elif UTILS_AVAILABLE:
            self.device_info = get_optimal_device(preferred=device, verbose=True)
            self.device = self.device_info.device
        else:
            # Fallback device detection
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self.device = torch.device(device)
            self.device_info = None
            print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.config = config
        
        # Setup AMP (Automatic Mixed Precision)
        self.use_amp = use_amp and self._supports_amp()
        if self.use_amp:
            if UTILS_AVAILABLE and self.device_info:
                self.scaler = get_grad_scaler(self.device_info, enabled=True)
            else:
                self.scaler = torch.amp.GradScaler("cuda")
            print("[OK] Automatic Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
            if use_amp:
                print("[!] AMP not supported on this device")
        
        # Get parameter groups
        param_groups = model.get_trainable_params()
        self.theta_E = param_groups["theta_E"]  # Expert params (inner loop)
        self.theta_O = param_groups["theta_O"]  # Other params (outer loop)
        
        # Optimizers
        self.inner_optimizer = torch.optim.SGD(
            self.theta_E, 
            lr=config.inner_lr,
        )
        self.outer_optimizer = torch.optim.Adam(
            self.theta_O,
            lr=config.outer_lr,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.outer_optimizer,
            T_max=config.epochs,
        )
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.global_step = 0
        self.best_val_auc = 0.0
    
    def _supports_amp(self) -> bool:
        """Check if device supports AMP."""
        if self.device_info is not None:
            return self.device_info.supports_amp
        # Fallback check
        return self.device.type == "cuda"
    
    def _get_amp_context(self):
        """Get AMP autocast context manager."""
        if self.use_amp:
            if UTILS_AVAILABLE and self.device_info:
                return get_amp_context(self.device_info, enabled=True)
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        # No-op context
        return torch.amp.autocast(device_type="cpu", enabled=False)
    
    def _to_device(self, data):
        """Move data to device with non-blocking transfer."""
        if UTILS_AVAILABLE:
            return to_device(data, self.device)
        # Fallback
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._to_device(d) for d in data)
        return data
    
    def inner_loop_ada(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        Inner loop adaptation (Domain A).
        
        θ'_E ← θ_E - β·∇_{θ_E}(L_cls)
        
        This modifies the model parameters IN-PLACE. 
        We must save them before this and restore after outer loop loss computation.
        """
        images, labels, domain_ids = self._to_device(batch)
        
        # Enable gradients
        self.model.train()
        
        # Forward pass
        with self._get_amp_context():
            outputs = self.model(images, domain_ids, labels)
            loss_cls = outputs["loss_cls"]
        
        # Update θ_E
        self.inner_optimizer.zero_grad()
        
        if self.use_amp and self.scaler:
            self.scaler.scale(loss_cls).backward()
            self.scaler.unscale_(self.inner_optimizer)
            torch.nn.utils.clip_grad_norm_(self.theta_E, max_norm=1.0)
            self.scaler.step(self.inner_optimizer)
            self.scaler.update()
        else:
            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(self.theta_E, max_norm=1.0)
            self.inner_optimizer.step()
            
        return outputs["loss_cls"].item()

    def train_step(
        self,
        batch_A: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_B: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single First-Order MAML training step.
        
        1. Save state of experts (θ_E)
        2. Adapt experts on Domain A: θ'_E = θ_E - β∇L_A
        3. Compute loss on Domain B with adapted experts: L_B(θ'_E, θ_O)
        4. Restore original experts: θ_E (original)
        5. Update θ_E and θ_O using gradients from L_B:
           θ_E ← θ_E - δ∇_{θ'_E} L_B (FOMAML approximation)
           θ_O ← θ_O - δ∇_{θ_O} L_B
        """
        self.model.train()
        
        # --- 1. Save State (Experts Only) ---
        # We only need to save θ_E because only they are updated in inner loop
        theta_E_state = {
            n: p.data.clone() 
            for n, p in self.model.named_parameters() 
            if p.requires_grad and ("moe_adapter" in n or "experts" in n or "router" in n)
        }
        
        # --- 2. Inner Loop Adaptation (Domain A) ---
        # Update θ_E in place
        loss_cls_A = self.inner_loop_ada(batch_A)
        
        # --- 3. Compute Outer Loss (Domain B) ---
        # Using adapted θ'_E and original θ_O
        images_B, labels_B, domain_ids_B = self._to_device(batch_B)
        
        with self._get_amp_context():
            outputs_B = self.model(images_B, domain_ids_B, labels_B)
            loss_total_B = outputs_B["loss_total"]
        
        # --- 4. Compute Gradients for Outer Update ---
        self.outer_optimizer.zero_grad() # Clears θ_O grads
        
        # We also need to clear θ_E grads from the inner loop update
        # (Though zero_grad above might have done it if they are in same optimizer, 
        # checking manually or just relying on zero_grad for all params is safer).
        # Actually standard practice:
        self.model.zero_grad()
        
        if self.use_amp and self.scaler:
            self.scaler.scale(loss_total_B).backward()
        else:
            loss_total_B.backward()
            
        # At this point:
        # p.grad contains ∇_{θ'_E} L_B (for experts)
        # p.grad contains ∇_{θ_O} L_B (for others)
        
        # Stop gradients to prevent accidental updates during restoration
        # We preserve the .grad attributes
        
        # --- 5. Restore Original Expert Weights ---
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in theta_E_state:
                    p.data.copy_(theta_E_state[n])
        
        # --- NaN Detection: Skip update if loss is NaN ---
        if torch.isnan(loss_total_B) or torch.isinf(loss_total_B):
            print(f"[!] NaN/Inf detected in loss, skipping batch")
            # Return zeros for this batch
            return {
                "loss_cls_A": 0.0,
                "loss_cls_B": 0.0,
                "loss_mim_B": 0.0,
                "loss_sis_B": 0.0,
                "loss_dal_B": 0.0,
                "loss_total_B": 0.0,
            }
        
        # --- 6. Meta-Update (Outer Loop) ---
        # Update both θ_E (initialization) and θ_O (shared)
        # using the gradients computed at θ'_E (FOMAML)
        
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.outer_optimizer)
            # Gradient clipping from config
            grad_clip = getattr(self.config, 'grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.theta_O + self.theta_E, max_norm=grad_clip)
            
            # Check for NaN in gradients
            has_nan_grad = False
            for p in self.theta_O + self.theta_E:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"[!] NaN/Inf in gradients, skipping update")
                self.scaler.update()
                return {
                    "loss_cls_A": loss_cls_A,
                    "loss_cls_B": 0.0,
                    "loss_mim_B": 0.0,
                    "loss_sis_B": 0.0,
                    "loss_dal_B": 0.0,
                    "loss_total_B": 0.0,
                }
            
            self.scaler.step(self.outer_optimizer)
            
            # Manual SGD step for theta_E using its current gradients
            with torch.no_grad():
                for p in self.theta_E:
                    if p.grad is not None and not torch.isnan(p.grad).any():
                        p.data -= self.config.outer_lr * p.grad
            
            self.scaler.update()
        else:
            # Gradient clipping from config (non-AMP path)
            grad_clip = getattr(self.config, 'grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.theta_O + self.theta_E, max_norm=grad_clip)
            self.outer_optimizer.step()
            # Manual step for theta_E
            with torch.no_grad():
                for p in self.theta_E:
                    if p.grad is not None and not torch.isnan(p.grad).any():
                        p.data -= self.config.outer_lr * p.grad
        
        losses = {
            "loss_cls_A": loss_cls_A,
            "loss_cls_B": outputs_B["loss_cls"].item(),
            "loss_mim_B": outputs_B["loss_mim"].item(),
            "loss_sis_B": outputs_B["loss_sis"].item(),
            "loss_dal_B": outputs_B.get("loss_dal", torch.tensor(0.0)).item(),
            "loss_total_B": outputs_B["loss_total"].item(),
        }
        
        self.global_step += 1
        
        return losses
    
    def sample_domain_batches(
        self,
        dataloader: DataLoader,
        num_domains: int = 2,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample batches from different domains for meta-training.
        
        Returns list of (images, labels, domain_ids) tuples.
        """
        # Group samples by domain
        domain_batches = defaultdict(list)
        
        for batch in dataloader:
            images, labels, domain_ids = batch
            # Get unique domains in this batch
            unique_domains = domain_ids.unique().tolist()
            
            for domain_id in unique_domains:
                mask = domain_ids == domain_id
                if mask.sum() > 0:
                    domain_batches[domain_id].append((
                        images[mask],
                        labels[mask],
                        domain_ids[mask],
                    ))
        
        return domain_batches
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with proper cross-domain batch pairing."""
        self.model.train()
        
        epoch_losses = defaultdict(list)
        
        # Group batches by domain
        domain_batches = defaultdict(list)
        for batch in train_loader:
            images, labels, domain_ids = batch
            # Use the majority domain in this batch
            dominant_domain = domain_ids.mode().values.item()
            domain_batches[dominant_domain].append(batch)
        
        # Get list of domains that have batches
        available_domains = list(domain_batches.keys())
        
        if len(available_domains) < 2:
            print(f"[!] Warning: Only {len(available_domains)} domain(s) available. MAML needs 2+.")
            # Fallback: just pair consecutive batches
            all_batches = list(train_loader)
            random.shuffle(all_batches)
            pairs = [(all_batches[i], all_batches[i+1]) for i in range(0, len(all_batches)-1, 2)]
        else:
            # Create cross-domain pairs: (batch from domain A, batch from domain B)
            pairs = []
            
            # Shuffle within each domain
            for d in available_domains:
                random.shuffle(domain_batches[d])
            
            # Create pairs by cycling through domains
            domain_indices = {d: 0 for d in available_domains}
            domain_list = list(available_domains)
            
            while True:
                # Pick two different domains
                random.shuffle(domain_list)
                domain_A, domain_B = domain_list[0], domain_list[1 % len(domain_list)]
                
                # Get next batch from each domain
                idx_A = domain_indices[domain_A]
                idx_B = domain_indices[domain_B]
                
                if idx_A >= len(domain_batches[domain_A]) or idx_B >= len(domain_batches[domain_B]):
                    break
                
                pairs.append((
                    domain_batches[domain_A][idx_A],
                    domain_batches[domain_B][idx_B]
                ))
                
                domain_indices[domain_A] += 1
                domain_indices[domain_B] += 1
        
        # Training loop
        pbar = tqdm(pairs, desc=f"Epoch {epoch}")
        
        for batch_A, batch_B in pbar:
            losses = self.train_step(batch_A, batch_B)
            
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['loss_total_B']:.4f}",
                "cls": f"{losses['loss_cls_B']:.4f}",
                "dal": f"{losses['loss_dal_B']:.4f}",
            })
        
        # Average losses
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        # Log to tensorboard
        for k, v in avg_losses.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            images, labels, domain_ids = self._to_device(batch)
            
            with self._get_amp_context():
                outputs = self.model(images, domain_ids, labels)
            
            total_loss += outputs["loss_total"].item()
            num_batches += 1
            
            # Collect predictions
            probs = torch.sigmoid(outputs["logits"].squeeze(-1))
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        import numpy as np
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
            # Calculate EER (Equal Error Rate)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=1)
            fnr = 1 - tpr
            # Find point where fpr and fnr are closest
            eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        except ValueError:
            auc = 0.5  # Handle case with single class
            eer = 0.5
        
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_auc": auc,
            "val_eer": eer,
        }
        
        # Log to tensorboard
        for k, v in metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        
        return metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "inner_optimizer_state_dict": self.inner_optimizer.state_dict(),
            "outer_optimizer_state_dict": self.outer_optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "global_step": self.global_step,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        print(f"[*] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.inner_optimizer.load_state_dict(checkpoint["inner_optimizer_state_dict"])
        self.outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint["epoch"], checkpoint.get("metrics", {})
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        save_dir: str = "./checkpoints",
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs (default from config)
            save_dir: Directory to save checkpoints
        """
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"Train losses: {train_losses}")
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            print(f"Val metrics: {val_metrics}")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save best model
            if val_metrics["val_auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["val_auc"]
                self.save_checkpoint(
                    str(save_dir / "best_model.pt"),
                    epoch,
                    val_metrics,
                )
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    str(save_dir / f"checkpoint_epoch{epoch}.pt"),
                    epoch,
                    val_metrics,
                )
        
        print(f"\n[*] Training complete! Best Val AUC: {self.best_val_auc:.4f}")
        self.writer.close()


def create_optimized_dataloaders(
    domain_paths: Dict[str, str],
    batch_size: int = 32,
    device_info: Optional['DeviceInfo'] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders with optimal settings for the detected device.
    
    Args:
        domain_paths: Dictionary mapping domain names to paths
        batch_size: Batch size per GPU
        device_info: DeviceInfo from get_optimal_device()
        num_workers: Override number of workers
    
    Returns:
        train_loader, val_loader
    """
    from data.transforms import get_train_transforms, get_val_transforms
    
    # Get optimal DataLoader kwargs
    if UTILS_AVAILABLE and device_info is not None:
        loader_kwargs = get_dataloader_kwargs(device_info, num_workers)
        # Disable pin_memory to avoid OOM in pin_memory thread
        loader_kwargs["pin_memory"] = False
    else:
        # Fallback defaults
        num_cpus = os.cpu_count() or 4
        loader_kwargs = {
            "num_workers": min(num_cpus, 4),
            "pin_memory": False,  # Disabled to prevent OOM
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
    
    print(f"DataLoader settings: {loader_kwargs}")
    
    train_dataset = MultiDomainDataset(
        domain_paths=domain_paths,
        transform=get_train_transforms(),
        split="train",
    )
    
    val_dataset = MultiDomainDataset(
        domain_paths=domain_paths,
        transform=get_val_transforms(),
        split="val",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_dataset.get_domain_sampler(),
        drop_last=True,
        **loader_kwargs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    return train_loader, val_loader


def main():
    """Main training script with automatic GPU detection."""
    parser = argparse.ArgumentParser(
        description="Train GM-DF model with automatic GPU detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True,
        help="Root directory containing domain folders"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["FaceForensics", "CelebDF", "WildDeepfake"],
        help="Domains to use for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--inner_lr", type=float, default=1e-4, help="MAML inner loop LR")
    parser.add_argument("--outer_lr", type=float, default=3e-6, help="MAML outer loop LR")
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (auto=detect best available)"
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (auto if None)")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GM-DF: Generalized Multi-Scenario Deepfake Detection")
    print("="*60 + "\n")
    
    # === Device Detection ===
    device_pref = None if args.device == "auto" else args.device
    if UTILS_AVAILABLE:
        device_info = get_optimal_device(preferred=device_pref, verbose=False)
        print_device_summary(device_info)
    else:
        device_info = None
        print("[!] Device utilities not available, using fallback detection")
    
    # === Create Config ===
    config = GMDFConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
    )
    
    # === Setup Domain Paths ===
    domain_paths = {
        domain: os.path.join(args.data_root, domain)
        for domain in args.domains
    }
    
    # Verify domains exist
    for domain, path in domain_paths.items():
        if not os.path.exists(path):
            print(f"[!] Warning: Domain path not found: {path}")
    
    # === Create Optimized DataLoaders ===
    print("\n[*] Creating optimized dataloaders...")
    train_loader, val_loader = create_optimized_dataloaders(
        domain_paths=domain_paths,
        batch_size=args.batch_size,
        device_info=device_info,
        num_workers=args.num_workers,
    )
    
    # === Build Model ===
    print("\n[*] Building model...")
    device = device_info.device if device_info else args.device
    model = build_model(config=config, device=device, verbose=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # === Create Trainer ===
    trainer = MAMLTrainer(
        model=model,
        config=config,
        device_info=device_info,
        log_dir=args.log_dir,
        use_amp=not args.no_amp,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        print(f"[*] Resumed from epoch {start_epoch}")
    
    # === Train ===
    print("\n[*] Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
