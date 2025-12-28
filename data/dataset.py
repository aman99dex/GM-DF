"""
GM-DF Multi-Domain Dataset
Supports multiple deepfake detection datasets with domain labels
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


class MultiDomainDataset(Dataset):
    """
    Multi-domain dataset for deepfake detection.
    
    Returns (image, label, domain_id) tuples where:
        - image: Preprocessed image tensor
        - label: 0 for real, 1 for fake
        - domain_id: Index of the domain/dataset
    
    Expected directory structure for each domain:
        domain_path/
        ├── real/
        │   ├── img001.jpg
        │   └── ...
        └── fake/
            ├── img001.jpg
            └── ...
    """
    
    # Default domain names matching config
    DOMAIN_NAMES = [
        "FaceForensics",
        "WildDeepfake",
        "CelebDF",
        "DFDC",
        "DeepFakeFace",
    ]
    
    def __init__(
        self,
        domain_paths: Dict[str, str],  # {domain_name: path}
        transform: Optional[Callable] = None,
        split: str = "train",
        split_ratio: float = 0.8,
        seed: int = 42,
        balance_domains: bool = True,
    ):
        """
        Args:
            domain_paths: Dictionary mapping domain names to their root paths
            transform: Image transform to apply
            split: "train" or "val"
            split_ratio: Fraction of data for training
            seed: Random seed for reproducible splits
            balance_domains: Whether to balance sampling across domains
        """
        super().__init__()
        self.domain_paths = domain_paths
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.balance_domains = balance_domains
        
        # Map domain names to indices
        self.domain_to_idx = {
            name: idx for idx, name in enumerate(self.DOMAIN_NAMES)
        }
        
        # Collect all samples
        self.samples = []  # List of (image_path, label, domain_id)
        self.domain_counts = {}
        
        random.seed(seed)
        
        for domain_name, domain_path in domain_paths.items():
            if domain_name not in self.domain_to_idx:
                print(f"Warning: Unknown domain '{domain_name}', skipping")
                continue
            
            domain_id = self.domain_to_idx[domain_name]
            domain_samples = self._load_domain(domain_path, domain_id)
            
            # Split data
            random.shuffle(domain_samples)
            split_idx = int(len(domain_samples) * split_ratio)
            
            if split == "train":
                domain_samples = domain_samples[:split_idx]
            else:
                domain_samples = domain_samples[split_idx:]
            
            self.samples.extend(domain_samples)
            self.domain_counts[domain_name] = len(domain_samples)
        
        print(f"Loaded {len(self.samples)} samples from {len(domain_paths)} domains")
        for domain, count in self.domain_counts.items():
            print(f"  - {domain}: {count} samples")
    
    def _load_domain(
        self, 
        domain_path: str, 
        domain_id: int,
    ) -> List[Tuple[str, int, int]]:
        """Load samples from a single domain."""
        samples = []
        domain_path = Path(domain_path)
        
        # Load real images (recursively to support nested structures like c23/videos/xxx/)
        real_path = domain_path / "real"
        if real_path.exists():
            for img_path in real_path.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    samples.append((str(img_path), 0, domain_id))
        
        # Load fake images (recursively to support nested structures)
        fake_path = domain_path / "fake"
        if fake_path.exists():
            for img_path in fake_path.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    samples.append((str(img_path), 1, domain_id))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_path, label, domain_id = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, domain_id
    
    def get_domain_sampler(self) -> WeightedRandomSampler:
        """Get a sampler that balances across domains AND classes (Real/Fake)."""
        # 1. Count samples per (domain, label) pair
        counts = {}  # (domain_id, label) -> count
        for _, label, domain_id in self.samples:
            key = (domain_id, label)
            counts[key] = counts.get(key, 0) + 1
            
        # 2. Compute weights: inverse frequency
        # Target weight per sample = Total / (NumGroups * CountInGroup)
        # Here we just use inverse count -> standard weighted sampling
        total = len(self.samples)
        sample_weights = []
        
        for _, label, domain_id in self.samples:
            count = counts.get((domain_id, label), 0)
            weight = 1.0 / max(count, 1.0)
            sample_weights.append(weight)
            
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.samples),
            replacement=True,
        )


class DomainBatchSampler:
    """
    Batch sampler that ensures each batch comes from a single domain.
    Used for meta-learning where we need domain-pure batches.
    """
    
    def __init__(
        self,
        dataset: MultiDomainDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group samples by domain
        self.domain_indices = {}
        for idx, (_, _, domain_id) in enumerate(dataset.samples):
            if domain_id not in self.domain_indices:
                self.domain_indices[domain_id] = []
            self.domain_indices[domain_id].append(idx)
    
    def __iter__(self):
        # Shuffle within each domain if needed
        domain_indices = {}
        for domain_id, indices in self.domain_indices.items():
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)
            domain_indices[domain_id] = indices
        
        # Create batches from each domain
        all_batches = []
        for domain_id, indices in domain_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only full batches
                    all_batches.append((domain_id, batch))
        
        # Shuffle batches across domains
        if self.shuffle:
            random.shuffle(all_batches)
        
        for domain_id, batch in all_batches:
            yield batch
    
    def __len__(self):
        total = 0
        for indices in self.domain_indices.values():
            total += len(indices) // self.batch_size
        return total


def create_meta_dataloaders(
    domain_paths: Dict[str, str],
    batch_size: int = 32,
    num_workers: int = 4,
    transform_train=None,
    transform_val=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for meta-learning training.
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    if transform_train is None:
        transform_train = get_train_transforms()
    if transform_val is None:
        transform_val = get_val_transforms()
    
    train_dataset = MultiDomainDataset(
        domain_paths=domain_paths,
        transform=transform_train,
        split="train",
    )
    
    val_dataset = MultiDomainDataset(
        domain_paths=domain_paths,
        transform=transform_val,
        split="val",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_dataset.get_domain_sampler(),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# Helper function to create dummy data for testing
def create_dummy_dataset(
    root: str = "/tmp/gmdf_dummy",
    num_images: int = 100,
    image_size: int = 224,
) -> Dict[str, str]:
    """Create a dummy dataset for testing."""
    import numpy as np
    
    root = Path(root)
    domain_paths = {}
    
    for domain_name in ["FaceForensics", "CelebDF"]:
        domain_path = root / domain_name
        
        for class_name in ["real", "fake"]:
            class_path = domain_path / class_name
            class_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_images // 2):
                # Create random image
                img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                img = Image.fromarray(img)
                img.save(class_path / f"img_{i:04d}.jpg")
        
        domain_paths[domain_name] = str(domain_path)
    
    return domain_paths
