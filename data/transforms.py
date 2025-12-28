"""
GM-DF Data Transforms
Face preprocessing and augmentation pipelines
"""

import torch
from torchvision import transforms
from typing import Tuple


import random
import io
from PIL import Image
import numpy as np

class JPEGCompression(object):
    """
    Apply JPEG compression augmentation.
    Simulates the compression artifacts common in deepfake datasets like FF++.
    """
    def __init__(self, quality_range=(50, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            # Save to memory buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            # Reload
            img = Image.open(buffer)
        return img

def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> transforms.Compose:
    """
    Get training transforms for deepfake detection.
    
    Includes:
    - Random horizontal flip
    - Random rotation
    - Color jitter
    - Normalization to CLIP's expected values
    
    Args:
        image_size: Target image size (default 224 for CLIP)
        mean: Normalization mean (CLIP default)
        std: Normalization std (CLIP default)
    
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        # Robustness augmentations for Deepfake Detection
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
        JPEGCompression(quality_range=(60, 100), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> transforms.Compose:
    """
    Get validation/test transforms for deepfake detection.
    
    No augmentation, only resize and normalize.
    
    Args:
        image_size: Target image size (default 224 for CLIP)
        mean: Normalization mean (CLIP default)
        std: Normalization std (CLIP default)
    
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_unnormalize_transform(
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> transforms.Compose:
    """Get transform to unnormalize images back to [0, 1] range."""
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    
    return transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=inv_std),
        transforms.Normalize(mean=inv_mean, std=[1, 1, 1]),
    ])
