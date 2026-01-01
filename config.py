"""
GM-DF Configuration
Hyperparameters from Section 4.1 of the paper
With stability fixes for training
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GMDFConfig:
    """Configuration for GM-DF model and training."""
    
    # === Model Architecture ===
    backbone: str = "ViT-B/16"  # CLIP backbone variant
    image_size: int = 224  # Input image size
    patch_size: int = 16  # ViT patch size
    hidden_dim: int = 768  # ViT-B/16 hidden dimension
    mlp_dim: int = 3072  # ViT-B/16 MLP dimension
    num_heads: int = 12  # ViT-B/16 attention heads
    num_layers: int = 12  # ViT-B/16 transformer layers
    
    # === Dataset-Embedding Generator (DEG) - Eq.2 ===
    num_experts: int = 3  # N independent experts (Section 3.1)
    domain_embed_dim: int = 64  # k dimension for c_d in Eq.2
    num_domains: int = 5  # FF++, WDF, Celeb-DF, DFDC, DFF
    
    # === Multi-Dataset Representation (MDP) - Eq.7-8 ===
    second_order_layers: Tuple[int, ...] = (8, 9, 10)  # Layers for Eq.7-8
    alpha_init: float = 0.1  # Initial α_l for second-order fusion
    
    # === Masked Image Modeling (MIM) - Eq.6 ===
    mask_ratio: float = 0.50  # Paper: 50% masking ratio
    mim_decoder_layers: int = 4  # Decoder depth
    mim_decoder_dim: int = 384  # Decoder hidden dimension
    
    # === Prompt Learning - Eq.9-11 ===
    prompt_length: int = 4  # Learnable context tokens
    tau_base: float = 0.07  # Base temperature (CLIP default)
    tau_phi: float = 0.01  # Temperature scaling factor φ in Eq.10
    
    # === MAML Training (MDEO) - Eq.13-15 ===
    # Best found: outer_lr=1e-5 works better than paper spec 3e-6
    inner_lr: float = 1e-4   # β - inner loop (experts/adaptation)
    outer_lr: float = 1e-5   # δ - outer loop (backbone) - OPTIMIZED
    inner_steps: int = 1     # Inner loop gradient steps
    
    # === General Training ===
    batch_size: int = 32  # Section 4.1
    epochs: int = 40  # Section 4.1
    weight_decay: float = 1e-4
    
    # === Loss Weights (Eq.14) - EXPERIMENT 3: Classification only ===
    lambda_cls: float = 1.0   # L_cls (classification)
    lambda_mim: float = 0.0   # DISABLED for experiment
    lambda_sis: float = 0.0   # DISABLED for experiment
    lambda_dal: float = 0.0   # DISABLED
    
    # === Training Strategy ===
    freeze_backbone: bool = True  # Freeze CLIP backbone (standard)
    grad_clip: float = 1.0  # Gradient clipping norm
    nan_skip_threshold: int = 10  # Stop if too many NaN batches
    
    # === Domain Names (for prompts) ===
    domain_names: Tuple[str, ...] = (
        "FaceForensics",
        "WildDeepfake", 
        "CelebDF",
        "DFDC",
        "DeepFakeFace",
    )


# Default configuration instance
default_config = GMDFConfig()
