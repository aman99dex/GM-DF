"""
GM-DF Configuration
Hyperparameters from Section 4.1 of the paper
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
    
    # === Dataset-Embedding Generator (DEG) ===
    num_experts: int = 3  # N independent experts (Section 3.1)
    domain_embed_dim: int = 64  # k dimension for c_d in Eq.2
    num_domains: int = 5  # FF++, WDF, Celeb-DF, DFDC, DFF
    
    # === Multi-Dataset Representation (MDP) ===
    second_order_layers: Tuple[int, ...] = (8, 9, 10)  # Layers for Eq.7-8
    alpha_init: float = 0.1  # Initial α_l for second-order fusion
    
    # === Masked Image Modeling (MIM) ===
    # === Masked Image Modeling (MIM) ===
    # Using Pixel Regression (MAE style) - no vocab needed
    mask_ratio: float = 0.50  # Increased for MAE (usually 50-75%)
    mim_decoder_layers: int = 4  # Slightly deeper for reconstruction
    mim_decoder_dim: int = 384  # Decoder hidden dimension
    
    # === Prompt Learning ===
    prompt_length: int = 4  # Learnable context tokens
    tau_base: float = 0.07  # Base temperature (CLIP default)
    tau_phi: float = 0.01  # Temperature scaling factor φ in Eq.10
    
    # === MAML Training (MDEO) ===
    # Inner: High LR (fast adaptation)
    # Outer: Low LR (stable meta-learning)
    inner_lr: float = 1e-3  # β (Inner loop needs to affect weights significantly)
    outer_lr: float = 1e-4  # δ (Outer loop updates initialization slowly)
    inner_steps: int = 1  # Inner loop gradient steps
    
    # === General Training ===
    batch_size: int = 32  # Section 4.1
    epochs: int = 40  # Section 4.1
    weight_decay: float = 1e-4
    
    # === Loss Weights ===
    lambda_mim: float = 1.0  # Weight for L_mim
    lambda_sis: float = 1.0  # Weight for L_sis (contrastive)
    
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
