"""
GM-DF Main Model
Implements: GMDF_Detector, PromptLearner (Eq.9-11), MIM_Decoder (Eq.6)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from collections import OrderedDict

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

try:
    from config import GMDFConfig
except ImportError:
    from ..config import GMDFConfig

from .modules import (
    DomainNorm, 
    MoE_Adapter, 
    SecondOrderAgg, 
    DomainEmbedding,
    create_moe_adapter,
)


class PromptLearner(nn.Module):
    """
    Domain-Class Disentangled Dynamic Prompts (Equations 9-11)
    
    T(d, c) = "A [e^d_dom] [e^c_cls] face showing forgery clues"
    
    Where:
        - e^d_dom ∈ R^512: Domain-specific embedding
        - e^c_cls ∈ {e_real, e_fake}: Class embedding
        - Adaptive temperature: τ_d = τ_base + φ·||e^d_dom||₂ (Eq.10)
    """
    
    def __init__(
        self,
        clip_model: nn.Module,
        num_domains: int = 5,
        domain_names: Tuple[str, ...] = None,
        prompt_length: int = 4,
        tau_base: float = 0.07,
        tau_phi: float = 0.01,
    ):
        super().__init__()
        
        # Get CLIP text encoder parameters
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # Vocabulary size and embedding dimension
        self.embed_dim = clip_model.token_embedding.embedding_dim  # 512 for ViT-B/16
        self.context_length = clip_model.context_length  # 77
        
        # Temperature parameters (Eq.10)
        self.tau_base = tau_base
        self.tau_phi = tau_phi
        
        # Domain names for prompt construction
        if domain_names is None:
            domain_names = (
                "FaceForensics",
                "WildDeepfake", 
                "CelebDF",
                "DFDC",
                "DeepFakeFace",
            )
        self.domain_names = domain_names
        self.num_domains = num_domains
        
        # Learnable domain embeddings e^d_dom
        self.domain_embeddings = nn.Parameter(
            torch.randn(num_domains, self.embed_dim) * 0.02
        )
        
        # Wildcard domain embedding e^0_dom for zero-shot
        self.wildcard_embedding = nn.Parameter(
            torch.randn(1, self.embed_dim) * 0.02
        )
        
        # Class embeddings e_real and e_fake
        self.class_embeddings = nn.Parameter(
            torch.randn(2, self.embed_dim) * 0.02  # [real, fake]
        )
        
        # Learnable context tokens (prefix)
        self.prompt_length = prompt_length
        self.context_vectors = nn.Parameter(
            torch.randn(prompt_length, self.embed_dim) * 0.02
        )
        
        # Template tokens (will be populated during forward)
        self._init_template_tokens(clip_model)
    
    def _init_template_tokens(self, clip_model):
        """Initialize fixed template token embeddings."""
        # Template: "A [DOMAIN] [CLASS] face showing forgery clues"
        # We'll construct: [context] [domain] [class] [face showing forgery clues]
        
        template_text = "face showing forgery clues"
        template_tokens = clip.tokenize([template_text])[0]  # (77,)
        
        # Move tokens to same device as clip model
        device = next(clip_model.parameters()).device
        template_tokens = template_tokens.to(device)
        
        with torch.no_grad():
            template_embed = clip_model.token_embedding(template_tokens)  # (77, 512)
        
        # Find actual token length (before padding)
        eot_idx = (template_tokens == 49407).nonzero(as_tuple=True)[0][0].item()
        
        # Store template embeddings (excluding SOS and padding)
        self.register_buffer(
            "template_embeddings",
            template_embed[1:eot_idx].clone()  # Exclude SOS token
        )
        
        # SOS and EOS tokens
        self.register_buffer("sos_token", template_embed[0:1].clone())
        self.register_buffer("eos_token", template_embed[eot_idx:eot_idx+1].clone())
    
    def construct_prompts(
        self,
        domain_ids: torch.Tensor,  # (B,)
        class_ids: torch.Tensor,   # (B,) - 0=real, 1=fake
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct prompt embeddings for given domains and classes.
        
        Returns:
            prompts: Token embeddings (B, context_length, embed_dim)
            prompt_lengths: Actual lengths for each prompt (B,)
        """
        B = domain_ids.size(0)
        device = domain_ids.device
        
        # Get domain and class embeddings
        domain_emb = self.domain_embeddings[domain_ids]  # (B, 512)
        class_emb = self.class_embeddings[class_ids]     # (B, 512)
        
        # Expand context vectors for batch
        context = self.context_vectors.unsqueeze(0).expand(B, -1, -1)  # (B, L, 512)
        
        # Construct full prompt:
        # [SOS] [context] [domain] [class] [template] [EOS] [padding...]
        sos = self.sos_token.unsqueeze(0).expand(B, -1, -1)  # (B, 1, 512)
        template = self.template_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, T, 512)
        eos = self.eos_token.unsqueeze(0).expand(B, -1, -1)  # (B, 1, 512)
        
        # Concatenate all parts
        prompt_parts = [
            sos,                          # (B, 1, 512)
            context,                      # (B, L, 512)
            domain_emb.unsqueeze(1),      # (B, 1, 512)
            class_emb.unsqueeze(1),       # (B, 1, 512)
            template,                     # (B, T, 512)
            eos,                          # (B, 1, 512)
        ]
        
        prompts = torch.cat(prompt_parts, dim=1)  # (B, 1+L+1+1+T+1, 512)
        actual_length = prompts.size(1)
        
        # Pad to context_length
        if actual_length < self.context_length:
            padding = torch.zeros(
                B, self.context_length - actual_length, self.embed_dim,
                device=device, dtype=prompts.dtype
            )
            prompts = torch.cat([prompts, padding], dim=1)
        else:
            prompts = prompts[:, :self.context_length, :]
            actual_length = self.context_length
        
        # Create length tensor
        prompt_lengths = torch.full((B,), actual_length, device=device, dtype=torch.long)
        
        return prompts, prompt_lengths
    
    def encode_text(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Encode prompt embeddings through CLIP's text transformer.
        
        Args:
            prompts: (B, context_length, embed_dim)
        
        Returns:
            Text features (B, embed_dim)
        """
        # Add positional embeddings
        x = prompts + self.positional_embedding.to(prompts.dtype)
        
        # Permute for transformer: (B, L, D) -> (L, B, D)
        x = x.permute(1, 0, 2)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Permute back: (L, B, D) -> (B, L, D)
        x = x.permute(1, 0, 2)
        
        # Apply layer norm
        x = self.ln_final(x)
        
        # Take features from EOS token position (last non-padding token)
        # For simplicity, use the position after template tokens
        eos_pos = 1 + self.prompt_length + 1 + 1 + self.template_embeddings.size(0)
        x = x[:, min(eos_pos, x.size(1) - 1), :]
        
        # Project to output dimension
        if self.text_projection is not None:
            x = x @ self.text_projection
        
        return x
    
    def get_adaptive_temperature(self, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive temperature τ_d (Equation 10).
        
        τ_d = τ_base + φ·||e^d_dom||₂
        """
        domain_emb = self.domain_embeddings[domain_ids]  # (B, 512)
        domain_norm = torch.norm(domain_emb, dim=-1)     # (B,)
        tau_d = self.tau_base + self.tau_phi * domain_norm
        return tau_d
    
    def forward(
        self,
        domain_ids: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get text features for given domain-class pairs.
        
        Returns:
            text_features: (B, embed_dim)
            temperatures: (B,) - adaptive temperatures
        """
        prompts, _ = self.construct_prompts(domain_ids, class_ids)
        text_features = self.encode_text(prompts.to(self.dtype))
        temperatures = self.get_adaptive_temperature(domain_ids)
        
        return text_features, temperatures


class MIM_Decoder(nn.Module):
    """
    Masked Image Modeling Decoder with Pixel Regression (MAE Style)
    
    Predicts normalized pixel values for masked patches:
    L_mim = MeanSquaredError(pred_pixels, target_pixels)
    
    This avoids the need for a pretrained tokenizer (which was missing/random in previous ver).
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        decoder_dim: int = 384,
        num_layers: int = 4,  # Increased depth for better reconstruction
        num_heads: int = 6,
        patch_size: int = 16,
        mask_ratio: float = 0.50,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Project from encoder dim to decoder dim
        self.encoder_to_decoder = nn.Linear(hidden_dim, decoder_dim)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        
        # Position embedding for decoder (learnable)
        # We need this because we are adding mask tokens back at specific positions
        # Size: (14x14) + 1 (cls) -> 197 for 224x224
        # We'll initialize dynamically or assume standard size
        self.num_patches = (224 // patch_size) ** 2
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, decoder_dim) * 0.02
        )
        
        # Lightweight decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # MAE uses norm_first typically
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Pixel predictor head
        # Projects to (patch_size * patch_size * 3) for RGB reconstruction
        self.output_dim = patch_size * patch_size * 3
        self.pixel_head = nn.Linear(decoder_dim, self.output_dim)
        
        # Normalization
        self.decoder_norm = nn.LayerNorm(decoder_dim)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        (B, 3, H, W) -> (B, N, patch_size**2 * 3)
        """
        p = self.patch_size
        assert images.shape[2] == images.shape[3] and images.shape[2] % p == 0
        
        h = w = images.shape[2] // p
        x = images.reshape(shape=(images.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(images.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.
        (B, N, patch_size**2 * 3) -> (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        images = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return images
    
    def forward(
        self,
        encoder_features: torch.Tensor,  # (B, N_visible+1, hidden_dim) - including CLS
        original_images: torch.Tensor,   # (B, 3, H, W)
        mask: Optional[torch.Tensor] = None, # (B, N) boolean mask where True=masked/removed
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE decoder.
        
        Note: The encoder_features here are technically the FULL set of features from CLIP.
        MAE usually operates on ONLY visible patches in the encoder.
        However, since we are using a frozen CLIP backbone, we get full features.
        We will simulate MAE by masking the features here before decoder.
        
        Args:
            encoder_features: Output from CLIP encoder (B, N+1, D)
            original_images: Original images (B, 3, H, W)
            mask: Optional mask
        
        Returns:
            loss: scalar
            pred_pixels: (B, N, 3*P*P)
            mask: (B, N)
        """
        B, N_plus_1, D = encoder_features.size()
        num_patches = N_plus_1 - 1
        device = encoder_features.device
        
        # 1. Prepare Target (Patches)
        target = self.patchify(original_images)
        
        # Normalize target patches (per patch) - common in MAE
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        
        # 2. Handle Mask
        # If mask is not provided, create one
        if mask is None:
            # Create random mask
            noise = torch.rand(B, num_patches, device=device)
            len_keep = int(num_patches * (1 - self.mask_ratio))
            
            # Sort noise to get indices
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # Mask: 1 is removed, 0 is kept (MAE convention is usually other way or indices, but let's stick to boolean)
            # Actually, let's use boolean where True = MASKED (lost)
            mask = torch.ones([B, num_patches], device=device)
            mask[:, :len_keep] = 0
            # Unshuffle to get mask in original order
            mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        
        # 3. Prepare Decoder Input
        # Get patch features (exclude CLS)
        x = encoder_features[:, 1:, :] # (B, N, D)
        
        # Project to decoder dim
        x = self.encoder_to_decoder(x) # (B, N, decoder_dim)
        
        # Apply mask: replace masked positions with MASK token
        mask_tokens = self.mask_token.expand(B, num_patches, -1)
        
        # If mask is boolean (approximating MAE with full tokens):
        # We replace features at masked positions with learnable mask token
        # This is slightly different from standard MAE (which drops tokens), but standard for BERT-style MIM on fixed backbones (BEiT style)
        x_full = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        
        # Add position embeddings
        if self.decoder_pos_embed.shape[1] != num_patches:
             # Resize pos embed if needed (e.g. different image size)
             # Simple interpolation
             pos_embed = F.interpolate(
                 self.decoder_pos_embed.permute(0, 2, 1), 
                 size=num_patches, 
                 mode='linear'
             ).permute(0, 2, 1)
        else:
             pos_embed = self.decoder_pos_embed
             
        x_full = x_full + pos_embed
        
        # 4. Decode
        x_decoded = self.decoder(x_full)
        x_decoded = self.decoder_norm(x_decoded)
        
        # 5. Predict
        pred = self.pixel_head(x_decoded) # (B, N, P*P*3)
        
        # 6. Compute Loss
        # Loss is only calculated on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean over pixels
        
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on masked patches
        
        return loss, pred, mask


class GMDF_Detector(nn.Module):
    """
    GM-DF: Generalized Multi-Scenario Deepfake Detection
    
    Main detector class that:
    1. Loads CLIP ViT-B/16 as frozen backbone
    2. Injects MoE_Adapter into transformer blocks (Eq.1-3)
    3. Extracts second-order features from layers 8-10 (Eq.7-8)
    4. Uses MIM decoder for reconstruction (Eq.6)
    5. Uses dynamic prompts for contrastive learning (Eq.9-11)
    """
    
    def __init__(
        self,
        config: Optional[GMDFConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        if config is None:
            config = GMDFConfig()
        self.config = config
        self.device = device
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(
            config.backbone, 
            device=device,
            jit=False
        )
        self.clip_model.float()  # Use float32 for training
        
        # Freeze/Unfreeze CLIP backbone based on config
        # Freezing is recommended to preserve pretrained features
        for param in self.clip_model.parameters():
            param.requires_grad = not config.freeze_backbone
        
        # Get vision transformer
        self.vision_transformer = self.clip_model.visual
        
        # Domain embeddings
        self.domain_embedding = DomainEmbedding(
            num_domains=config.num_domains,
            embed_dim=config.domain_embed_dim,
        )
        
        # Inject MoE adapters into transformer blocks
        self._inject_moe_adapters()
        
        # Second-order feature aggregation
        self.second_order_agg = SecondOrderAgg(
            hidden_dim=config.hidden_dim,
            num_layers=len(config.second_order_layers),
            output_dim=512,  # CLIP's output dim
        )
        
        # MIM decoder (MAE Style)
        self.mim_decoder = MIM_Decoder(
            hidden_dim=config.hidden_dim,
            decoder_dim=config.mim_decoder_dim,
            num_layers=config.mim_decoder_layers,
            patch_size=config.patch_size,
            mask_ratio=config.mask_ratio,
        )
        
        # Prompt learner
        self.prompt_learner = PromptLearner(
            clip_model=self.clip_model,
            num_domains=config.num_domains,
            domain_names=config.domain_names,
            prompt_length=config.prompt_length,
            tau_base=config.tau_base,
            tau_phi=config.tau_phi,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Binary: real (0) vs fake (1)
        )
        
        # Storage for intermediate features
        self.intermediate_features = {}
        self._register_hooks()
    
    def _inject_moe_adapters(self):
        """
        Inject MoE adapters into CLIP's transformer blocks.
        
        Replaces the MLP in each block with MoE_Adapter wrapper:
        x_ffn = DNorm(MoE_Adapter(x_att), c_d) + x_att  (Eq.3)
        """
        # Access transformer blocks
        transformer = self.vision_transformer.transformer
        
        self.moe_adapters = nn.ModuleList()
        
        for i, block in enumerate(transformer.resblocks):
            # Get original MLP
            original_mlp = block.mlp
            
            # Create MoE adapter wrapping the original MLP
            moe_adapter = create_moe_adapter(original_mlp, self.config)
            self.moe_adapters.append(moe_adapter)
            
            # Replace MLP with adapter
            block.mlp = moe_adapter
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        transformer = self.vision_transformer.transformer
        
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.intermediate_features[layer_idx] = output.clone()
            return hook
        
        # Register hooks for layers 8, 9, 10
        for layer_idx in self.config.second_order_layers:
            if layer_idx < len(transformer.resblocks):
                transformer.resblocks[layer_idx].register_forward_hook(
                    get_hook(layer_idx)
                )
    
    def encode_image(
        self,
        images: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Encode images through CLIP vision encoder with MoE adaptation.
        
        Returns:
            cls_features: CLS token features (B, 512)
            all_features: All token features (B, N+1, 768)
            layer_features: Features from second-order layers
        """
        # Get domain embeddings
        c_d = self.domain_embedding(domain_ids)  # (B, domain_embed_dim)
        
        # Patch embedding
        x = self.vision_transformer.conv1(images)  # (B, 768, H/16, W/16)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, 768, N)
        x = x.permute(0, 2, 1)  # (B, N, 768)
        
        # Add CLS token
        cls_token = self.vision_transformer.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, 768)
        
        # Add positional embedding
        x = x + self.vision_transformer.positional_embedding.to(x.dtype)
        
        # Pre-layer norm
        x = self.vision_transformer.ln_pre(x)
        
        # Pass through transformer blocks
        # Permute for transformer: (B, N, D) -> (N, B, D)
        x = x.permute(1, 0, 2)
        
        # Clear intermediate features
        self.intermediate_features = {}
        
        # Forward through transformer with MoE (adapters receive domain embedding)
        for i, block in enumerate(self.vision_transformer.transformer.resblocks):
            # Pass domain embedding to MoE adapter
            if hasattr(block.mlp, 'domain_norm'):
                # Attention
                attn_out = block.attn(block.ln_1(x), block.ln_1(x), block.ln_1(x), need_weights=False)[0]
                x = x + attn_out
                
                # MLP with domain conditioning
                ln_out = block.ln_2(x)
                # Permute for batch-first: (N, B, D) -> (B, N, D)
                ln_out_bf = ln_out.permute(1, 0, 2)
                mlp_out = block.mlp(ln_out_bf, c_d)
                # Permute back: (B, N, D) -> (N, B, D)
                mlp_out = mlp_out.permute(1, 0, 2)
                x = x + mlp_out
            else:
                x = block(x)
            
            # Store intermediate features for second-order aggregation
            if i in self.config.second_order_layers:
                self.intermediate_features[i] = x.permute(1, 0, 2).clone()
        
        # Permute back: (N, B, D) -> (B, N, D)
        x = x.permute(1, 0, 2)
        
        # Post-layer norm
        x = self.vision_transformer.ln_post(x)
        
        # Get CLS token and project
        cls_token_out = x[:, 0, :]  # (B, 768)
        if self.vision_transformer.proj is not None:
            cls_features = cls_token_out @ self.vision_transformer.proj  # (B, 512)
        else:
            cls_features = cls_token_out
        
        # Collect layer features for second-order aggregation
        layer_features = [
            self.intermediate_features[i] 
            for i in self.config.second_order_layers 
            if i in self.intermediate_features
        ]
        
        return cls_features, x, layer_features, cls_token_out
    
    def _compute_domain_alignment_loss(
        self,
        features: torch.Tensor,  # (B, D)
        domain_ids: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Compute Domain Alignment Loss using Maximum Mean Discrepancy (MMD).
        
        Aligns feature distributions across different domains to learn 
        domain-invariant representations.
        
        MMD(P, Q) = ||μ_P - μ_Q||² + ||Σ_P - Σ_Q||_F
        
        For computational efficiency, we use a simplified version:
        L_dal = Σ_{d1 ≠ d2} ||μ_{d1} - μ_{d2}||²
        
        Args:
            features: Visual features (B, D)
            domain_ids: Domain indices (B,)
        
        Returns:
            loss_dal: Scalar loss
        """
        unique_domains = domain_ids.unique()
        
        # Need at least 2 domains for alignment
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Compute mean features per domain
        domain_means = []
        for d in unique_domains:
            mask = domain_ids == d
            if mask.sum() > 0:
                domain_mean = features[mask].mean(dim=0)  # (D,)
                domain_means.append(domain_mean)
        
        # Compute pairwise MMD (simplified: mean distance)
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        for i in range(len(domain_means)):
            for j in range(i + 1, len(domain_means)):
                # L2 distance between means
                loss = loss + F.mse_loss(domain_means[i], domain_means[j])
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def forward(
        self,
        images: torch.Tensor,
        domain_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for GM-DF detector.
        
        Args:
            images: Input images (B, 3, 224, 224)
            domain_ids: Domain indices (B,)
            labels: Ground truth labels (B,) for training, 0=real, 1=fake
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with:
                - logits: Classification logits (B, 1)
                - loss_total: Total loss (if labels provided)
                - loss_cls: Classification loss
                - loss_mim: MIM reconstruction loss
                - loss_sis: Contrastive loss
                - visual_features: Visual features (if return_features)
                - text_features: Text features (if return_features)
        """
        outputs = {}
        
        # Encode images
        cls_features, all_features, layer_features, cls_token_768 = self.encode_image(images, domain_ids)
        
        # Second-order feature aggregation (Eq.7-8) - uses 768D CLS token
        visual_features = self.second_order_agg(layer_features, cls_token_768)  # (B, 512)
        
        # Classification
        logits = self.classifier(visual_features)  # (B, 1)
        outputs["logits"] = logits
        
        if labels is not None:
            # Classification loss (Eq.12)
            loss_cls = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), 
                labels.float()
            )
            outputs["loss_cls"] = loss_cls
            
            # MIM loss (Eq.6)
            loss_mim, _, _ = self.mim_decoder(all_features, images)
            outputs["loss_mim"] = loss_mim
            
            # Contrastive loss (Eq.11) - InfoNCE formulation
            # Get text features for all domain-class combinations
            text_features, temperatures = self.prompt_learner(domain_ids, labels.long())
            
            # Normalize features
            visual_features_norm = F.normalize(visual_features, dim=-1)
            text_features_norm = F.normalize(text_features, dim=-1)
            
            # Compute pairwise similarity matrix (B, B)
            # Each row: similarity of visual[i] with all text features
            logits = torch.matmul(visual_features_norm, text_features_norm.T)  # (B, B)
            
            # Scale by temperature (use mean temperature for batch)
            temperature = temperatures.mean().clamp(min=0.01)  # Prevent division by very small values
            logits = logits / temperature
            
            # InfoNCE loss: diagonal entries are positive pairs
            # Labels: each visual[i] should match text[i]
            batch_labels = torch.arange(visual_features_norm.size(0), device=visual_features_norm.device)
            loss_sis = F.cross_entropy(logits, batch_labels)
            outputs["loss_sis"] = loss_sis
            
            # Domain Alignment Loss (L_dal) using MMD
            # Encourages domain-invariant feature learning
            loss_dal = self._compute_domain_alignment_loss(visual_features, domain_ids)
            outputs["loss_dal"] = loss_dal
            
            # Clamp individual losses to prevent NaN propagation
            loss_cls = torch.clamp(loss_cls, min=0.0, max=100.0)
            loss_mim = torch.clamp(loss_mim, min=0.0, max=100.0)
            loss_sis = torch.clamp(loss_sis, min=0.0, max=100.0)
            loss_dal = torch.clamp(loss_dal, min=0.0, max=100.0)
            
            # Total loss (Eq.14 + L_dal)
            loss_total = (
                loss_cls + 
                self.config.lambda_mim * loss_mim + 
                self.config.lambda_sis * loss_sis +
                self.config.lambda_dal * loss_dal
            )
            outputs["loss_total"] = loss_total
        
        if return_features:
            outputs["visual_features"] = visual_features
            if labels is not None:
                outputs["text_features"] = text_features
        
        return outputs
    
    def get_trainable_params(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get parameter groups for MAML training.
        
        Returns:
            Dictionary with:
                - theta_E: MoE expert parameters (inner loop)
                - theta_O: Other trainable parameters (outer loop)
        """
        theta_E = []  # Expert parameters
        theta_O = []  # Other parameters
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if "moe_adapter" in name or "experts" in name or "router" in name:
                theta_E.append(param)
            else:
                theta_O.append(param)
        
        return {"theta_E": theta_E, "theta_O": theta_O}
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict real/fake labels for images.
        
        Returns:
            probabilities: (B,) tensor with fake probabilities
        """
        outputs = self.forward(images, domain_ids)
        probs = torch.sigmoid(outputs["logits"].squeeze(-1))
        return probs


def build_model(
    config: Optional[GMDFConfig] = None, 
    device: Optional[str] = None,
    verbose: bool = True,
) -> GMDF_Detector:
    """
    Factory function to build GM-DF model with automatic device detection.
    
    Args:
        config: Model configuration
        device: Device preference ("cuda", "mps", "cpu", or "auto"/None for auto-detect)
        verbose: Print device information
    
    Returns:
        GMDF_Detector model on optimal device
    """
    # Use device utilities for automatic detection
    try:
        from utils import get_optimal_device, print_device_summary
        device_info = get_optimal_device(preferred=device, verbose=False)
        if verbose:
            print_device_summary(device_info)
        device = device_info.device
    except ImportError:
        # Fallback if utils not available
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if verbose:
            print(f"Using device: {device}")
        device = torch.device(device)
    
    model = GMDF_Detector(config=config, device=device)
    return model.to(device)

