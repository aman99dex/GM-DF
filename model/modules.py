"""
GM-DF Core Modules
Implements: DomainNorm (Eq.2), MoE_Expert, MoE_Adapter (Eq.1,4-5), SecondOrderAgg (Eq.7-8)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from config import GMDFConfig
except ImportError:
    from ..config import GMDFConfig


class DomainNorm(nn.Module):
    """
    Domain-Specific Normalization (Equation 2)
    
    DNorm(x, c_d) = ((x - μ) / σ) ⊙ (γ₀ + W_γ·c_d) + (β₀ + W_β·c_d)
    
    Where:
        - c_d ∈ R^k: Domain embedding for domain d
        - W_γ, W_β ∈ R^{hidden_dim × k}: Projection matrices
        - γ₀, β₀ ∈ R^hidden_dim: Base (domain-invariant) affine parameters
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        domain_embed_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.domain_embed_dim = domain_embed_dim
        self.eps = eps
        
        # Base affine parameters (domain-invariant)
        self.gamma_0 = nn.Parameter(torch.ones(hidden_dim))
        self.beta_0 = nn.Parameter(torch.zeros(hidden_dim))
        
        # Domain-conditioned projection matrices
        self.W_gamma = nn.Linear(domain_embed_dim, hidden_dim, bias=False)
        self.W_beta = nn.Linear(domain_embed_dim, hidden_dim, bias=False)
        
        # Initialize projections to small values for stability
        nn.init.normal_(self.W_gamma.weight, std=0.02)
        nn.init.normal_(self.W_beta.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, c_d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D) where B=batch, N=sequence, D=hidden_dim
            c_d: Domain embedding of shape (B, k) or (k,) for single domain
        
        Returns:
            Normalized tensor of shape (B, N, D)
        """
        # Ensure c_d has batch dimension
        if c_d.dim() == 1:
            c_d = c_d.unsqueeze(0).expand(x.size(0), -1)
        
        # Layer normalization: (x - μ) / σ
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Domain-conditioned scale and shift
        # W_γ·c_d and W_β·c_d: (B, k) -> (B, D)
        gamma_d = self.W_gamma(c_d)  # (B, D)
        beta_d = self.W_beta(c_d)    # (B, D)
        
        # Final affine: γ = γ₀ + W_γ·c_d, β = β₀ + W_β·c_d
        gamma = self.gamma_0 + gamma_d  # (B, D)
        beta = self.beta_0 + beta_d     # (B, D)
        
        # Apply scale and shift: (B, N, D) * (B, 1, D) + (B, 1, D)
        out = x_norm * gamma.unsqueeze(1) + beta.unsqueeze(1)
        
        return out


class MoE_Expert(nn.Module):
    """
    Single Expert Feed-Forward Network
    
    Standard 2-layer MLP matching ViT's FFN structure:
    Linear(hidden_dim → mlp_dim) → GELU → Linear(mlp_dim → hidden_dim)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D)
        Returns:
            Output tensor of shape (B, N, D)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ExpertRouter(nn.Module):
    """
    Attention-based Expert Routing (Equations 4-5)
    
    Computes attention over expert outputs:
    Q = K = Concat(ΔF_θ^1(x), ..., ΔF_θ^N(x))
    Attention = softmax(Q·K^T / √d_k) · V
    
    Where V is the stacked expert outputs.
    """
    
    def __init__(
        self,
        num_experts: int = 3,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Project expert outputs to routing dimension
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable scale factor
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(hidden_dim)))
    
    def forward(self, expert_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            expert_outputs: Tensor of shape (B, N, num_experts, D)
                           Stacked outputs from all experts
        
        Returns:
            aggregated: Weighted sum of expert outputs (B, N, D)
            routing_weights: Attention weights (B, N, num_experts)
        """
        B, N, E, D = expert_outputs.shape
        
        # Pool over sequence dimension for routing decisions
        # (B, N, E, D) -> (B, E, D) via mean pooling
        pooled = expert_outputs.mean(dim=1)  # (B, E, D)
        
        # Compute Q and K from pooled expert representations
        Q = self.query_proj(pooled)  # (B, E, D)
        K = self.key_proj(pooled)    # (B, E, D)
        
        # Attention scores: (B, E, D) @ (B, D, E) -> (B, E, E)
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) * self.scale
        
        # Sum attention across expert dimension to get routing weights
        # (B, E, E) -> (B, E) via softmax and sum
        routing_weights = F.softmax(attn_scores.mean(dim=-1), dim=-1)  # (B, E)
        
        # Expand routing weights for all positions
        routing_weights_expanded = routing_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, E, 1)
        
        # Weighted sum of expert outputs
        # (B, N, E, D) * (B, 1, E, 1) -> sum over E -> (B, N, D)
        aggregated = (expert_outputs * routing_weights_expanded).sum(dim=2)
        
        return aggregated, routing_weights


class MoE_Adapter(nn.Module):
    """
    Mixture of Experts Adapter (Equations 1, 4-5)
    
    F(x) = F_θ(x) + ΔF^n_θ(x)
    
    Where:
        - F_θ(x): Original (frozen) MLP output (domain-invariant)
        - ΔF^n_θ(x): Aggregated expert output (domain-specific)
    
    This module wraps the original MLP and adds expert-based adaptation.
    """
    
    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        num_experts: int = 3,
        domain_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Freeze original MLP
        for param in self.original_mlp.parameters():
            param.requires_grad = False
        
        # Create N expert networks
        self.experts = nn.ModuleList([
            MoE_Expert(hidden_dim, mlp_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Expert router with attention-based aggregation
        self.router = ExpertRouter(num_experts, hidden_dim)
        
        # Domain normalization (applied after MoE aggregation)
        self.domain_norm = DomainNorm(hidden_dim, domain_embed_dim)
        
        # Scaling factor for residual connection
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self, 
        x: torch.Tensor, 
        c_d: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            c_d: Domain embedding of shape (B, k) or None
        
        Returns:
            Output tensor of shape (B, N, D)
        """
        # Original MLP output (frozen, domain-invariant)
        with torch.no_grad():
            base_output = self.original_mlp(x)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: (B, N, E, D)
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Aggregate expert outputs via attention routing
        delta_output, routing_weights = self.router(expert_outputs)
        
        # Apply domain normalization if domain embedding provided
        if c_d is not None:
            delta_output = self.domain_norm(delta_output, c_d)
        
        # Final output: F(x) = F_θ(x) + α·ΔF^n_θ(x)
        output = base_output + self.alpha * delta_output
        
        return output
    
    def get_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get routing weights for analysis/visualization."""
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=2)
        _, routing_weights = self.router(expert_outputs)
        return routing_weights


class SecondOrderAgg(nn.Module):
    """
    Second-Order Feature Aggregation (Equations 7-8)
    
    Aggregates cross-layer features from ViT layers 8-10:
    
    Δ^l_so(x) = Σ_h Σ_i a^{l'}_h,i(x) · W^{l',h}_VO · MLP^l(z^l_i)
    
    v(x) = Proj(z^L_cls + Σ_{l=8}^{10} α_l · Norm(Δ^l_so(x)))
    
    Captures texture anomalies through feature correlations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 3,  # Layers 8, 9, 10
        output_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Learnable fusion weights α_l for each layer
        self.alpha = nn.Parameter(torch.ones(num_layers) * 0.1)
        
        # Layer normalization for each aggregated feature
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Projection to output dimension (Eq.8)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # For computing second-order statistics
        self.use_covariance = True
    
    def compute_second_order(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute second-order statistics (covariance-like features).
        
        Args:
            features: Tensor of shape (B, N, D) - spatial tokens
        
        Returns:
            Aggregated second-order features of shape (B, D)
        """
        B, N, D = features.shape
        
        # Center the features
        features_centered = features - features.mean(dim=1, keepdim=True)
        
        # Compute covariance-like features using batch matrix multiplication
        # (B, D, N) @ (B, N, D) -> (B, D, D)
        cov = torch.bmm(features_centered.transpose(1, 2), features_centered) / (N - 1)
        
        # Extract diagonal (variance) and mean of off-diagonal (correlation)
        # This is more memory efficient than full covariance
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (B, D)
        
        # Add mean pooling for first-order
        first_order = features.mean(dim=1)  # (B, D)
        
        # Combine first and second order
        combined = first_order + 0.1 * diag
        
        return combined
    
    def forward(
        self,
        layer_features: list,  # List of (B, N, D) tensors from layers 8, 9, 10
        cls_token: torch.Tensor,  # (B, D) - final CLS token
    ) -> torch.Tensor:
        """
        Args:
            layer_features: List of feature tensors from intermediate layers
                           Each tensor is (B, N, D) where N includes CLS token
            cls_token: Final layer CLS token (B, D)
        
        Returns:
            Visual features v(x) of shape (B, output_dim)
        """
        assert len(layer_features) == self.num_layers, \
            f"Expected {self.num_layers} layer features, got {len(layer_features)}"
        
        # Aggregate second-order features from each layer
        aggregated = cls_token.clone()  # Start with z^L_cls
        
        for i, (features, norm) in enumerate(zip(layer_features, self.layer_norms)):
            # Extract spatial tokens (exclude CLS)
            spatial_tokens = features[:, 1:, :]  # (B, N-1, D)
            
            # Compute second-order features
            second_order = self.compute_second_order(spatial_tokens)  # (B, D)
            
            # Normalize and weight
            second_order_norm = norm(second_order)
            aggregated = aggregated + self.alpha[i] * second_order_norm
        
        # Final projection
        output = self.proj(aggregated)
        
        return output


class DomainEmbedding(nn.Module):
    """
    Learnable domain embeddings c_d for each domain.
    Used in DomainNorm and prompt construction.
    """
    
    def __init__(
        self,
        num_domains: int = 5,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.embed_dim = embed_dim
        
        # Learnable embeddings for each domain
        self.embeddings = nn.Embedding(num_domains, embed_dim)
        
        # Initialize with small values
        nn.init.normal_(self.embeddings.weight, std=0.02)
    
    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            domain_ids: Tensor of domain indices (B,) with values in [0, num_domains)
        
        Returns:
            Domain embeddings of shape (B, embed_dim)
        """
        return self.embeddings(domain_ids)


# For convenience, export a function to create the full MoE block
def create_moe_adapter(
    original_mlp: nn.Module,
    config: Optional[GMDFConfig] = None,
) -> MoE_Adapter:
    """Create a MoE_Adapter that wraps the original MLP."""
    if config is None:
        config = GMDFConfig()
    
    return MoE_Adapter(
        original_mlp=original_mlp,
        hidden_dim=config.hidden_dim,
        mlp_dim=config.mlp_dim,
        num_experts=config.num_experts,
        domain_embed_dim=config.domain_embed_dim,
    )
