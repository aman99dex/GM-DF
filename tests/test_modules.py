"""
GM-DF Unit Tests
Tests for shape verification and forward pass validation
"""

import sys
import torch
import pytest

sys.path.insert(0, "..")
from config import GMDFConfig


class TestDomainNorm:
    """Tests for DomainNorm module."""
    
    def test_shape(self):
        """Test that output shape matches input shape."""
        from model.modules import DomainNorm
        
        hidden_dim = 768
        domain_embed_dim = 64
        batch_size = 4
        seq_len = 197  # 196 patches + 1 CLS token
        
        norm = DomainNorm(hidden_dim, domain_embed_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        c_d = torch.randn(batch_size, domain_embed_dim)
        
        out = norm(x, c_d)
        
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_single_domain_embedding(self):
        """Test with single (unbatched) domain embedding."""
        from model.modules import DomainNorm
        
        hidden_dim = 768
        domain_embed_dim = 64
        batch_size = 4
        seq_len = 197
        
        norm = DomainNorm(hidden_dim, domain_embed_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        c_d = torch.randn(domain_embed_dim)  # Single embedding
        
        out = norm(x, c_d)
        
        assert out.shape == x.shape


class TestMoEExpert:
    """Tests for MoE_Expert module."""
    
    def test_shape(self):
        """Test expert output shape."""
        from model.modules import MoE_Expert
        
        hidden_dim = 768
        mlp_dim = 3072
        batch_size = 4
        seq_len = 197
        
        expert = MoE_Expert(hidden_dim, mlp_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        out = expert(x)
        
        assert out.shape == x.shape


class TestExpertRouter:
    """Tests for ExpertRouter module."""
    
    def test_shape(self):
        """Test router output shapes."""
        from model.modules import ExpertRouter
        
        num_experts = 3
        hidden_dim = 768
        batch_size = 4
        seq_len = 197
        
        router = ExpertRouter(num_experts, hidden_dim)
        expert_outputs = torch.randn(batch_size, seq_len, num_experts, hidden_dim)
        
        aggregated, weights = router(expert_outputs)
        
        assert aggregated.shape == (batch_size, seq_len, hidden_dim)
        assert weights.shape == (batch_size, num_experts)
    
    def test_weights_sum_to_one(self):
        """Test that routing weights sum to 1."""
        from model.modules import ExpertRouter
        
        router = ExpertRouter(num_experts=3, hidden_dim=768)
        expert_outputs = torch.randn(2, 197, 3, 768)
        
        _, weights = router(expert_outputs)
        weight_sums = weights.sum(dim=-1)
        
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


class TestSecondOrderAgg:
    """Tests for SecondOrderAgg module."""
    
    def test_shape(self):
        """Test second-order aggregation output shape."""
        from model.modules import SecondOrderAgg
        
        hidden_dim = 768
        num_layers = 3
        output_dim = 512
        batch_size = 4
        seq_len = 197
        
        agg = SecondOrderAgg(hidden_dim, num_layers, output_dim)
        
        layer_features = [
            torch.randn(batch_size, seq_len, hidden_dim)
            for _ in range(num_layers)
        ]
        cls_token = torch.randn(batch_size, hidden_dim)
        
        out = agg(layer_features, cls_token)
        
        assert out.shape == (batch_size, output_dim)


class TestDomainEmbedding:
    """Tests for DomainEmbedding module."""
    
    def test_shape(self):
        """Test domain embedding output shape."""
        from model.modules import DomainEmbedding
        
        num_domains = 5
        embed_dim = 64
        batch_size = 4
        
        emb = DomainEmbedding(num_domains, embed_dim)
        domain_ids = torch.randint(0, num_domains, (batch_size,))
        
        out = emb(domain_ids)
        
        assert out.shape == (batch_size, embed_dim)


class TestMIMDecoder:
    """Tests for MIM_Decoder module."""
    
    def test_shape(self):
        """Test MIM decoder output shapes."""
        from model.gm_df import MIM_Decoder
        
        hidden_dim = 768
        decoder_dim = 384
        vocab_size = 8192
        batch_size = 2
        num_patches = 196  # 14x14
        
        decoder = MIM_Decoder(
            hidden_dim=hidden_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
        )
        
        encoder_features = torch.randn(batch_size, num_patches + 1, hidden_dim)
        images = torch.randn(batch_size, 3, 224, 224)
        
        loss, pred_logits, mask = decoder(encoder_features, images)
        
        assert pred_logits.shape == (batch_size, num_patches, vocab_size)
        assert mask.shape == (batch_size, num_patches)
        assert loss.dim() == 0  # Scalar


class TestPromptLearner:
    """Tests for PromptLearner module (requires CLIP)."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(), 
        reason="CLIP requires GPU for efficient testing"
    )
    def test_shape(self):
        """Test prompt learner output shapes."""
        import clip
        from model.gm_df import PromptLearner
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
        
        batch_size = 2
        num_domains = 5
        
        learner = PromptLearner(
            clip_model=clip_model,
            num_domains=num_domains,
        ).to(device)
        
        domain_ids = torch.randint(0, num_domains, (batch_size,)).to(device)
        class_ids = torch.randint(0, 2, (batch_size,)).to(device)
        
        text_features, temperatures = learner(domain_ids, class_ids)
        
        assert text_features.shape == (batch_size, 512)
        assert temperatures.shape == (batch_size,)


class TestGMDFDetector:
    """Integration tests for full GMDF_Detector (requires CLIP)."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CLIP requires GPU for efficient testing"
    )
    def test_forward_pass(self):
        """Test full forward pass of GMDF_Detector."""
        from model.gm_df import build_model
        
        config = GMDFConfig(num_domains=5)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = build_model(config=config, device=device)
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        domain_ids = torch.randint(0, 5, (batch_size,)).to(device)
        labels = torch.randint(0, 2, (batch_size,)).to(device)
        
        outputs = model(images, domain_ids, labels)
        
        assert "logits" in outputs
        assert "loss_total" in outputs
        assert "loss_cls" in outputs
        assert "loss_mim" in outputs
        assert "loss_sis" in outputs
        
        assert outputs["logits"].shape == (batch_size, 1)
        assert outputs["loss_total"].dim() == 0
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CLIP requires GPU for efficient testing"
    )
    def test_predict(self):
        """Test prediction mode."""
        from model.gm_df import build_model
        
        config = GMDFConfig(num_domains=5)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = build_model(config=config, device=device)
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        domain_ids = torch.randint(0, 5, (batch_size,)).to(device)
        
        probs = model.predict(images, domain_ids)
        
        assert probs.shape == (batch_size,)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CLIP requires GPU for efficient testing"
    )
    def test_trainable_params(self):
        """Test parameter grouping for MAML."""
        from model.gm_df import build_model
        
        config = GMDFConfig(num_domains=5)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = build_model(config=config, device=device)
        param_groups = model.get_trainable_params()
        
        assert "theta_E" in param_groups
        assert "theta_O" in param_groups
        assert len(param_groups["theta_E"]) > 0
        assert len(param_groups["theta_O"]) > 0


# Quick sanity check for CPU-only testing
class TestModulesNoGPU:
    """Tests that run on CPU without CLIP."""
    
    def test_all_modules_instantiate(self):
        """Test that all core modules can be instantiated."""
        from model.modules import (
            DomainNorm,
            MoE_Expert,
            ExpertRouter,
            MoE_Adapter,
            SecondOrderAgg,
            DomainEmbedding,
        )
        
        # DomainNorm
        norm = DomainNorm(768, 64)
        assert norm is not None
        
        # MoE_Expert
        expert = MoE_Expert(768, 3072)
        assert expert is not None
        
        # ExpertRouter
        router = ExpertRouter(3, 768)
        assert router is not None
        
        # SecondOrderAgg
        agg = SecondOrderAgg(768, 3, 512)
        assert agg is not None
        
        # DomainEmbedding
        emb = DomainEmbedding(5, 64)
        assert emb is not None


if __name__ == "__main__":
    # Run quick tests
    print("Running CPU tests...")
    
    test = TestModulesNoGPU()
    test.test_all_modules_instantiate()
    print("✓ All modules instantiate correctly")
    
    test = TestDomainNorm()
    test.test_shape()
    test.test_single_domain_embedding()
    print("✓ DomainNorm tests passed")
    
    test = TestMoEExpert()
    test.test_shape()
    print("✓ MoE_Expert tests passed")
    
    test = TestExpertRouter()
    test.test_shape()
    test.test_weights_sum_to_one()
    print("✓ ExpertRouter tests passed")
    
    test = TestSecondOrderAgg()
    test.test_shape()
    print("✓ SecondOrderAgg tests passed")
    
    test = TestDomainEmbedding()
    test.test_shape()
    print("✓ DomainEmbedding tests passed")
    
    print("\n✓ All CPU tests passed!")
    print("\nNote: GMDF_Detector tests require GPU with CLIP installed.")
    print("Run 'pytest tests/test_modules.py -v' for full test suite.")
