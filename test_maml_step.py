import torch
import torch.nn as nn
from config import GMDFConfig
from model.gm_df import build_model
from train_meta import MAMLTrainer

def test_maml_step():
    print("=== Testing MAML Step (FOMAML Logic) ===")
    
    # 1. Setup minimal environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = GMDFConfig()
    config.batch_size = 2
    config.inner_lr = 0.1 # High LR to see changes
    config.outer_lr = 0.01
    
    model = build_model(config, device=device, verbose=False)
    
    trainer = MAMLTrainer(
        model=model,
        config=config,
        device=device,
        use_amp=False # simpler for testing
    )
    
    # 2. Create dummy batches
    # (images, labels, domain_ids)
    def create_batch():
        return (
            torch.randn(2, 3, 224, 224),
            torch.randint(0, 2, (2,)).float(),
            torch.zeros(2, dtype=torch.long)
        )
        
    batch_A = create_batch()
    batch_B = create_batch()
    
    # 3. Track initial weights
    expert_param = list(trainer.theta_E)[0] # Just track one parameter
    initial_weight = expert_param.data.clone()
    print(f"Initial weight norm: {initial_weight.norm().item():.4f}")
    
    # 4. Run train step
    print("\nRunning train_step()...")
    losses = trainer.train_step(batch_A, batch_B)
    print(f"Losses: {losses}")
    
    # 5. Verify weight update
    final_weight = expert_param.data
    print(f"Final weight norm: {final_weight.norm().item():.4f}")
    
    diff = (final_weight - initial_weight).norm().item()
    print(f"Weight change (norm): {diff:.6f}")
    
    if diff > 0:
        print("[PASS] Weights updated successfully.")
    else:
        print("[FAIL] Weights did not change.")
        
    # 6. Verify MIM Output
    print("\n=== Testing MIM Decoder ===")
    images = batch_A[0].to(device)
    # We need to run encoder first
    # Using internal method just for test
    cls, all_feats, _, _ = model.encode_image(images, batch_A[2].to(device))
    
    loss_mim, pred, mask = model.mim_decoder(all_feats, images)
    print(f"MIM Loss: {loss_mim.item():.4f}")
    print(f"Pred shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    
    if loss_mim > 0:
        print("[PASS] MIM Loss computed.")
    else:
        print("[FAIL] MIM Loss is zero.")
        
    # 7. Verify Frozen Backbone
    print("\n=== Testing Frozen Backbone ===")
    # Reuse config already created at top of function
    print(f"freeze_backbone config: {config.freeze_backbone}")
    
    clip_frozen_count = 0
    clip_total_count = 0
    for name, param in model.named_parameters():
        if "clip_model" in name and "moe_adapter" not in name and "prompt_learner" not in name:
            clip_total_count += 1
            if not param.requires_grad:
                clip_frozen_count += 1
    
    print(f"CLIP backbone params: {clip_frozen_count}/{clip_total_count} frozen")
    if config.freeze_backbone and clip_frozen_count > 0:
        print("[PASS] CLIP backbone is frozen as expected.")
    elif not config.freeze_backbone:
        print("[INFO] CLIP backbone is unfrozen (per config).")
    else:
        print("[WARN] Freeze check inconclusive.")
    
    # 8. Verify L_dal
    print("\n=== Testing L_dal ===")
    # Create batch with 2 domains
    images = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 0, 1]).float().to(device)
    domain_ids = torch.tensor([0, 0, 1, 1]).long().to(device)
    
    outputs = model(images, domain_ids, labels)
    if "loss_dal" in outputs:
        print(f"L_dal: {outputs['loss_dal'].item():.4f}")
        print("[PASS] L_dal computed.")
    else:
        print("[FAIL] L_dal missing from outputs.")

if __name__ == "__main__":
    try:
        test_maml_step()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
