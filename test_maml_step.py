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

if __name__ == "__main__":
    try:
        test_maml_step()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
