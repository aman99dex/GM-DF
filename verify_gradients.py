
import torch
import torch.nn as nn
from model.gm_df import build_model
from config import GMDFConfig

def check_gradients():
    print("Building model...")
    config = GMDFConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config=config, device=device, verbose=False)
    
    # Create dummy data
    B = 4
    images = torch.randn(B, 3, 224, 224).to(device)
    domain_ids = torch.zeros(B, dtype=torch.long).to(device)
    labels = torch.randint(0, 2, (B,)).float().to(device) # Labels for loss
    
    print("Running forward pass...")
    outputs = model(images, domain_ids, labels=labels)
    loss_sis = outputs["loss_sis"]
    loss_total = outputs["loss_total"]
    
    print(f"Loss SIS: {loss_sis.item()}")
    print(f"Loss Total: {loss_total.item()}")
    
    print("Running backward pass...")
    loss_total.backward()
    
    # Check Gradients
    print("\n--- Gradient Verification ---")
    
    # 1. CLIP Backbone (Should be FROZEN -> None or 0)
    conv1_grad = model.clip_model.visual.conv1.weight.grad
    if conv1_grad is None:
        print("[OK] CLIP Backbone (Visual) is FROZEN (grad is None)")
    else:
        print(f"[FAIL] CLIP Backbone (Visual) HAS GRADIENTS: {conv1_grad.abs().sum()}")

    # 2. Text Transformer (Should be FROZEN -> None)
    # The transformer is in model.clip_model.transformer
    # Checking first layer
    resblock_grad = model.clip_model.transformer.resblocks[0].ln_1.weight.grad
    if resblock_grad is None:
        print("[OK] CLIP Backbone (Text) is FROZEN (grad is None)")
    else:
        print(f"[FAIL] CLIP Backbone (Text) HAS GRADIENTS: {resblock_grad.abs().sum()}")

    # 3. MoE Adapters (Should have gradients from Visual path)
    # model.vision_transformer.transformer.resblocks[0].mlp is the adapter
    adapter = model.vision_transformer.transformer.resblocks[0].mlp
    # It has experts
    expert_weight = adapter.experts[0].fc1.weight
    if expert_weight.grad is not None:
        grad_sum = expert_weight.grad.abs().sum().item()
        print(f"[OK] MoE Adapters have gradients (Gradient Flowing): {grad_sum:.6f}")
    else:
        print("[FAIL] MoE Adapters have NO gradients!")

    # 4. Prompt Learner (Should have gradients from Text path via loss_sis)
    # context_vectors
    prompt_grad = model.prompt_learner.context_vectors.grad
    if prompt_grad is not None:
        grad_sum = prompt_grad.abs().sum().item()
        print(f"[OK] Prompt Learner tokens have gradients (Cross-Learning Active): {grad_sum:.6f}")
    else:
        print("[FAIL] Prompt Learner tokens have NO gradients!")

    # 5. Domain Embedding (Used in Adapters and Prompts)
    # self.embeddings is an nn.Embedding layer, so we check .weight.grad
    dom_grad = model.domain_embedding.embeddings.weight.grad
    if dom_grad is not None:
        print(f"[OK] Domain Embeddings have gradients: {dom_grad.abs().sum().item():.6f}")
    else:
        print("[FAIL] Domain Embeddings have NO gradients!")

if __name__ == "__main__":
    check_gradients()
