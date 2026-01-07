"""
Stable Diffusion Face Generator for GM-DF Training
Generates synthetic fake faces using Stable Diffusion for deepfake detection training.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Check for diffusers
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("[!] diffusers not installed. Run: pip install diffusers transformers accelerate")


# Face-related prompts for generating diverse fake faces
FACE_PROMPTS = [
    "professional headshot portrait of a person, studio lighting, high quality, 4k",
    "close-up portrait photograph of a human face, natural lighting, sharp focus",
    "realistic portrait photo of a person looking at camera, neutral background",
    "high resolution photograph of a person's face, professional photography",
    "portrait of a person with neutral expression, studio portrait lighting",
    "realistic human face photo, frontal view, professional headshot",
    "detailed portrait photograph, natural skin texture, high definition",
    "corporate headshot of a professional, clean background, sharp",
    "candid portrait photo of a face, soft lighting, 8k resolution",
    "photorealistic portrait, frontal face, studio quality lighting",
]

# Negative prompt to improve quality
NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, art, sketch, "
    "deformed, distorted, disfigured, bad anatomy, wrong anatomy, "
    "extra limbs, missing limbs, floating limbs, mutated hands, "
    "blurry, low quality, low resolution, watermark, text, logo, "
    "multiple faces, multiple people, group photo"
)


def generate_sd_faces(
    output_dir: str,
    num_images: int = 2000,
    batch_size: int = 4,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    seed: int = 42,
    use_fp16: bool = True,
):
    """
    Generate synthetic face images using Stable Diffusion.
    
    Args:
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        batch_size: Batch size for generation (lower if OOM)
        model_id: Hugging Face model ID
        seed: Random seed for reproducibility
        use_fp16: Use FP16 for lower memory usage
    """
    if not DIFFUSERS_AVAILABLE:
        print("Error: diffusers library not available")
        return
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Stable Diffusion Face Generator")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Target images: {num_images}")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[*] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[!] CUDA not available, using CPU (will be slow)")
        use_fp16 = False  # FP16 not supported on CPU
    
    # Load model
    print(f"[*] Loading Stable Diffusion model...")
    dtype = torch.float16 if use_fp16 else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # Disable for speed (we're generating faces, not NSFW)
        requires_safety_checker=False,
    )
    
    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[*] xFormers memory efficient attention enabled")
        except:
            print("[!] xFormers not available, using standard attention")
    
    print(f"[*] Model loaded successfully")
    
    # Generate images
    generator = torch.Generator(device=device).manual_seed(seed)
    
    num_batches = (num_images + batch_size - 1) // batch_size
    generated_count = 0
    
    print(f"\n[*] Generating {num_images} images in {num_batches} batches...")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        # Calculate how many images to generate in this batch
        remaining = num_images - generated_count
        current_batch_size = min(batch_size, remaining)
        
        if current_batch_size <= 0:
            break
        
        # Select prompts (cycle through)
        prompts = [FACE_PROMPTS[(batch_idx * batch_size + i) % len(FACE_PROMPTS)] 
                   for i in range(current_batch_size)]
        
        try:
            # Generate
            with torch.inference_mode():
                result = pipe(
                    prompt=prompts,
                    negative_prompt=[NEGATIVE_PROMPT] * current_batch_size,
                    num_inference_steps=25,  # Reduced for speed
                    guidance_scale=7.5,
                    generator=generator,
                    height=512,
                    width=512,
                )
            
            # Save images
            for i, image in enumerate(result.images):
                img_idx = generated_count + i
                # Resize to 224x224 for CLIP
                image_resized = image.resize((224, 224))
                save_path = output_path / f"sd_fake_{img_idx:05d}.jpg"
                image_resized.save(save_path, "JPEG", quality=95)
            
            generated_count += len(result.images)
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] OOM at batch {batch_idx}, reducing batch size...")
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            continue
        except Exception as e:
            print(f"\n[!] Error at batch {batch_idx}: {e}")
            continue
    
    print(f"\n[*] Generated {generated_count} images")
    print(f"[*] Saved to: {output_path}")
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    
    return generated_count


def setup_sd_domain(data_root: str, num_images: int = 2000):
    """
    Set up the StableDiffusion domain folder structure.
    
    Creates:
        data_root/StableDiffusion/
            ├── real/  (symlink or copy from another domain)
            └── fake/  (generated SD images)
    """
    sd_root = Path(data_root) / "StableDiffusion"
    fake_dir = sd_root / "fake"
    real_dir = sd_root / "real"
    
    # Create directories
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] SD Domain Setup")
    print(f"    Root: {sd_root}")
    print(f"    Fake: {fake_dir}")
    print(f"    Real: {real_dir}")
    
    # Check if we have real images from other domains
    ff_real = Path(data_root) / "FaceForensics" / "real"
    celeb_real = Path(data_root) / "Celeb-DF-v2" / "real"
    
    # Try to find existing real images
    source_real = None
    if ff_real.exists() and list(ff_real.glob("*"))[:1]:
        source_real = ff_real
    elif celeb_real.exists() and list(celeb_real.glob("*"))[:1]:
        source_real = celeb_real
    
    if source_real:
        print(f"[*] Will use real images from: {source_real}")
        # Create symlinks or copy a subset
        real_images = list(source_real.glob("*.jpg")) + list(source_real.glob("*.png"))
        # Take up to num_images/2 real images to balance
        real_subset = real_images[:num_images]
        
        print(f"[*] Linking {len(real_subset)} real images...")
        for i, src_img in enumerate(tqdm(real_subset, desc="Linking real")):
            dst_img = real_dir / f"real_{i:05d}{src_img.suffix}"
            if not dst_img.exists():
                try:
                    # Try symlink first (faster, saves space)
                    dst_img.symlink_to(src_img)
                except OSError:
                    # Fallback to copy on Windows if symlinks fail
                    import shutil
                    shutil.copy2(src_img, dst_img)
    else:
        print("[!] No existing real images found. You'll need to add them manually to:")
        print(f"    {real_dir}")
    
    return fake_dir, real_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion fake faces for GM-DF training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_root", type=str, default="datasets",
                        help="Root data directory")
    parser.add_argument("--num_images", type=int, default=2000,
                        help="Number of fake images to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip_setup", action="store_true",
                        help="Skip domain folder setup")
    
    args = parser.parse_args()
    
    # Setup domain structure
    if not args.skip_setup:
        fake_dir, real_dir = setup_sd_domain(args.data_root, args.num_images)
    else:
        fake_dir = Path(args.data_root) / "StableDiffusion" / "fake"
    
    # Generate fake images
    print("\n" + "="*60)
    print("Starting Stable Diffusion Face Generation")
    print("="*60)
    #chl gya
    count = generate_sd_faces(
        output_dir=str(fake_dir),
        num_images=args.num_images,
        batch_size=args.batch_size,
        model_id=args.model,
        seed=args.seed,
    )
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Generated: {count} fake images")
    print(f"Location: {fake_dir}")
    print("\nNext step: Run training with StableDiffusion domain")
    print("="*60)


if __name__ == "__main__":
    main()
