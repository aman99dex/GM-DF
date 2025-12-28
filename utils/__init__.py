"""
GM-DF Device Utilities
Automatic GPU detection and optimization for NVIDIA CUDA, Apple MPS, and CPU fallback.
"""

import os
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about the selected compute device."""
    device: torch.device
    device_type: str  # "cuda", "mps", or "cpu"
    device_name: str
    memory_gb: Optional[float] = None
    supports_amp: bool = True  # Automatic Mixed Precision
    pin_memory: bool = True  # For DataLoader


def get_optimal_device(preferred: Optional[str] = None, verbose: bool = True) -> DeviceInfo:
    """
    Automatically detect and return the optimal compute device.
    
    Priority order:
    1. User-specified device (if valid and available)
    2. NVIDIA CUDA GPU
    3. Apple MPS (Metal Performance Shaders)
    4. CPU fallback
    
    Args:
        preferred: User-preferred device ("cuda", "mps", "cpu", or "auto")
        verbose: Print device information
    
    Returns:
        DeviceInfo with optimal device configuration
    """
    # Handle user preference
    if preferred and preferred != "auto":
        if preferred == "cuda":
            if torch.cuda.is_available():
                return _get_cuda_device(verbose)
            else:
                if verbose:
                    print("[!] CUDA requested but not available, falling back...")
        elif preferred == "mps":
            if torch.backends.mps.is_available():
                return _get_mps_device(verbose)
            else:
                if verbose:
                    print("[!] MPS requested but not available, falling back...")
        elif preferred == "cpu":
            return _get_cpu_device(verbose)
    
    # Auto-detection priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return _get_cuda_device(verbose)
    elif torch.backends.mps.is_available():
        return _get_mps_device(verbose)
    else:
        return _get_cpu_device(verbose)


def _get_cuda_device(verbose: bool = True) -> DeviceInfo:
    """Configure NVIDIA CUDA device."""
    device = torch.device("cuda")
    
    # Get device properties
    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    device_name = props.name
    memory_gb = props.total_memory / (1024 ** 3)
    
    # Enable TF32 for Ampere+ GPUs (better performance)
    if props.major >= 8:  # Ampere or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    if verbose:
        print(f"[*] Using NVIDIA CUDA GPU: {device_name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        if props.major >= 8:
            print(f"   TF32 enabled for faster training")
    
    return DeviceInfo(
        device=device,
        device_type="cuda",
        device_name=device_name,
        memory_gb=memory_gb,
        supports_amp=True,
        pin_memory=True,
    )


def _get_mps_device(verbose: bool = True) -> DeviceInfo:
    """Configure Apple MPS (Metal Performance Shaders) device."""
    device = torch.device("mps")
    
    # Get device name from system
    import platform
    chip = platform.processor()
    device_name = f"Apple {chip}" if chip else "Apple Silicon"
    
    if verbose:
        print(f"[*] Using Apple MPS (Metal): {device_name}")
        print(f"   Note: Some operations may fall back to CPU")
    
    return DeviceInfo(
        device=device,
        device_type="mps",
        device_name=device_name,
        memory_gb=None,  # Unified memory, not easily queryable
        supports_amp=False,  # MPS has limited AMP support
        pin_memory=False,  # pin_memory not supported with MPS
    )


def _get_cpu_device(verbose: bool = True) -> DeviceInfo:
    """Configure CPU device."""
    device = torch.device("cpu")
    
    # Get CPU info
    import platform
    device_name = platform.processor() or "CPU"
    
    # Set number of threads for optimal CPU performance
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    
    if verbose:
        print(f"[*] Using CPU: {device_name}")
        print(f"   Threads: {num_threads}")
        print(f"   [!] Training will be slow without GPU")
    
    return DeviceInfo(
        device=device,
        device_type="cpu",
        device_name=device_name,
        memory_gb=None,
        supports_amp=False,
        pin_memory=False,
    )


def get_dataloader_kwargs(device_info: DeviceInfo, num_workers: Optional[int] = None) -> dict:
    """
    Get optimal DataLoader kwargs based on device.
    
    Args:
        device_info: DeviceInfo from get_optimal_device()
        num_workers: Override number of workers (default: auto-detect)
    
    Returns:
        Dictionary of DataLoader kwargs
    """
    # Determine optimal number of workers
    if num_workers is None:
        num_cpus = os.cpu_count() or 4
        if device_info.device_type == "cuda":
            # Use more workers with CUDA for async data loading
            num_workers = min(num_cpus, 8)
        elif device_info.device_type == "mps":
            # MPS works well with fewer workers
            num_workers = min(num_cpus, 4)
        else:
            # CPU: fewer workers to avoid overhead
            num_workers = min(num_cpus, 2)
    
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device_info.pin_memory,
        "persistent_workers": num_workers > 0,
    }
    
    # Add prefetch factor for async loading (PyTorch 2.0+)
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    
    return kwargs


def to_device(data, device: torch.device):
    """
    Move data to device, handling different data types.
    
    Args:
        data: Tensor, list, tuple, or dict of tensors
        device: Target device
    
    Returns:
        Data moved to device with same structure
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(d, device) for d in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_amp_context(device_info: DeviceInfo, enabled: bool = True):
    """
    Get automatic mixed precision context manager.
    
    Args:
        device_info: DeviceInfo from get_optimal_device()
        enabled: Whether to enable AMP (if supported)
    
    Returns:
        Context manager for mixed precision
    """
    if enabled and device_info.supports_amp and device_info.device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        # No-op context manager
        return torch.amp.autocast(device_type="cpu", enabled=False)


def get_grad_scaler(device_info: DeviceInfo, enabled: bool = True):
    """
    Get gradient scaler for mixed precision training.
    
    Args:
        device_info: DeviceInfo from get_optimal_device()
        enabled: Whether to enable gradient scaling
    
    Returns:
        GradScaler instance (may be disabled)
    """
    if enabled and device_info.supports_amp and device_info.device_type == "cuda":
        return torch.amp.GradScaler("cuda")
    else:
        # Return a scaler that's effectively disabled
        return torch.amp.GradScaler("cuda", enabled=False)


def print_device_summary(device_info: DeviceInfo):
    """Print a summary of the compute device configuration."""
    print("\n" + "=" * 50)
    print("Device Configuration Summary")
    print("=" * 50)
    print(f"  Device Type: {device_info.device_type.upper()}")
    print(f"  Device Name: {device_info.device_name}")
    if device_info.memory_gb:
        print(f"  GPU Memory:  {device_info.memory_gb:.1f} GB")
    print(f"  AMP Support: {'Yes' if device_info.supports_amp else 'No'}")
    print(f"  Pin Memory:  {'Yes' if device_info.pin_memory else 'No'}")
    print("=" * 50 + "\n")


# Quick test
if __name__ == "__main__":
    print("Testing device detection...\n")
    device_info = get_optimal_device(verbose=True)
    print_device_summary(device_info)
    
    # Test tensor operations
    print("Testing tensor operations...")
    x = torch.randn(100, 100)
    x = to_device(x, device_info.device)
    y = x @ x.T
    print(f"[OK] Matrix multiplication on {device_info.device_type}: {y.shape}")
    
    # Test DataLoader kwargs
    kwargs = get_dataloader_kwargs(device_info)
    print(f"\nOptimal DataLoader kwargs: {kwargs}")
