"""
GPU Detection and Management Utilities
"""

import logging
import subprocess
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def detect_gpus() -> List[Dict]:
    """
    Detect available GPUs using nvidia-smi

    Returns:
        List of GPU info dicts with keys: name, memory_total, memory_free, index
    """
    gpus = []

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': int(parts[2]),
                            'memory_free_mb': int(parts[3]),
                        })

            logger.info(f"Detected {len(gpus)} GPU(s)")

    except FileNotFoundError:
        logger.debug("nvidia-smi not found - no NVIDIA GPUs available")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    return gpus


def has_gpu() -> bool:
    """Check if any GPU is available"""
    return len(detect_gpus()) > 0


def get_gpu_memory() -> Dict[str, int]:
    """
    Get total and free GPU memory

    Returns:
        Dict with 'total_mb' and 'free_mb' keys (sum across all GPUs)
    """
    gpus = detect_gpus()

    if not gpus:
        return {'total_mb': 0, 'free_mb': 0}

    total = sum(gpu['memory_total_mb'] for gpu in gpus)
    free = sum(gpu['memory_free_mb'] for gpu in gpus)

    return {'total_mb': total, 'free_mb': free}


def estimate_gpu_memory_needed(model_params_millions: float, batch_size: int = 1) -> int:
    """
    Estimate GPU memory needed for a model

    Args:
        model_params_millions: Number of model parameters in millions
        batch_size: Batch size for inference/training

    Returns:
        Estimated memory in MB
    """
    # Very rough estimation:
    # - 4 bytes per parameter (fp32)
    # - ~2x for gradients and optimizer states (training)
    # - ~1.5x for activations and workspace

    params_mb = (model_params_millions * 4)  # Convert to MB
    training_overhead = params_mb * 2  # Gradients + optimizer
    activation_overhead = params_mb * 0.5 * batch_size

    total_mb = params_mb + training_overhead + activation_overhead

    # Add 20% safety margin
    return int(total_mb * 1.2)


def can_fit_model(model_params_millions: float, batch_size: int = 1) -> bool:
    """
    Check if model can fit in available GPU memory

    Args:
        model_params_millions: Number of model parameters in millions
        batch_size: Batch size

    Returns:
        True if model likely fits, False otherwise
    """
    memory = get_gpu_memory()

    if memory['free_mb'] == 0:
        return False

    needed = estimate_gpu_memory_needed(model_params_millions, batch_size)

    return memory['free_mb'] >= needed


def recommend_batch_size(model_params_millions: float, max_batch_size: int = 32) -> int:
    """
    Recommend batch size based on available GPU memory

    Args:
        model_params_millions: Number of model parameters in millions
        max_batch_size: Maximum batch size to try

    Returns:
        Recommended batch size
    """
    memory = get_gpu_memory()

    if memory['free_mb'] == 0:
        return 1  # CPU fallback

    # Binary search for largest batch size that fits
    left, right = 1, max_batch_size
    best = 1

    while left <= right:
        mid = (left + right) // 2
        needed = estimate_gpu_memory_needed(model_params_millions, mid)

        if needed <= memory['free_mb']:
            best = mid
            left = mid + 1
        else:
            right = mid - 1

    return best


def get_cuda_version() -> Optional[str]:
    """Get CUDA version if available"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Parse CUDA version from nvidia-smi output
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)

    except Exception as e:
        logger.debug(f"Failed to get CUDA version: {e}")

    return None


def get_gpu_requirements_summary() -> Dict:
    """
    Get comprehensive GPU requirements summary

    Returns:
        Dict with GPU availability, memory, CUDA version, etc.
    """
    gpus = detect_gpus()
    memory = get_gpu_memory()
    cuda_version = get_cuda_version()

    return {
        'gpu_available': len(gpus) > 0,
        'gpu_count': len(gpus),
        'gpus': gpus,
        'total_memory_gb': round(memory['total_mb'] / 1024, 2),
        'free_memory_gb': round(memory['free_mb'] / 1024, 2),
        'cuda_version': cuda_version,
        'nvidia_smi_available': has_nvidia_smi(),
    }


def has_nvidia_smi() -> bool:
    """Check if nvidia-smi is available"""
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return True
    except:
        return False


def check_pytorch_cuda() -> bool:
    """Check if PyTorch can access CUDA"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_tensorflow_gpu() -> bool:
    """Check if TensorFlow can access GPU"""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        return False


def get_ml_framework_gpu_status() -> Dict:
    """
    Check GPU availability for common ML frameworks

    Returns:
        Dict with framework GPU status
    """
    return {
        'pytorch_cuda': check_pytorch_cuda(),
        'tensorflow_gpu': check_tensorflow_gpu(),
        'nvidia_gpus_detected': has_gpu(),
    }
