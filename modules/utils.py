#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for AI Chibi Character Generator.

This module provides helper functions for the AI chibi character generator.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
        logger.info(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.warning("CUDA is not available. Using CPU for processing.")
    
    return cuda_available


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get the device to use for processing.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from a file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded PIL Image
    """
    try:
        image = Image.open(image_path).convert('RGB')
        logger.debug(f"Loaded image from {image_path} with size {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {e}")
        raise


def save_image(image: Union[Image.Image, np.ndarray], output_path: Union[str, Path]) -> None:
    """
    Save an image to a file.
    
    Args:
        image: PIL Image or numpy array to save
        output_path: Path to save the image to
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Save the image
        image.save(output_path)
        logger.debug(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        raise


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Args:
        tensor: PyTorch tensor with shape (C, H, W) or (1, C, H, W)
        
    Returns:
        PIL Image
    """
    # Make sure the tensor is on CPU
    tensor = tensor.cpu()
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Rearrange from (C, H, W) to (H, W, C)
    tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy array
    array = tensor.numpy()
    
    # Scale to 0-255 if necessary
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(array)


def image_to_tensor(image: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.
    
    Args:
        image: PIL Image
        device: Device to put the tensor on
        
    Returns:
        PyTorch tensor with shape (1, C, H, W)
    """
    # Convert to numpy array
    array = np.array(image)
    
    # Rearrange from (H, W, C) to (C, H, W)
    array = array.transpose(2, 0, 1)
    
    # Scale to 0-1 if necessary
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(array).unsqueeze(0)
    
    # Move to device
    tensor = tensor.to(device)
    
    return tensor


def create_progress_bar(total: int, desc: str = 'Processing') -> Any:
    """
    Create a progress bar for tracking processing progress.
    
    Args:
        total: Total number of steps
        desc: Description for the progress bar
        
    Returns:
        Progress bar object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        logger.warning("tqdm not installed. Using simple progress logging.")
        return SimpleProgressBar(total, desc)


class SimpleProgressBar:
    """
    Simple progress bar for when tqdm is not available.
    """
    
    def __init__(self, total: int, desc: str = 'Processing'):
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_percent = -1
    
    def update(self, n: int = 1) -> None:
        self.current += n
        percent = int(self.current / self.total * 100)
        if percent > self.last_percent:
            logger.info(f"{self.desc}: {percent}% ({self.current}/{self.total})")
            self.last_percent = percent
    
    def close(self) -> None:
        logger.info(f"{self.desc}: 100% (Complete)")