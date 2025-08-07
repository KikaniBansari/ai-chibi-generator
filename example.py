#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for AI Chibi Character Generator.

This script demonstrates how to use the AI chibi character generator
with a sample image.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Import the chibi generator module
from chibi_generator import main as chibi_main
from modules.utils import check_cuda_availability, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Run the example script.
    """
    # Check CUDA availability
    check_cuda_availability()
    device = get_device()
    
    logger.info(f"Using device: {device}")
    
    # Create a sample image path
    # In a real scenario, you would use an actual image file
    sample_image_path = "sample_image.jpg"
    
    # Check if the sample image exists
    if not os.path.exists(sample_image_path):
        logger.warning(f"Sample image not found: {sample_image_path}")
        logger.info("Please provide a sample image or use your own image.")
        logger.info("Example usage: python chibi_generator.py --input your_image.jpg --output output_folder")
        return 1
    
    # Set up arguments for the chibi generator
    sys.argv = [
        "chibi_generator.py",
        "--input", sample_image_path,
        "--output", "example_output",
        "--style", "chibi",
        "--format", "glb",
        "--device", device,
        "--verbose"
    ]
    
    # Run the chibi generator
    logger.info("Running chibi generator with sample image...")
    return chibi_main()


if __name__ == "__main__":
    sys.exit(main())