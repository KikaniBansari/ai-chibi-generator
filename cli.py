#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for AI Chibi Character Generator.

This script provides a simple command-line interface for the AI chibi character generator.
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


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='AI Chibi Character Generator CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a chibi character from a photo')
    generate_parser.add_argument('--input', '-i', type=str, required=True, help='Path to input photo')
    generate_parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    generate_parser.add_argument('--style', '-s', type=str, default='chibi', 
                              choices=['chibi', 'anime', 'cartoon'], help='Stylization type')
    generate_parser.add_argument('--format', '-f', type=str, default='glb', 
                              choices=['glb', 'fbx', 'obj'], help='Output 3D format')
    generate_parser.add_argument('--device', '-d', type=str, 
                              default=get_device(), help='Device to run models on (cuda/cpu)')
    generate_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display information about the system')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Display version information')
    
    return parser.parse_args()


def display_info():
    """
    Display information about the system.
    """
    logger.info("AI Chibi Character Generator")
    logger.info("System Information:")
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
        logger.info(f"Current Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Display Python version
    logger.info(f"Python Version: {sys.version}")
    
    # Display PyTorch version
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
    except ImportError:
        logger.warning("PyTorch not installed")
    
    # Display other dependencies
    try:
        import numpy
        logger.info(f"NumPy Version: {numpy.__version__}")
    except ImportError:
        logger.warning("NumPy not installed")
    
    try:
        import PIL
        logger.info(f"PIL Version: {PIL.__version__}")
    except ImportError:
        logger.warning("PIL not installed")
    
    try:
        import trimesh
        logger.info(f"Trimesh Version: {trimesh.__version__}")
    except ImportError:
        logger.warning("Trimesh not installed")


def display_version():
    """
    Display version information.
    """
    logger.info("AI Chibi Character Generator v0.1.0")


def main():
    """
    Main function to run the CLI.
    """
    args = parse_args()
    
    if args.command == 'generate':
        # Set up arguments for the chibi generator
        sys.argv = [
            "chibi_generator.py",
            "--input", args.input,
            "--output", args.output,
            "--style", args.style,
            "--format", args.format,
            "--device", args.device
        ]
        
        if args.verbose:
            sys.argv.append("--verbose")
        
        # Run the chibi generator
        return chibi_main()
    
    elif args.command == 'info':
        display_info()
        return 0
    
    elif args.command == 'version':
        display_version()
        return 0
    
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())