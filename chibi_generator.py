#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Chibi Character Generator

This script processes photos to generate chibi-style 3D characters using AI models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

# Import local modules
from modules.preprocessing import preprocess_image
from modules.image_to_3d import generate_3d_model
from modules.chibi_stylization import apply_chibi_style
from modules.post_processing import refine_mesh, generate_textures
from modules.export import export_model

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate chibi-style 3D characters from photos')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input photo')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
    parser.add_argument('--style', '-s', type=str, default='chibi', 
                        choices=['chibi', 'anime', 'cartoon'], help='Stylization type')
    parser.add_argument('--format', '-f', type=str, default='glb', 
                        choices=['glb', 'fbx', 'obj'], help='Output 3D format')
    parser.add_argument('--device', '-d', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run models on (cuda/cpu)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main function to run the chibi character generation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    try:
        # Step 1: Preprocess the input image
        logger.info("Step 1: Preprocessing input image...")
        preprocessed_image = preprocess_image(input_path, device=args.device)
        
        # Step 2: Generate 3D model from image
        logger.info("Step 2: Generating 3D model from image...")
        base_model = generate_3d_model(preprocessed_image, device=args.device)
        
        # Step 3: Apply chibi stylization
        logger.info(f"Step 3: Applying {args.style} stylization...")
        stylized_model = apply_chibi_style(base_model, style=args.style, device=args.device)
        
        # Step 4: Post-process the model (refine mesh, generate textures)
        logger.info("Step 4: Post-processing model...")
        refined_model = refine_mesh(stylized_model)
        textured_model = generate_textures(refined_model, style=args.style)
        
        # Step 5: Export the model to the specified format
        logger.info(f"Step 5: Exporting model to {args.format} format...")
        output_path = export_model(textured_model, output_dir, format=args.format)
        
        logger.info(f"Successfully generated chibi character: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating chibi character: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())