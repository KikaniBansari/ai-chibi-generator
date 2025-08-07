#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for AI Chibi Character Generator.

This script tests the functionality of the AI chibi character generator.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.preprocessing import preprocess_image
from modules.image_to_3d import generate_3d_model
from modules.chibi_stylization import apply_chibi_style
from modules.post_processing import refine_mesh, generate_textures
from modules.export import export_model
from modules.utils import get_device


class TestChibiGenerator(unittest.TestCase):
    """
    Test cases for the AI chibi character generator.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Set up test data directory
        self.test_dir = Path(__file__).parent
        self.test_data_dir = self.test_dir / 'test_data'
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Set up test output directory
        self.test_output_dir = self.test_dir / 'test_output'
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Set device for testing
        self.device = get_device(prefer_cuda=False)  # Use CPU for testing
        
        # Create a dummy test image if it doesn't exist
        self.test_image_path = self.test_data_dir / 'test_image.jpg'
        if not self.test_image_path.exists():
            self._create_dummy_image(self.test_image_path)
    
    def _create_dummy_image(self, path):
        """
        Create a dummy test image.
        """
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple gradient image
            width, height = 512, 512
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a gradient
            for y in range(height):
                for x in range(width):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = int(255 * (x + y) / (width + height))
                    image[y, x] = [r, g, b]
            
            # Save the image
            Image.fromarray(image).save(path)
            
        except ImportError:
            # If PIL is not available, create an empty file
            with open(path, 'wb') as f:
                f.write(b'')
    
    def test_preprocessing(self):
        """
        Test the preprocessing module.
        """
        try:
            # Preprocess the test image
            preprocessed = preprocess_image(self.test_image_path, device=self.device)
            
            # Check that the preprocessing was successful
            self.assertIsNotNone(preprocessed)
            self.assertIn('image_tensor', preprocessed)
            self.assertIn('face_data', preprocessed)
            
        except Exception as e:
            self.fail(f"Preprocessing failed with error: {e}")
    
    def test_image_to_3d(self):
        """
        Test the image-to-3D module.
        """
        try:
            # Preprocess the test image
            preprocessed = preprocess_image(self.test_image_path, device=self.device)
            
            # Generate a 3D model
            model_data = generate_3d_model(preprocessed, device=self.device)
            
            # Check that the model generation was successful
            self.assertIsNotNone(model_data)
            self.assertIn('mesh', model_data)
            
        except Exception as e:
            self.fail(f"Image-to-3D conversion failed with error: {e}")
    
    def test_chibi_stylization(self):
        """
        Test the chibi stylization module.
        """
        try:
            # Preprocess the test image
            preprocessed = preprocess_image(self.test_image_path, device=self.device)
            
            # Generate a 3D model
            model_data = generate_3d_model(preprocessed, device=self.device)
            
            # Apply chibi stylization
            stylized_data = apply_chibi_style(model_data, device=self.device)
            
            # Check that the stylization was successful
            self.assertIsNotNone(stylized_data)
            self.assertIn('mesh', stylized_data)
            self.assertIn('style', stylized_data)
            
        except Exception as e:
            self.fail(f"Chibi stylization failed with error: {e}")
    
    def test_post_processing(self):
        """
        Test the post-processing module.
        """
        try:
            # Preprocess the test image
            preprocessed = preprocess_image(self.test_image_path, device=self.device)
            
            # Generate a 3D model
            model_data = generate_3d_model(preprocessed, device=self.device)
            
            # Apply chibi stylization
            stylized_data = apply_chibi_style(model_data, device=self.device)
            
            # Refine the mesh
            refined_data = refine_mesh(stylized_data)
            
            # Generate textures
            textured_data = generate_textures(refined_data)
            
            # Check that the post-processing was successful
            self.assertIsNotNone(refined_data)
            self.assertIsNotNone(textured_data)
            self.assertIn('mesh', textured_data)
            self.assertIn('texture_map', textured_data)
            
        except Exception as e:
            self.fail(f"Post-processing failed with error: {e}")
    
    def test_export(self):
        """
        Test the export module.
        """
        try:
            # Preprocess the test image
            preprocessed = preprocess_image(self.test_image_path, device=self.device)
            
            # Generate a 3D model
            model_data = generate_3d_model(preprocessed, device=self.device)
            
            # Apply chibi stylization
            stylized_data = apply_chibi_style(model_data, device=self.device)
            
            # Refine the mesh
            refined_data = refine_mesh(stylized_data)
            
            # Generate textures
            textured_data = generate_textures(refined_data)
            
            # Export the model
            output_path = export_model(textured_data, self.test_output_dir, format='obj')
            
            # Check that the export was successful
            self.assertTrue(output_path.exists())
            
        except Exception as e:
            self.fail(f"Export failed with error: {e}")
    
    def test_full_pipeline(self):
        """
        Test the full pipeline from image to 3D model.
        """
        try:
            # Preprocess the test image
            preprocessed = preprocess_image(self.test_image_path, device=self.device)
            
            # Generate a 3D model
            model_data = generate_3d_model(preprocessed, device=self.device)
            
            # Apply chibi stylization
            stylized_data = apply_chibi_style(model_data, device=self.device)
            
            # Refine the mesh
            refined_data = refine_mesh(stylized_data)
            
            # Generate textures
            textured_data = generate_textures(refined_data)
            
            # Export the model
            output_path = export_model(textured_data, self.test_output_dir, format='obj')
            
            # Check that the export was successful
            self.assertTrue(output_path.exists())
            
        except Exception as e:
            self.fail(f"Full pipeline failed with error: {e}")


if __name__ == '__main__':
    unittest.main()