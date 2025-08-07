#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export module for AI Chibi Character Generator.

This module handles exporting 3D models to various formats.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def export_model(model_data: Dict[str, Any], 
                output_dir: Union[str, Path], 
                format: str = 'glb') -> Path:
    """
    Export the 3D model to the specified format.
    
    Args:
        model_data: Dictionary containing the 3D model data
        output_dir: Output directory path
        format: Output format ('glb', 'fbx', 'obj')
        
    Returns:
        Path to the exported model file
    """
    logger.info(f"Exporting 3D model to {format.upper()} format...")
    
    # Extract mesh from model data
    mesh = model_data['mesh']
    style = model_data.get('style', 'chibi')
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_filename = f"chibi_character_{style}.{format.lower()}"
    output_path = output_dir / output_filename
    
    # Export based on format
    if format.lower() == 'glb':
        export_glb(mesh, output_path)
    elif format.lower() == 'fbx':
        export_fbx(mesh, output_path)
    elif format.lower() == 'obj':
        export_obj(mesh, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    logger.info(f"Model exported successfully to: {output_path}")
    
    # Generate a preview image
    preview_path = generate_preview(mesh, output_dir, style)
    
    return output_path


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """
    Export the mesh to GLB format.
    
    Args:
        mesh: 3D mesh to export
        output_path: Output file path
    """
    # Export to GLB format
    # GLB is a binary format that includes the mesh and textures
    mesh.export(str(output_path), file_type='glb')


def export_fbx(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """
    Export the mesh to FBX format.
    
    Args:
        mesh: 3D mesh to export
        output_path: Output file path
    """
    # Export to FBX format
    # Note: trimesh doesn't directly support FBX export
    # In a real implementation, you would use a library like PyMesh or Blender Python API
    
    # For now, we'll export to OBJ as a fallback
    logger.warning("FBX export not directly supported by trimesh. Exporting to OBJ instead.")
    mesh.export(str(output_path.with_suffix('.obj')), file_type='obj')
    
    # In a real implementation, you would use something like:
    # import bpy
    # bpy.ops.import_scene.obj(filepath=str(output_path.with_suffix('.obj')))
    # bpy.ops.export_scene.fbx(filepath=str(output_path))


def export_obj(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """
    Export the mesh to OBJ format.
    
    Args:
        mesh: 3D mesh to export
        output_path: Output file path
    """
    # Export to OBJ format
    mesh.export(str(output_path), file_type='obj')
    
    # If we have a texture, export the MTL and texture files as well
    if hasattr(mesh.visual, 'texture') and mesh.visual.texture is not None:
        # Export the MTL file
        mtl_path = output_path.with_suffix('.mtl')
        with open(mtl_path, 'w') as f:
            f.write(f"newmtl material0\n")
            f.write(f"Ka 1.000 1.000 1.000\n")  # Ambient color
            f.write(f"Kd 1.000 1.000 1.000\n")  # Diffuse color
            f.write(f"Ks 0.000 0.000 0.000\n")  # Specular color
            f.write(f"d 1.0\n")  # Transparency
            f.write(f"illum 2\n")  # Illumination model
            f.write(f"map_Kd {output_path.stem}.png\n")  # Texture map
        
        # Export the texture file
        texture_path = output_path.with_name(f"{output_path.stem}.png")
        mesh.visual.texture.save(str(texture_path))


def generate_preview(mesh: trimesh.Trimesh, output_dir: Path, style: str = 'chibi') -> Path:
    """
    Generate a preview image of the 3D model.
    
    Args:
        mesh: 3D mesh to render
        output_dir: Output directory path
        style: Style name for the filename
        
    Returns:
        Path to the preview image
    """
    logger.info("Generating preview image...")
    
    # Create a scene with the mesh
    scene = trimesh.Scene(mesh)
    
    # Set up camera and lighting
    # In a real implementation, you would set up proper camera and lighting
    
    # Render the scene
    # In a real implementation, you would use a renderer like pyrender
    # For now, we'll use trimesh's built-in renderer
    preview = scene.save_image(resolution=(1024, 1024), visible=True)
    
    # Save the preview image
    preview_path = output_dir / f"chibi_character_{style}_preview.png"
    with open(preview_path, 'wb') as f:
        f.write(preview)
    
    logger.info(f"Preview image generated: {preview_path}")
    
    return preview_path