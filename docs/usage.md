# AI Chibi Character Generator Usage Guide

This guide explains how to use the AI Chibi Character Generator to create chibi-style 3D characters from photos.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Installing from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-chibi-generator.git
   cd ai-chibi-generator
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Basic Usage

### Command Line Interface

The AI Chibi Character Generator provides a simple command-line interface for generating chibi characters from photos.

```bash
python cli.py generate --input path/to/photo.jpg --output output_folder
```

### Options

- `--input`, `-i`: Path to the input photo (required)
- `--output`, `-o`: Output directory (default: "output")
- `--style`, `-s`: Stylization type (choices: "chibi", "anime", "cartoon", default: "chibi")
- `--format`, `-f`: Output 3D format (choices: "glb", "fbx", "obj", default: "glb")
- `--device`, `-d`: Device to run models on ("cuda" or "cpu", default: "cuda" if available, otherwise "cpu")
- `--verbose`, `-v`: Enable verbose output

### Example

```bash
python cli.py generate --input photos/portrait.jpg --output chibi_models --style chibi --format glb --verbose
```

## Python API

You can also use the AI Chibi Character Generator as a Python library in your own projects.

```python
from modules.preprocessing import preprocess_image
from modules.image_to_3d import generate_3d_model
from modules.chibi_stylization import apply_chibi_style
from modules.post_processing import refine_mesh, generate_textures
from modules.export import export_model

# Preprocess the input image
preprocessed_image = preprocess_image("path/to/photo.jpg")

# Generate 3D model from image
base_model = generate_3d_model(preprocessed_image)

# Apply chibi stylization
stylized_model = apply_chibi_style(base_model, style="chibi")

# Post-process the model
refined_model = refine_mesh(stylized_model)
textured_model = generate_textures(refined_model, style="chibi")

# Export the model
output_path = export_model(textured_model, "output_folder", format="glb")
print(f"Model exported to: {output_path}")
```

## Advanced Usage

### Customizing Stylization

You can customize the stylization by modifying the parameters in the `apply_chibi_style` function:

```python
stylized_model = apply_chibi_style(
    base_model,
    style="chibi",
    head_scale=1.5,  # Adjust head size (larger values = bigger head)
    body_scale=0.8,  # Adjust body size (smaller values = smaller body)
)
```

### Using Different Models

The AI Chibi Character Generator is designed to be modular, allowing you to use different models for different stages of the pipeline. You can replace the default models with your own by modifying the appropriate modules.

## Troubleshooting

### Common Issues

- **CUDA Out of Memory**: If you encounter CUDA out of memory errors, try reducing the image size or using a smaller model.
- **No Face Detected**: Make sure the input photo contains a clear, front-facing portrait.
- **Export Errors**: If you encounter errors during export, try using a different format (e.g., OBJ instead of GLB).

### Getting Help

If you encounter any issues or have questions, please open an issue on the GitHub repository.