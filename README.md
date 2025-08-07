# AI Chibi Character Generator

This agent generates chibi-style 3D characters from photos using AI models for image processing, 3D model generation, and chibi art stylization.

## Features

- Photo-to-3D conversion using state-of-the-art AI models
- Chibi stylization of 3D models
- 3D post-processing for mesh refinement
- Export to common 3D formats (.GLB/.FBX/.OBJ)

## Requirements

- Python 3.8+
- PyTorch
- Blender (for 3D post-processing)
- Additional dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-chibi-generator.git
cd ai-chibi-generator

# Install dependencies
pip install -r requirements.txt
```
If the above doesn't work for you; simply download the zip file and then in your command terminal run the below
```bash
# Create a virtual environment (if you haven't already)
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Then install requirements
pip install -r requirements.txt
```


## Usage

```bash
python chibi_generator.py --input path/to/photo.jpg --output path/to/output/folder
```

## Pipeline Overview

1. **Input and Preprocessing**: Process user-uploaded photos
2. **3D Face/Body Reconstruction**: Convert photos to 3D models
3. **Chibi Stylization**: Apply chibi style to the 3D models
4. **Refinement & Texturing**: Refine mesh proportions and textures
5. **Export**: Generate 3D model files for download

## License

MIT
