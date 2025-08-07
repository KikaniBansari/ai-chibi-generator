#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-chibi-generator",
    version="0.1.0",
    author="AI Chibi Generator Team",
    author_email="example@example.com",
    description="AI agent that generates chibi-style 3D characters from photos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-chibi-generator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "open3d>=0.13.0",
        "trimesh>=3.9.0",
        "pyrender>=0.1.45",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "huggingface-hub>=0.8.0",
        "diffusers>=0.10.0",
        "transformers>=4.20.0",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "chibi-generator=cli:main",
        ],
    },
)