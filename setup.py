#!/usr/bin/env python3
"""
Setup script for AI Wallpaper Generator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="ai-wallpaper",
    version="4.5.3",
    author="AI Wallpaper Team",
    description="Ultra-high-quality 4K wallpaper generation using AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-wallpaper",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'ai_wallpaper': [
            'config/*.yaml',
            'config/schemas/*.json'
        ]
    },
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0",
        "pyyaml>=6.0",
        "requests>=2.25",
        "Pillow>=9.0",
        "psutil>=5.8",
        "torch>=2.0",
        "torchvision>=0.15",
        "diffusers>=0.21",
        "transformers>=4.25",
        "accelerate>=0.20",
        "safetensors>=0.3",
        "omegaconf>=2.3",
        "einops>=0.6",
        "sentencepiece>=0.1",
        "openai>=1.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'ai-wallpaper=ai_wallpaper.cli.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Artistic Software",
    ],
)