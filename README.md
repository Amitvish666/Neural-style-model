# Neural Style Transfer with PyTorch

A comprehensive implementation of Neural Style Transfer using PyTorch and VGG19, featuring both command-line and web interfaces for creating artistic image stylizations.

## 🎨 Overview

Neural Style Transfer is a deep learning technique that combines the content of one image with the artistic style of another. This implementation uses a pre-trained VGG19 network to extract content and style features, then optimizes a generated image to match both representations.

## ✨ Features

- **PyTorch Implementation**: Built with modern PyTorch for optimal performance
- **VGG19 Feature Extraction**: Uses pre-trained VGG19 for robust feature representations
- **Web Interface**: User-friendly Gradio-based web interface
- **Command Line Tool**: Flexible CLI for batch processing and automation
- **Configurable Parameters**: Adjust content/style weights, optimization steps, and more
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Progress Monitoring**: Real-time progress tracking with intermediate results
- **Preset Configurations**: Pre-defined settings for different artistic effects

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install torch torchvision pillow numpy matplotlib gradio tqdm
