"""
Utility functions for Neural Style Transfer.

This module contains helper functions for image loading, preprocessing,
postprocessing, and visualization.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_image(image_path):
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Cannot load image {image_path}: {e}")

def preprocess_image(image, target_size=512):
    """
    Preprocess PIL image for neural style transfer.
    
    Args:
        image: PIL Image object
        target_size: Target size for the longer dimension
        
    Returns:
        Preprocessed PIL Image
    """
    # Calculate new size maintaining aspect ratio
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int((height * target_size) / width)
    else:
        new_height = target_size
        new_width = int((width * target_size) / height)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def pil_to_tensor(image, device='cpu'):
    """
    Convert PIL image to PyTorch tensor.
    
    Args:
        image: PIL Image object
        device: Target device for tensor (str or torch.device)
        
    Returns:
        PyTorch tensor of shape (1, 3, height, width) with values in [0, 1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Convert device to string if it's a torch.device object
    if hasattr(device, 'type'):
        device_str = str(device)
    else:
        device_str = device
    return tensor.to(device_str)

def tensor_to_pil(tensor):
    """
    Convert PyTorch tensor to PIL image.
    
    Args:
        tensor: PyTorch tensor of shape (1, 3, height, width) or (3, height, width)
        
    Returns:
        PIL Image object
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Ensure tensor is on CPU
    tensor = tensor.cpu()
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    transform = transforms.ToPILImage()
    return transform(tensor)

def load_and_preprocess_image(image_path, target_size=512, device='cpu'):
    """
    Complete pipeline to load and preprocess an image.
    
    Args:
        image_path: Path to image file or PIL Image object
        target_size: Target size for preprocessing
        device: Target device for tensor (str or torch.device)
        
    Returns:
        Preprocessed image tensor
    """
    # Handle both file paths and PIL Images
    if isinstance(image_path, str):
        # Load image from path
        image = load_image(image_path)
    else:
        # Assume it's already a PIL Image
        image = image_path
    
    # Preprocess
    image = preprocess_image(image, target_size)
    
    # Convert to tensor
    tensor = pil_to_tensor(image, device)
    
    return tensor

def save_image(tensor, save_path, quality=95):
    """
    Save tensor as image file.
    
    Args:
        tensor: Image tensor
        save_path: Path to save the image
        quality: JPEG quality (if saving as JPEG)
    """
    # Convert to PIL
    pil_image = tensor_to_pil(tensor)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Save with appropriate quality settings
    if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
        pil_image.save(save_path, 'JPEG', quality=quality, optimize=True)
    else:
        pil_image.save(save_path)
    
    print(f"Image saved to: {save_path}")

def normalize_tensor(tensor):
    """
    Normalize tensor to [0, 1] range.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Normalized tensor
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    if tensor_max - tensor_min > 0:
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    return tensor

def visualize_comparison(content_image, style_image, stylized_image, 
                        save_path=None, figsize=(15, 5)):
    """
    Create a comparison visualization of content, style, and stylized images.
    
    Args:
        content_image: Content image tensor or PIL image
        style_image: Style image tensor or PIL image
        stylized_image: Stylized result tensor or PIL image
        save_path: Optional path to save the comparison
        figsize: Figure size for matplotlib
    """
    # Convert tensors to PIL if needed
    if torch.is_tensor(content_image):
        content_image = tensor_to_pil(content_image)
    if torch.is_tensor(style_image):
        style_image = tensor_to_pil(style_image)
    if torch.is_tensor(stylized_image):
        stylized_image = tensor_to_pil(stylized_image)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(content_image)
    axes[0].set_title('Content Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(style_image)
    axes[1].set_title('Style Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(stylized_image)
    axes[2].set_title('Stylized Result', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()

def create_image_grid(images, titles=None, cols=3, figsize=(12, 8), save_path=None):
    """
    Create a grid of images for visualization.
    
    Args:
        images: List of PIL images or tensors
        titles: Optional list of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size
        save_path: Optional path to save the grid
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, image in enumerate(images):
        row, col = i // cols, i % cols
        
        # Convert tensor to PIL if needed
        if torch.is_tensor(image):
            image = tensor_to_pil(image)
        
        axes[row, col].imshow(image)
        
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i], fontsize=12)
        
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image grid saved to: {save_path}")
    
    plt.show()

def get_image_info(image_path):
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    if not os.path.exists(image_path):
        return None
    
    try:
        with Image.open(image_path) as img:
            info = {
                'path': image_path,
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'file_size': os.path.getsize(image_path)
            }
        return info
    except Exception as e:
        print(f"Error reading image info: {e}")
        return None

def prepare_images_for_web(content_path, style_path, output_dir="temp"):
    """
    Prepare images for web interface by creating appropriately sized versions.
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_dir: Directory to save processed images
        
    Returns:
        Dictionary with paths to processed images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and resize images for preview
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    
    # Create preview versions (smaller for web display)
    content_preview = preprocess_image(content_img, target_size=256)
    style_preview = preprocess_image(style_img, target_size=256)
    
    # Save preview images
    content_preview_path = os.path.join(output_dir, "content_preview.jpg")
    style_preview_path = os.path.join(output_dir, "style_preview.jpg")
    
    content_preview.save(content_preview_path, quality=85)
    style_preview.save(style_preview_path, quality=85)
    
    return {
        'content_preview': content_preview_path,
        'style_preview': style_preview_path,
        'content_original': content_path,
        'style_original': style_path
    }
