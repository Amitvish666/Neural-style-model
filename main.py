#!/usr/bin/env python3
"""
Neural Style Transfer - Command Line Interface
A PyTorch implementation of Neural Style Transfer using VGG19.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt

from neural_style_transfer import NeuralStyleTransfer
from utils import load_and_preprocess_image, save_image, tensor_to_pil
from config import Config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer using PyTorch and VGG19",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --content examples/content.jpg --style examples/style.jpg
  python main.py --content photo.jpg --style art.jpg --output result.jpg --steps 300
  python main.py --content photo.jpg --style art.jpg --content-weight 1.0 --style-weight 1000000
        """
    )
    
    parser.add_argument('--content', type=str, required=True,
                       help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                       help='Path to style image')
    parser.add_argument('--output', type=str, default='output_stylized.jpg',
                       help='Output image path (default: output_stylized.jpg)')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of optimization steps (default: 500)')
    parser.add_argument('--content-weight', type=float, default=1.0,
                       help='Weight for content loss (default: 1.0)')
    parser.add_argument('--style-weight', type=float, default=1000000.0,
                       help='Weight for style loss (default: 1000000.0)')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Size to resize images to (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for optimizer (default: 0.01)')
    parser.add_argument('--show-progress', action='store_true',
                       help='Show intermediate results every 100 steps')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       default='auto', help='Device to use (default: auto)')
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input arguments and files."""
    # Check if content image exists
    if not os.path.exists(args.content):
        print(f"Error: Content image '{args.content}' not found.")
        sys.exit(1)
    
    # Check if style image exists
    if not os.path.exists(args.style):
        print(f"Error: Style image '{args.style}' not found.")
        sys.exit(1)
    
    # Check if output directory exists, create if not
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Validate parameters
    if args.steps <= 0:
        print("Error: Number of steps must be positive.")
        sys.exit(1)
    
    if args.content_weight < 0 or args.style_weight < 0:
        print("Error: Content and style weights must be non-negative.")
        sys.exit(1)
    
    if args.image_size <= 0:
        print("Error: Image size must be positive.")
        sys.exit(1)

def main():
    """Main function to run Neural Style Transfer from command line."""
    args = parse_arguments()
    validate_inputs(args)
    
    # Print configuration
    print("="*50)
    print("Neural Style Transfer - PyTorch Implementation")
    print("="*50)
    print(f"Content image: {args.content}")
    print(f"Style image: {args.style}")
    print(f"Output image: {args.output}")
    print(f"Steps: {args.steps}")
    print(f"Content weight: {args.content_weight}")
    print(f"Style weight: {args.style_weight}")
    print(f"Image size: {args.image_size}")
    print(f"Learning rate: {args.lr}")
    print("="*50)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Load and preprocess images
        print("Loading and preprocessing images...")
        content_image = load_and_preprocess_image(
            args.content, target_size=args.image_size, device=device
        )
        style_image = load_and_preprocess_image(
            args.style, target_size=args.image_size, device=device
        )
        
        # Initialize Neural Style Transfer
        print("Initializing Neural Style Transfer model...")
        nst = NeuralStyleTransfer(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            device=device
        )
        
        # Run style transfer
        print("Starting style transfer optimization...")
        stylized_image = nst.transfer_style(
            content_image=content_image,
            style_image=style_image,
            num_steps=args.steps,
            learning_rate=args.lr,
            show_progress=args.show_progress
        )
        
        # Save result
        print(f"Saving result to {args.output}...")
        save_image(stylized_image, args.output)
        
        print("Style transfer completed successfully!")
        
        # Display final result if possible
        if args.show_progress:
            try:
                pil_image = tensor_to_pil(stylized_image)
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(tensor_to_pil(content_image))
                plt.title('Content Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(tensor_to_pil(style_image))
                plt.title('Style Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pil_image)
                plt.title('Stylized Result')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
                plt.show()
                print("Comparison saved as 'comparison.png'")
            except Exception as e:
                print(f"Could not display result: {e}")
                
    except Exception as e:
        print(f"Error during style transfer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
