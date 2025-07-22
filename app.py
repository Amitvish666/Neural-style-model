"""
Web Interface for Neural Style Transfer using Gradio.

This module provides a user-friendly web interface for the Neural Style Transfer
application, allowing users to upload images and adjust parameters easily.
"""

import gradio as gr
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

from neural_style_transfer import NeuralStyleTransfer
from utils import (
    load_and_preprocess_image, 
    tensor_to_pil, 
    save_image, 
    visualize_comparison,
    create_image_grid
)
from config import Config, PRESETS

# Global variables
nst_model = None
device = None

def initialize_model():
    """Initialize the Neural Style Transfer model."""
    global nst_model, device
    device = Config.get_device(prefer_gpu=True)
    nst_model = NeuralStyleTransfer(device=device)
    print("Neural Style Transfer model initialized successfully!")

def style_transfer_interface(content_image, style_image, preset, custom_content_weight, 
                           custom_style_weight, custom_steps, image_size, learning_rate, 
                           show_intermediate):
    """
    Main function for the Gradio interface.
    
    Args:
        content_image: Uploaded content image
        style_image: Uploaded style image  
        preset: Selected preset configuration
        custom_content_weight: Custom content weight
        custom_style_weight: Custom style weight
        custom_steps: Custom number of steps
        image_size: Target image size
        learning_rate: Learning rate for optimization
        show_intermediate: Whether to show intermediate results
        
    Returns:
        Tuple of (stylized_image, comparison_image, info_text)
    """
    if content_image is None or style_image is None:
        return None, None, "Please upload both content and style images."
    
    try:
        # Initialize model if not already done
        if nst_model is None:
            initialize_model()
        
        # Ensure model is properly initialized
        if nst_model is None:
            return None, None, "Error: Could not initialize the Neural Style Transfer model."
        
        # Get parameters based on preset or custom values
        if preset != "custom":
            params = PRESETS[preset]
            content_weight = params['content_weight']
            style_weight = params['style_weight']
            steps = params['steps']
            info_text = f"Using preset: {preset} - {params['description']}\n"
        else:
            content_weight = custom_content_weight
            style_weight = custom_style_weight
            steps = custom_steps
            info_text = "Using custom parameters\n"
        
        # Validate parameters
        image_size = Config.validate_image_size(image_size)
        steps = Config.validate_steps(steps)
        content_weight = Config.validate_content_weight(content_weight)
        style_weight = Config.validate_style_weight(style_weight)
        
        # Update model weights
        nst_model.set_loss_weights(content_weight, style_weight)
        
        info_text += f"Parameters: Content Weight={content_weight}, Style Weight={style_weight}, Steps={steps}\n"
        info_text += f"Image Size={image_size}, Learning Rate={learning_rate}\n"
        info_text += f"Device: {device}\n\n"
        
        # Convert PIL images to tensors
        content_tensor = load_and_preprocess_image(
            content_image, target_size=image_size, device=device
        )
        style_tensor = load_and_preprocess_image(
            style_image, target_size=image_size, device=device
        )
        
        info_text += "Starting style transfer...\n"
        
        # Perform style transfer
        stylized_tensor = nst_model.transfer_style(
            content_image=content_tensor,
            style_image=style_tensor,
            num_steps=steps,
            learning_rate=learning_rate,
            show_progress=show_intermediate
        )
        
        # Convert result to PIL
        stylized_image = tensor_to_pil(stylized_tensor)
        
        # Create comparison image
        comparison_fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(content_image)
        axes[0].set_title('Content Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(style_image)
        axes[1].set_title('Style Image', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(stylized_image)
        axes[2].set_title('Stylized Result', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        comparison_image = Image.open(buf)
        plt.close()
        
        info_text += "Style transfer completed successfully!"
        
        return stylized_image, comparison_image, info_text
        
    except Exception as e:
        error_msg = f"Error during style transfer: {str(e)}"
        print(error_msg)
        return None, None, error_msg

def load_example_images():
    """Load example images if available."""
    example_content = None
    example_style = None
    
    # Try to load example images from examples directory
    content_path = "examples/content_mountain.svg"
    style_path = "examples/style_van_gogh.svg"
    
    if os.path.exists(content_path):
        try:
            example_content = content_path
        except:
            pass
    
    if os.path.exists(style_path):
        try:
            example_style = style_path
        except:
            pass
    
    return example_content, example_style

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Create the interface
    with gr.Blocks(
        title="Neural Style Transfer",
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .input-section {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🎨 Neural Style Transfer with VGG19
        
        Transform your images with the artistic style of famous paintings using deep learning!
        Upload a content image and a style image to create stunning artistic combinations.
        
        **Instructions:**
        1. Upload your content image (the photo you want to stylize)
        2. Upload your style image (the artwork whose style you want to apply)
        3. Choose a preset or customize parameters
        4. Click "Generate Stylized Image" and wait for the magic to happen!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input Images")
                
                content_image = gr.Image(
                    label="Content Image",
                    type="pil",
                    height=300,
                    show_download_button=True
                )
                
                style_image = gr.Image(
                    label="Style Image", 
                    type="pil",
                    height=300,
                    show_download_button=True
                )
                
                gr.Markdown("### ⚙️ Parameters")
                
                preset = gr.Dropdown(
                    choices=["balanced", "more_content", "more_style", "quick", "high_quality", "custom"],
                    value="balanced",
                    label="Preset Configuration",
                    info="Choose a preset or select 'custom' for manual control"
                )
                
                with gr.Group(visible=False) as custom_params:
                    custom_content_weight = gr.Slider(
                        minimum=0.1,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        label="Content Weight",
                        info="Higher values preserve more content details"
                    )
                    
                    custom_style_weight = gr.Slider(
                        minimum=1000,
                        maximum=10000000,
                        value=1000000,
                        step=10000,
                        label="Style Weight",
                        info="Higher values apply more style"
                    )
                    
                    custom_steps = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="Optimization Steps",
                        info="More steps = better quality but slower"
                    )
                
                image_size = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Image Size",
                    info="Larger sizes produce higher quality but take longer"
                )
                
                learning_rate = gr.Slider(
                    minimum=0.001,
                    maximum=0.1,
                    value=0.01,
                    step=0.001,
                    label="Learning Rate",
                    info="Controls optimization speed"
                )
                
                show_intermediate = gr.Checkbox(
                    label="Show Intermediate Results",
                    value=False,
                    info="Display progress during optimization (slower)"
                )
                
                generate_btn = gr.Button(
                    "🎨 Generate Stylized Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Results")
                
                stylized_output = gr.Image(
                    label="Stylized Result",
                    type="pil",
                    height=400,
                    show_download_button=True
                )
                
                comparison_output = gr.Image(
                    label="Comparison View",
                    type="pil", 
                    height=300,
                    show_download_button=True
                )
                
                info_output = gr.Textbox(
                    label="Process Information",
                    max_lines=10,
                    interactive=False
                )
        
        # Example section
        gr.Markdown("### 🎯 Examples")
        
        example_content, example_style = load_example_images()
        
        if example_content and example_style:
            gr.Examples(
                examples=[
                    [example_content, example_style, "balanced", 1.0, 1000000, 300, 512, 0.01, False]
                ],
                inputs=[content_image, style_image, preset, custom_content_weight, 
                       custom_style_weight, custom_steps, image_size, learning_rate, show_intermediate]
            )
        
        # Event handlers
        def toggle_custom_params(preset_value):
            return gr.Group(visible=(preset_value == "custom"))
        
        preset.change(
            fn=toggle_custom_params,
            inputs=preset,
            outputs=custom_params
        )
        
        generate_btn.click(
            fn=style_transfer_interface,
            inputs=[
                content_image, style_image, preset, custom_content_weight,
                custom_style_weight, custom_steps, image_size, learning_rate, show_intermediate
            ],
            outputs=[stylized_output, comparison_output, info_output]
        )
        
        # Footer information
        gr.Markdown("""
        ---
        ### 📚 About Neural Style Transfer
        
        This implementation is based on the paper "A Neural Algorithm of Artistic Style" by Gatys et al. (2015).
        It uses a pre-trained VGG19 network to extract content and style features, then optimizes a generated
        image to match the content of one image and the style of another.
        
        **Tips for best results:**
        - Use high-contrast style images with clear artistic patterns
        - Content images with clear subjects work better than busy scenes
        - Experiment with different content/style weight ratios
        - Higher steps generally produce better quality but take longer
        
        **Performance Notes:**
        - GPU acceleration is automatically used if available
        - Processing time varies from 1-10 minutes depending on parameters and hardware
        - Larger image sizes require more memory and processing time
        """)
    
    return demo

def main():
    """Main function to launch the web interface."""
    # Create necessary directories
    Config.create_directories()
    
    # Initialize model
    print("Initializing Neural Style Transfer model...")
    initialize_model()
    
    # Create and launch interface
    print("Creating web interface...")
    demo = create_interface()
    
    print(f"Launching web interface on {Config.WEB_HOST}:{Config.WEB_PORT}")
    demo.launch(
        server_name=Config.WEB_HOST,
        server_port=Config.WEB_PORT,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
