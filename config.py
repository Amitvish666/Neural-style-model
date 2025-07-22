"""
Configuration settings for Neural Style Transfer.

This module contains default configurations, model parameters,
and other settings used throughout the application.
"""

import torch
from pathlib import Path

class Config:
    """Configuration class containing all default settings."""
    
    # Model Configuration
    MODEL_NAME = "vgg19"
    PRETRAINED = True
    
    # Image Processing
    DEFAULT_IMAGE_SIZE = 512
    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 256
    PREVIEW_SIZE = 256
    
    # Optimization Parameters
    DEFAULT_STEPS = 500
    MAX_STEPS = 2000
    MIN_STEPS = 50
    DEFAULT_LEARNING_RATE = 0.01
    
    # Loss Weights
    DEFAULT_CONTENT_WEIGHT = 1.0
    DEFAULT_STYLE_WEIGHT = 1000000.0
    MIN_CONTENT_WEIGHT = 0.0
    MAX_CONTENT_WEIGHT = 100.0
    MIN_STYLE_WEIGHT = 1000.0
    MAX_STYLE_WEIGHT = 10000000.0
    
    # VGG Feature Layers
    CONTENT_LAYERS = ['conv4_2']
    STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    # VGG Layer Mapping (layer index -> layer name)
    VGG_LAYER_MAPPING = {
        '0': 'conv1_1',   # First conv layer
        '5': 'conv2_1',   # Second block first conv
        '10': 'conv3_1',  # Third block first conv
        '19': 'conv4_1',  # Fourth block first conv
        '21': 'conv4_2',  # Fourth block second conv (main content layer)
        '28': 'conv5_1'   # Fifth block first conv
    }
    
    # ImageNet Normalization (for VGG)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # File Formats
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    OUTPUT_FORMAT = 'JPEG'
    OUTPUT_QUALITY = 95
    
    # Directories
    OUTPUT_DIR = "outputs"
    TEMP_DIR = "temp"
    EXAMPLES_DIR = "examples"
    
    # Web Interface Settings
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 7860
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Performance Settings
    BATCH_SIZE = 1  # NST typically processes one image at a time
    NUM_WORKERS = 0  # For data loading (0 for main thread)
    
    # Progress Display
    PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N steps
    INTERMEDIATE_DISPLAY_INTERVAL = 100  # Show intermediate results every N steps
    
    # Device Selection
    @staticmethod
    def get_device(prefer_gpu=True):
        """
        Get the best available device.
        
        Args:
            prefer_gpu: Whether to prefer GPU if available
            
        Returns:
            torch.device object
        """
        if prefer_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    @staticmethod
    def get_memory_info():
        """
        Get memory information for the current device.
        
        Returns:
            Dictionary with memory information
        """
        if torch.cuda.is_available():
            return {
                'device': 'cuda',
                'total_memory': torch.cuda.get_device_properties(0).total_memory,
                'allocated_memory': torch.cuda.memory_allocated(),
                'reserved_memory': torch.cuda.memory_reserved()
            }
        else:
            return {'device': 'cpu', 'note': 'CPU memory info not available'}
    
    @staticmethod
    def validate_image_size(size):
        """
        Validate image size is within acceptable bounds.
        
        Args:
            size: Target image size
            
        Returns:
            Validated size (clamped to bounds)
        """
        return max(Config.MIN_IMAGE_SIZE, min(size, Config.MAX_IMAGE_SIZE))
    
    @staticmethod
    def validate_steps(steps):
        """
        Validate number of optimization steps.
        
        Args:
            steps: Number of steps
            
        Returns:
            Validated steps (clamped to bounds)
        """
        return max(Config.MIN_STEPS, min(steps, Config.MAX_STEPS))
    
    @staticmethod
    def validate_content_weight(weight):
        """
        Validate content loss weight.
        
        Args:
            weight: Content weight
            
        Returns:
            Validated weight
        """
        return max(Config.MIN_CONTENT_WEIGHT, min(weight, Config.MAX_CONTENT_WEIGHT))
    
    @staticmethod
    def validate_style_weight(weight):
        """
        Validate style loss weight.
        
        Args:
            weight: Style weight
            
        Returns:
            Validated weight
        """
        return max(Config.MIN_STYLE_WEIGHT, min(weight, Config.MAX_STYLE_WEIGHT))
    
    @staticmethod
    def get_default_params():
        """
        Get default parameters for style transfer.
        
        Returns:
            Dictionary with default parameters
        """
        return {
            'image_size': Config.DEFAULT_IMAGE_SIZE,
            'steps': Config.DEFAULT_STEPS,
            'learning_rate': Config.DEFAULT_LEARNING_RATE,
            'content_weight': Config.DEFAULT_CONTENT_WEIGHT,
            'style_weight': Config.DEFAULT_STYLE_WEIGHT,
            'content_layers': Config.CONTENT_LAYERS,
            'style_layers': Config.STYLE_LAYERS
        }
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist."""
        directories = [Config.OUTPUT_DIR, Config.TEMP_DIR, Config.EXAMPLES_DIR]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        print(f"Directories created/verified: {directories}")

# Style Transfer Presets
PRESETS = {
    'balanced': {
        'content_weight': 1.0,
        'style_weight': 1000000.0,
        'steps': 500,
        'description': 'Balanced content and style preservation'
    },
    'more_content': {
        'content_weight': 5.0,
        'style_weight': 100000.0,
        'steps': 300,
        'description': 'Emphasizes content preservation'
    },
    'more_style': {
        'content_weight': 0.1,
        'style_weight': 10000000.0,
        'steps': 800,
        'description': 'Emphasizes style transfer'
    },
    'quick': {
        'content_weight': 1.0,
        'style_weight': 1000000.0,
        'steps': 200,
        'description': 'Fast processing with decent quality'
    },
    'high_quality': {
        'content_weight': 1.0,
        'style_weight': 1000000.0,
        'steps': 1000,
        'description': 'High quality output (slower processing)'
    }
}

# Example style descriptions for web interface
STYLE_DESCRIPTIONS = {
    'impressionist': 'Soft brushstrokes and light effects typical of Impressionist paintings',
    'abstract': 'Bold colors and geometric patterns for abstract artistic effects',
    'classical': 'Traditional painting techniques with realistic representation',
    'modern': 'Contemporary artistic styles with experimental techniques',
    'watercolor': 'Soft, flowing effects reminiscent of watercolor paintings',
    'oil_painting': 'Rich textures and bold strokes characteristic of oil paintings'
}
