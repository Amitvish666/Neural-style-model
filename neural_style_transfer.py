"""
Neural Style Transfer Implementation using PyTorch and VGG19.

This module implements the core neural style transfer algorithm based on
"A Neural Algorithm of Artistic Style" by Gatys et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from utils import tensor_to_pil, normalize_tensor
from config import Config

class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for Neural Style Transfer.
    Extracts features from specific layers used for content and style representation.
    """
    
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pre-trained VGG19 model
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        # Define feature extraction layers
        self.features = vgg
        
        # Layer indices for feature extraction
        # These correspond to conv layers in VGG19
        self.layer_names = {
            '0': 'conv1_1',   # Content representation
            '5': 'conv2_1',   # Style representation
            '10': 'conv3_1',  # Style representation
            '19': 'conv4_1',  # Style representation
            '21': 'conv4_2',  # Content representation (primary)
            '28': 'conv5_1'   # Style representation
        }
        
        # Default layers for content and style
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    def forward(self, x):
        """
        Forward pass through VGG19 to extract features.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing features from specified layers
        """
        features = {}
        
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features[self.layer_names[name]] = x
        
        return features

class GramMatrix(nn.Module):
    """
    Compute Gram matrix for style representation.
    The Gram matrix captures correlations between feature maps,
    representing the style of an image.
    """
    
    def forward(self, x):
        """
        Compute Gram matrix.
        
        Args:
            x: Feature tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Gram matrix of shape (batch_size, channels, channels)
        """
        batch_size, channels, height, width = x.size()
        
        # Reshape to (batch_size, channels, height*width)
        features = x.view(batch_size, channels, height * width)
        
        # Compute Gram matrix: features * features^T
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        gram = gram / (channels * height * width)
        
        return gram

class StyleLoss(nn.Module):
    """
    Style loss module that computes the loss between Gram matrices.
    """
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.target = self.gram(target_feature).detach()
    
    def forward(self, input_feature):
        """
        Compute style loss.
        
        Args:
            input_feature: Generated image features
            
        Returns:
            Style loss value
        """
        G = self.gram(input_feature)
        return F.mse_loss(G, self.target)

class ContentLoss(nn.Module):
    """
    Content loss module that computes the MSE between feature representations.
    """
    
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()
    
    def forward(self, input_feature):
        """
        Compute content loss.
        
        Args:
            input_feature: Generated image features
            
        Returns:
            Content loss value
        """
        return F.mse_loss(input_feature, self.target)

class NeuralStyleTransfer:
    """
    Main Neural Style Transfer class that orchestrates the style transfer process.
    """
    
    def __init__(self, content_weight=1.0, style_weight=1000000.0, device=None):
        """
        Initialize Neural Style Transfer.
        
        Args:
            content_weight: Weight for content loss (default: 1.0)
            style_weight: Weight for style loss (default: 1000000.0)
            device: Device to run on (cuda/cpu)
        """
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractor
        self.feature_extractor = VGGFeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # VGG normalization (ImageNet stats)
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
    
    def normalize_vgg(self, tensor):
        """
        Normalize tensor for VGG input.
        
        Args:
            tensor: Input tensor with values in [0, 1]
            
        Returns:
            Normalized tensor for VGG
        """
        # Ensure tensor is in [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Normalize using ImageNet stats
        mean = self.vgg_mean.view(1, 3, 1, 1)
        std = self.vgg_std.view(1, 3, 1, 1)
        
        return (tensor - mean) / std
    
    def create_loss_modules(self, content_features, style_features):
        """
        Create loss modules for content and style.
        
        Args:
            content_features: Features extracted from content image
            style_features: Features extracted from style image
            
        Returns:
            Tuple of (content_losses, style_losses)
        """
        content_losses = []
        style_losses = []
        
        # Content loss (typically conv4_2)
        for layer in self.feature_extractor.content_layers:
            if layer in content_features:
                content_loss = ContentLoss(content_features[layer])
                content_losses.append(content_loss)
        
        # Style losses (multiple layers)
        for layer in self.feature_extractor.style_layers:
            if layer in style_features:
                style_loss = StyleLoss(style_features[layer])
                style_losses.append(style_loss)
        
        return content_losses, style_losses
    
    def compute_total_loss(self, generated_features, content_losses, style_losses):
        """
        Compute total loss combining content and style losses.
        
        Args:
            generated_features: Features from generated image
            content_losses: List of content loss modules
            style_losses: List of style loss modules
            
        Returns:
            Tuple of (total_loss, content_loss_value, style_loss_value)
        """
        total_content_loss = 0
        total_style_loss = 0
        
        # Compute content loss
        content_idx = 0
        for layer in self.feature_extractor.content_layers:
            if layer in generated_features and content_idx < len(content_losses):
                total_content_loss += content_losses[content_idx](generated_features[layer])
                content_idx += 1
        
        # Compute style loss
        style_idx = 0
        for layer in self.feature_extractor.style_layers:
            if layer in generated_features and style_idx < len(style_losses):
                total_style_loss += style_losses[style_idx](generated_features[layer])
                style_idx += 1
        
        # Weighted combination
        total_loss = (self.content_weight * total_content_loss + 
                     self.style_weight * total_style_loss)
        
        content_loss_val = total_content_loss.item() if hasattr(total_content_loss, 'item') else float(total_content_loss)
        style_loss_val = total_style_loss.item() if hasattr(total_style_loss, 'item') else float(total_style_loss)
        
        return total_loss, content_loss_val, style_loss_val
    
    def transfer_style(self, content_image, style_image, num_steps=500, 
                      learning_rate=0.01, show_progress=True):
        """
        Perform neural style transfer.
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            show_progress: Whether to show progress bar and intermediate results
            
        Returns:
            Stylized image tensor
        """
        print("Extracting features from content and style images...")
        
        # Extract features
        with torch.no_grad():
            content_features = self.feature_extractor(self.normalize_vgg(content_image))
            style_features = self.feature_extractor(self.normalize_vgg(style_image))
        
        # Create loss modules
        content_losses, style_losses = self.create_loss_modules(
            content_features, style_features
        )
        
        # Initialize generated image (start with content image + noise)
        generated_image = content_image.clone()
        generated_image.requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.LBFGS([generated_image], lr=learning_rate)
        
        # Progress tracking
        if show_progress:
            pbar = tqdm(range(num_steps), desc="Style Transfer")
        
        step_count = [0]  # Use list to modify in closure
        
        def closure():
            """Optimizer closure function."""
            step_count[0] += 1
            
            # Clamp generated image to valid range
            generated_image.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            
            # Extract features from generated image
            generated_features = self.feature_extractor(
                self.normalize_vgg(generated_image)
            )
            
            # Compute losses
            total_loss, content_loss, style_loss = self.compute_total_loss(
                generated_features, content_losses, style_losses
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update progress
            if show_progress and step_count[0] % 10 == 0:
                pbar.set_postfix({
                    'Total': f'{total_loss.item():.2e}',
                    'Content': f'{content_loss:.2e}',
                    'Style': f'{style_loss:.2e}'
                })
            
            # Show intermediate results
            if show_progress and step_count[0] % 100 == 0:
                try:
                    import matplotlib.pyplot as plt
                    with torch.no_grad():
                        img = tensor_to_pil(generated_image.detach())
                        plt.figure(figsize=(8, 6))
                        plt.imshow(img)
                        plt.title(f'Step {step_count[0]}')
                        plt.axis('off')
                        plt.show()
                except:
                    pass  # Skip if matplotlib not available or in headless environment
            
            return total_loss
        
        # Optimization loop
        for i in range(num_steps):
            optimizer.step(closure)
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # Final clamping and detachment
        with torch.no_grad():
            generated_image.clamp_(0, 1)
        
        print("Style transfer completed!")
        return generated_image.detach()
    
    def get_loss_weights(self):
        """Get current loss weights."""
        return {
            'content_weight': self.content_weight,
            'style_weight': self.style_weight
        }
    
    def set_loss_weights(self, content_weight=None, style_weight=None):
        """
        Update loss weights.
        
        Args:
            content_weight: New content weight
            style_weight: New style weight
        """
        if content_weight is not None:
            self.content_weight = content_weight
        if style_weight is not None:
            self.style_weight = style_weight
