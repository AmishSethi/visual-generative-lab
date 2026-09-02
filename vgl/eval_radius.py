#!/usr/bin/env python
"""
Enhanced evaluation script for continuous radius DiT model.
Automatically infers model parameters from checkpoint path and evaluates 
interpolation vs extrapolation performance with multiple metrics.
Now includes proper CFG (Classifier-Free Guidance) support.
"""
import os
import re
import torch
import numpy as np
from PIL import Image, ImageDraw
import argparse
import json
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import model components
from vgl.models import DiT_models_continuous as DiT_models
from vgl.unet_models import UNet_models
from vgl.unet_models_song import SongUNet_models
from vgl.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from vgl.flow_matching import FlowMatching

import os as _os

# Storage root for datasets, checkpoints and results.
# Override for your own machine:  export VGL_ROOT=/path/to/scratch
VGL_ROOT = _os.environ.get("VGL_ROOT", _os.path.expanduser("~/vgl-data"))


def parse_checkpoint_path(ckpt_path):
    """Parse model configuration from checkpoint path."""
    # Get the folder name from the path - handle both old and new folder structures
    path_parts = Path(ckpt_path).parts
    folder_name = None
    
    # First try to find 'circle_model' or 'songunet' (old structure)
    for part in path_parts:
        if 'circle_model' in part or 'songunet' in part.lower():
            folder_name = part
            break
    
    # If not found, look for ablation folder structure (new structure)
    if not folder_name:
        for part in path_parts:
            if any(x in part for x in ['radius_', 'position_', 'rotation_']):
                folder_name = part
                break
    
    if not folder_name:
        raise ValueError(f"Could not find folder name containing model info in path: {ckpt_path}")
    
    print(f"Parsing configuration from folder: {folder_name}")
    
    
    config = {
        'folder_name': folder_name
    }
    
    # Parse image size
    size_match = re.search(r'(\d+)x(\d+)', folder_name)
    if size_match:
        config['image_size'] = int(size_match.group(1))
    else:
        config['image_size'] = 64  # default
    
    # Parse radius embedding type (order matters to avoid substring collisions)
    name_lower = folder_name.lower()
    if 'single_linear' in name_lower or 'single-linear' in name_lower:
        config['radius_embedding_type'] = 'single_linear'
    elif 'raw' in name_lower:
        config['radius_embedding_type'] = 'raw'
    elif 'rotary' in name_lower:
        config['radius_embedding_type'] = 'rotary'
    elif 'sinusoidal' in name_lower:
        config['radius_embedding_type'] = 'sinusoidal'
    elif 'linear' in name_lower:
        config['radius_embedding_type'] = 'linear'
    else:
        config['radius_embedding_type'] = 'linear'  # default
    
    # Parse conditioning method
    if 'adaln' in name_lower:
        config['conditioning_method'] = 'adaln'
    elif 'concat' in name_lower:
        config['conditioning_method'] = 'concat'
    else:
        config['conditioning_method'] = 'concat'  # default
    
    # Parse CFG settings (respect explicit no_cfg)
    if 'no_cfg' in name_lower or 'nocfg' in name_lower:
        config['radius_dropout_prob'] = 0.0
        config['use_cfg'] = False
        config['null_embedding_type'] = 'none'
    elif 'cfg' in name_lower:
        config['radius_dropout_prob'] = 0.1
        config['use_cfg'] = True  # Flag to indicate CFG was used in training
        
        # UPDATED: Check for null embedding type including "none"
        if 'none' in name_lower or 'no_emb' in name_lower or 'noemb' in name_lower:
            config['null_embedding_type'] = 'none'
        elif 'zero' in name_lower:
            config['null_embedding_type'] = 'zero'
        elif 'learnable' in name_lower or 'learn' in name_lower:
            config['null_embedding_type'] = 'learnable'
        else:
            # Default for CFG - could be any of the three, try to be more specific
            # Look for more specific keywords
            if 'cfg_none' in name_lower:
                config['null_embedding_type'] = 'none'
            elif 'cfg_zero' in name_lower:
                config['null_embedding_type'] = 'zero'
            else:
                config['null_embedding_type'] = 'learnable'  # default for CFG
    else:
        config['radius_dropout_prob'] = 0.0
        config['use_cfg'] = False
        config['null_embedding_type'] = 'none'
    
    # Parse training radius range
    range_match = re.search(r'(\d+)to(\d+)', folder_name)
    if range_match:
        config['train_min_radius'] = int(range_match.group(1))
        config['train_max_radius'] = int(range_match.group(2))
    else:
        # Try alternative patterns
        range_match = re.search(r'_(\d+)_(\d+)_', folder_name)
        if range_match:
            config['train_min_radius'] = int(range_match.group(1))
            config['train_max_radius'] = int(range_match.group(2))
        else:
            # For new ablation folder structure, use default radius range
            # Based on the original training data: circle_dataset_16_radii_5to20_64x64_NEW
            print(f"Warning: Could not parse radius range from folder name: {folder_name}")
            print("Using default radius range for ablation experiments: 5 to 20")
            config['train_min_radius'] = 5
            config['train_max_radius'] = 20
    
    # NEW: Parse bias information
    bias_match = re.search(r'bias(\d+)', folder_name)
    if bias_match:
        config['bias'] = int(bias_match.group(1))
        print(f"Detected bias: {config['bias']}")
        # Adjust training range to actual pixel values
        config['actual_min_radius'] = config['train_min_radius'] - config['bias']
        config['actual_max_radius'] = config['train_max_radius'] - config['bias']
        print(f"Actual training radii: {config['actual_min_radius']} to {config['actual_max_radius']} pixels")
    else:
        config['bias'] = 0
        config['actual_min_radius'] = config['train_min_radius']
        config['actual_max_radius'] = config['train_max_radius']
    
    # Set other defaults
    config['use_latent_diffusion'] = False
    config['null_radius'] = 0.0
    
    # Parse flow matching
    if 'flow' in name_lower:
        config['use_flow_matching'] = True
        print(f"Detected flow matching model")
    else:
        config['use_flow_matching'] = False
    
    # Parse architecture (DiT, UNet, or SongUNet)
    if 'songunet' in name_lower:
        config['architecture'] = 'songunet'
        config['model_name'] = 'SongUNet-S'  # Correct model key for SongUNet
        # NEW SongUNet models use sinusoidal embedding and adaln conditioning
        # Override generic parsing unless explicitly specified in folder name
        if 'linear' not in name_lower and 'sinusoidal' not in name_lower and 'rotary' not in name_lower:
            config['radius_embedding_type'] = 'sinusoidal'  # NEW models use sinusoidal
        if 'adaln' not in name_lower and 'concat' not in name_lower:
            # No explicit conditioning in folder name, use adaln for NEW models
            config['conditioning_method'] = 'adaln'  # NEW models use adaln
    elif 'unet' in name_lower:
        config['architecture'] = 'unet'
    else:
        config['architecture'] = 'dit'
    
    # Parse model name/size from path components, e.g., "000-DiT-B-2" or "DiT-B2" or "DiT-S/2" or "UNet-S"
    model_name = None
    for part in reversed(path_parts):
        # Check for SongUNet models
        m_songunet = re.search(r'SongUNet-([A-Za-z]+)', part)
        if m_songunet:
            base = m_songunet.group(1).upper()
            model_name = f"SongUNet-{base}"
            break
        # Check for UNet models
        m_unet = re.search(r'UNet-([A-Za-z]+)', part)
        if m_unet:
            base = m_unet.group(1).upper()
            model_name = f"UNet-{base}"
            break
        # Check for DiT models
        m = re.search(r'DiT-([A-Za-z]+)[-/](\d+)', part)
        if m:
            base = m.group(1).upper()
            ds = m.group(2)
            model_name = f"DiT-{base}/{ds}"
            break
        m2 = re.search(r'DiT-([A-Za-z]+)(\d+)', part)
        if m2:
            base = m2.group(1).upper()
            ds = m2.group(2)
            model_name = f"DiT-{base}/{ds}"
            break
    if model_name is None:
        full_path_str = str(Path(ckpt_path))
        # Check for SongUNet in full path
        m_songunet = re.search(r'SongUNet-([A-Za-z]+)', full_path_str)
        if m_songunet:
            base = m_songunet.group(1).upper()
            model_name = f"SongUNet-{base}"
        # Check for UNet in full path
        elif m_unet := re.search(r'UNet-([A-Za-z]+)', full_path_str):
            base = m_unet.group(1).upper()
            model_name = f"UNet-{base}"
        else:
            m = re.search(r'DiT-([A-Za-z]+)[-/](\d+)', full_path_str)
            if m:
                base = m.group(1).upper()
                ds = m.group(2)
                model_name = f"DiT-{base}/{ds}"
    if model_name is None:
        # Default based on architecture
        if config['architecture'] == 'songunet':
            model_name = "SongUNet-S"
        elif config['architecture'] == 'unet':
            model_name = "UNet-S"
        else:
            model_name = "DiT-S/2"
    config['model_name'] = model_name
    
    return config


def get_evaluation_radii(train_min, train_max, bias=0, interp_mode='odd'):
    """Get exact training, interpolation and extrapolation radius values, accounting for bias."""
    # If there's a bias, we need to test actual radii 1-4 for lower extrapolation
    if bias != 0:
        print(f"Using biased evaluation: labels {train_min} to {train_max} (actual radii {train_min-bias} to {train_max-bias} pixels)")
        
        # For biased models, we want to test specific ACTUAL radii and map them to the correct labels
        
        # Exact training radii (subset of actual training values)
        actual_train_min = train_min - bias
        actual_train_max = train_max - bias
        training_actual_radii = [float(r) for r in range(actual_train_min, actual_train_max + 1, 2)]  # Every 2nd radius
        training_labels = [r + bias for r in training_actual_radii]
        
        # Interpolation set inside the training range
        if interp_mode == 'midpoints':
            # Half-steps between integers, e.g., 5.5, 6.5, ..., (max-0.5)
            interpolation_actual_radii = [r + 0.5 for r in range(actual_train_min, actual_train_max)]
        else:
            # Default to midpoints to avoid overlap with training samples (which use every 2nd radius)
            interpolation_actual_radii = [r + 0.5 for r in range(actual_train_min, actual_train_max)]
        
        # Convert actual radii to labels for generation
        interpolation_labels = [r + bias for r in interpolation_actual_radii]
        
        # Extrapolation: Test actual radii 1-4 for lower, and higher values for upper
        extrapolation_actual_radii = []
        
        # Lower extrapolation: Test actual radii 1, 2, 3, 4
        for actual_radius in range(1, 5):
            extrapolation_actual_radii.append(float(actual_radius))
        
        # Upper extrapolation: Test actual radii beyond training range
        for i in range(1, 5):
            actual_radius = actual_train_max + i
            extrapolation_actual_radii.append(float(actual_radius))
        
        # Convert actual radii to labels for generation
        extrapolation_labels = [r + bias for r in extrapolation_actual_radii]
        
        print(f"Exact training: actual radii {training_actual_radii} -> generation labels {training_labels}")
        print(f"Interpolation: actual radii {interpolation_actual_radii} -> generation labels {interpolation_labels}")
        print(f"Lower extrapolation: actual radii {extrapolation_actual_radii[:4]} -> generation labels {extrapolation_labels[:4]}")
        print(f"Upper extrapolation: actual radii {extrapolation_actual_radii[4:]} -> generation labels {extrapolation_labels[4:]}")
        
        return training_labels, interpolation_labels, extrapolation_labels, training_actual_radii, interpolation_actual_radii, extrapolation_actual_radii
    else:
        # No bias - standard evaluation
        
        # Exact training radii (subset of training values)
        training_radii = [float(r) for r in range(train_min, train_max + 1, 2)]  # Every 2nd radius
        
        # Interpolation set inside the training range
        if interp_mode == 'midpoints':
            interpolation_radii = [r + 0.5 for r in range(train_min, train_max)]
        else:
            # Default to midpoints to avoid overlap with training samples (which use every 2nd radius)
            interpolation_radii = [r + 0.5 for r in range(train_min, train_max)]
        
        # Extrapolation: 4 values below (down to 1) and 4 values above
        extrapolation_radii = []
        
        # Below range
        for i in range(1, 5):
            r = train_min - i
            if r >= 1:
                extrapolation_radii.append(float(r))
        extrapolation_radii.sort()  # Sort in ascending order
        
        # Above range - extend to include 25 and 30
        for i in range(1, 5):
            r = train_max + i
            extrapolation_radii.append(float(r))
        # Add radius 25 and 30 for additional extrapolation examples
        extrapolation_radii.append(25.0)
        extrapolation_radii.append(30.0)
        
        return training_radii, interpolation_radii, extrapolation_radii, training_radii, interpolation_radii, extrapolation_radii


def generate_ground_truth_circle(radius, image_size=64, circle_color=(255, 128, 100), 
                                background_color=(255, 255, 255), antialiasing=4):
    """Generate a ground truth circle image (same as training data)."""
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_radius = radius * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    center = large_size // 2
    left = center - large_radius
    top = center - large_radius
    right = center + large_radius
    bottom = center + large_radius
    
    draw.ellipse([left, top, right, bottom], fill=circle_color, outline=circle_color)
    
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def extract_circle_mask(image, threshold_method='otsu'):
    """Extract binary mask of the circle from an image."""
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply thresholding to get binary mask
    if threshold_method == 'otsu':
        # Otsu's thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if needed (ensure circle is white)
        if np.mean(mask) > 127:  # If mostly white, invert
            mask = 255 - mask
    else:
        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Convert to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def calculate_circle_metrics(generated_img, ground_truth_img, gt_radius):
    """
    Calculate various metrics between generated and ground truth circles.
    
    Returns:
        dict with IoU, precision, recall, estimated radius, and radius errors
    """
    # Extract masks
    gen_mask = extract_circle_mask(generated_img)
    gt_mask = extract_circle_mask(ground_truth_img)
    
    # Calculate IoU
    iou = calculate_iou(gen_mask, gt_mask)
    
    # Calculate precision and recall
    intersection = np.logical_and(gen_mask, gt_mask).sum()
    gen_pixels = gen_mask.sum()
    gt_pixels = gt_mask.sum()
    
    precision = intersection / gen_pixels if gen_pixels > 0 else 0
    recall = intersection / gt_pixels if gt_pixels > 0 else 0
    
    # Estimate radius from generated mask
    if gen_pixels > 0:
        # Approximate radius assuming circle
        estimated_area = gen_pixels
        estimated_radius = np.sqrt(estimated_area / np.pi)
    else:
        estimated_radius = 0
    
    # Calculate radius errors
    radius_error_raw = estimated_radius - gt_radius
    radius_error_percentage = (radius_error_raw / gt_radius * 100) if gt_radius > 0 else 0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'estimated_radius': estimated_radius,
        'gt_radius': gt_radius,
        'radius_error_raw': radius_error_raw,
        'radius_error_percentage': radius_error_percentage,
        'gen_mask': gen_mask,
        'gt_mask': gt_mask
    }


def load_model(config, device):
    """Load the trained DiT model."""
    # Determine input size and channels
    input_size = config['image_size']
    in_channels = 3  # Default
    out_channels = 3  # Default for SongUNet
    
    # Load checkpoint first to check model configuration
    print(f"Loading checkpoint...")
    checkpoint = torch.load(config['ckpt_path'], map_location=device, weights_only=False)
    
    # Get the state dict to check
    state_dict = None
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if this is a VAE model (latent diffusion) - only for DiT/UNet, not SongUNet
    # Be careful not to match "novae" or "no_vae" (no VAE) - only match "vae" as a standalone word
    folder_lower = config['folder_name'].lower()
    is_vae_model = ('_vae_' in folder_lower or 
                   folder_lower.startswith('vae_') or 
                   folder_lower.endswith('_vae') or 
                   folder_lower == 'vae') and 'novae' not in folder_lower and 'no_vae' not in folder_lower
    # run_config records this explicitly and is authoritative; the folder-name
    # heuristic above is a fallback for older checkpoints. Path-only detection
    # silently builds a pixel model for latent checkpoints in dirs like hires128_vae/.
    if config.get('use_latent_diffusion'):
        is_vae_model = True

    if is_vae_model and config.get('architecture') != 'songunet':
        config['use_latent_diffusion'] = True
        print(f"VAE model detected - using latent diffusion")
        # For latent diffusion, we work in 8x8 latent space for 64x64 images
        input_size = input_size // 8
        print(f"Adjusted input size for latent diffusion: {input_size}x{input_size}")
        
        # VAE models use 4 input channels (latent space) and different output dimensions
        # Check the actual checkpoint to get the correct dimensions
        for key in state_dict.keys():
            if 'x_embedder.proj.weight' in key:
                checkpoint_in_channels = state_dict[key].shape[1]
                print(f"VAE checkpoint has {checkpoint_in_channels} input channels")
                in_channels = checkpoint_in_channels
                break
        
        # Check output dimensions from final layer
        if 'final_layer.linear.weight' in state_dict:
            checkpoint_out_features = state_dict['final_layer.linear.weight'].shape[0]
            print(f"VAE checkpoint expects {checkpoint_out_features} output features")
            # This affects the patch size calculation for the final layer
    
    # For SongUNet, detect channels from checkpoint for backwards compatibility
    elif config.get('architecture') == 'songunet':
        # Default to 3-channel RGB, but adapt based on checkpoint
        in_channels = 3
        out_channels = 3
        config['use_latent_diffusion'] = False
        
        # For SongUNet, check the checkpoint to determine the actual channels used
        # Look for the first encoder conv layer to determine channels
        found_channels = False
        for key in state_dict.keys():
            if 'enc.' in key and 'conv.weight' in key:
                checkpoint_channels = state_dict[key].shape[1]
                print(f"SongUNet checkpoint has {checkpoint_channels} input channels (from {key})")
                found_channels = True
                
                if checkpoint_channels == 3:
                    print("Using standard 3-channel RGB architecture")
                    config['use_latent_diffusion'] = False
                    in_channels = 3
                    out_channels = 3
                elif checkpoint_channels > 3:
                    if is_vae_model:
                        print(f"Using {checkpoint_channels}-channel VAE latent architecture")
                        config['use_latent_diffusion'] = True
                        input_size = input_size // 8
                        print(f"Adjusted input size for latent diffusion: {input_size}x{input_size}")
                        in_channels = checkpoint_channels
                        out_channels = checkpoint_channels
                    else:
                        # Extra channels likely come from concat conditioning, not VAE
                        print(f"Using {checkpoint_channels}-channel concat-conditioned architecture")
                        config['use_latent_diffusion'] = False
                        if config.get('conditioning_method') != 'concat':
                            print("Detected extra input channels; overriding conditioning_method to 'concat'")
                        config['conditioning_method'] = 'concat'
                        # For concat, checkpoint channels include +1 conditioning channel
                        base_channels = checkpoint_channels - 1
                        in_channels = max(1, base_channels)
                        out_channels = in_channels
                break
        
        if not found_channels:
            print(f"SongUNet detected - using default 3-channel RGB architecture")
        
        # Store channels in config for later use
        config['detected_in_channels'] = in_channels
        config['detected_out_channels'] = out_channels
    else:
        # For DiT/UNet, keep existing logic but also check for channel detection
        for key in state_dict.keys():
            if 'x_embedder.proj.weight' in key:  # DiT patch embedding
                detected_channels = state_dict[key].shape[1]
                print(f"Detected {detected_channels} input channels from DiT patch embedding")
                in_channels = detected_channels
                break
            elif 'input_conv.weight' in key:  # UNet input conv
                detected_channels = state_dict[key].shape[1]
                print(f"Detected {detected_channels} input channels from UNet input conv")
                in_channels = detected_channels
                break
    
    # Check positional embedding size to infer conditioning method (for DiT models only)
    if 'pos_embed' in state_dict and config.get('architecture') == 'dit':
        pos_embed_shape = state_dict['pos_embed'].shape
        model_name = config.get('model_name', 'DiT-S/2')
        
        # Determine number of patches based on model name and input size.
        # Parse the patch size off the end of the name rather than substring-matching:
        # 'S/1' is a substring of 'S/16', which silently gave patch-16 models a
        # patch-1 patch count.
        try:
            patch_size = int(model_name.rsplit('/', 1)[1])
        except (IndexError, ValueError):
            patch_size = 2
        num_patches = (input_size // patch_size) ** 2
        
        expected_adaln_positions = num_patches
        expected_concat_positions = num_patches + 2  # +2 for time and radius tokens
        
        actual_positions = pos_embed_shape[1]
        
        if actual_positions == expected_concat_positions:
            print(f"Detected concat conditioning from pos_embed shape: {pos_embed_shape}")
            print(f"Expected {expected_concat_positions} positions (patches={num_patches} + 2 tokens), got {actual_positions}")
            config['conditioning_method'] = 'concat'
        elif actual_positions == expected_adaln_positions:
            print(f"Detected adaln conditioning from pos_embed shape: {pos_embed_shape}")
            print(f"Expected {expected_adaln_positions} positions (patches={num_patches}), got {actual_positions}")
            config['conditioning_method'] = 'adaln'
        else:
            print(f"Warning: Unexpected pos_embed shape {pos_embed_shape}. Expected {expected_adaln_positions} (adaln) or {expected_concat_positions} (concat)")
            print(f"Model: {model_name}, Input size: {input_size}, Patches: {num_patches}")
            print(f"Keeping original conditioning method: {config['conditioning_method']}")
    
    # Initialize model - select by inferred model name and architecture
    model_key = config.get('model_name', 'DiT-S/2')
    architecture = config.get('architecture', 'dit')
    
    # For VAE models, we need to use the checkpoint dimensions directly
    if config.get('use_latent_diffusion', False):
        print(f"Initializing VAE model with checkpoint-derived dimensions")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Input channels: {in_channels}")
    
    if architecture == 'songunet':
        if model_key not in SongUNet_models:
            print(f"Warning: model '{model_key}' not found. Falling back to 'SongUNet-S'.")
            model_key = "SongUNet-S"
        
        # SongUNet handles concat conditioning internally, so pass the base input channels
        print(f"Model initialization: img_resolution={input_size}, in_channels={in_channels}, out_channels={out_channels}, conditioning={config['conditioning_method']}")
        print(f"DEBUG: About to create SongUNet with in_channels={in_channels}, out_channels={out_channels}")
        
        model = SongUNet_models[model_key](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=out_channels,  # Use detected output channels
            conditioning_type='radius',
            radius_embedding_type=config['radius_embedding_type'],
            conditioning_method=config['conditioning_method'],
            radius_dropout_prob=config['radius_dropout_prob'],
        ).to(device)
        
        print(f"DEBUG: Created model, checking actual channels...")
        # Check what the model actually has
        for name, param in model.named_parameters():
            if 'enc.64x64_conv.weight' in name:
                print(f"DEBUG: Model {name} shape: {param.shape}")
                break
    elif architecture == 'unet':
        if model_key not in UNet_models:
            print(f"Warning: model '{model_key}' not found. Falling back to 'UNet-S'.")
            model_key = "UNet-S"
        model = UNet_models[model_key](
            input_size=input_size,
            in_channels=in_channels,
            radius_embedding_type=config['radius_embedding_type'],
            conditioning_method=config['conditioning_method'],
            radius_dropout_prob=config['radius_dropout_prob'],
            null_radius=config['null_radius'],
            null_embedding_type=config['null_embedding_type'],
            radius_text_table=config.get('radius_text_table')
        ).to(device)
    else:
        if model_key not in DiT_models:
            print(f"Warning: model '{model_key}' not found. Falling back to 'DiT-S/2'.")
            model_key = "DiT-S/2"
        model = DiT_models[model_key](
            input_size=input_size,
            in_channels=in_channels,
            radius_embedding_type=config['radius_embedding_type'],
            conditioning_method=config['conditioning_method'],
            radius_dropout_prob=config['radius_dropout_prob'],
            null_radius=config['null_radius'],
            null_embedding_type=config['null_embedding_type'],
            radius_text_table=config.get('radius_text_table')
        ).to(device)
    
    # Load the state dict with appropriate handling for SongUNet
    if config.get('architecture') == 'songunet':
        # For SongUNet, load the state dict directly
        state_dict = checkpoint['ema'] if 'ema' in checkpoint else checkpoint['model']
        
        # Load with strict=False to ignore missing/unexpected keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded SongUNet checkpoint")
        if missing:
            print(f"Missing keys: {len(missing)} (expected for different architectures)")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} (expected for different architectures)")
    else:
        # For DiT/UNet, use strict loading
        if 'ema' in checkpoint:
            model.load_state_dict(checkpoint['ema'], strict=False)
            print("Loaded EMA weights")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Loaded model weights")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded weights directly")
    
    model.eval()
    return model


def generate_samples(
    model,
    diffusion,
    radius_values,
    config,
    device,
    num_samples=3,
    cfg_scale=1.0,
    batch_size=1,
    num_sampling_steps=250,
):
    """Generate samples for given radius values with optional CFG."""
    samples_dict = {}
    
    input_size = config['image_size']
    use_cfg = config.get('use_cfg', False) and cfg_scale > 1.0
    use_latent_diffusion = config.get('use_latent_diffusion', False)
    
    # Load VAE if needed
    vae = None
    if use_latent_diffusion:
        try:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
            vae.eval()
            print("Loaded VAE for latent diffusion decoding")
            # For latent diffusion, we sample in latent space (8x8 for 64x64 images)
            input_size = input_size // 8
            print(f"Using latent space size: {input_size}x{input_size}")
        except Exception as e:
            print(f"Warning: Could not load VAE: {e}. Falling back to direct pixel sampling.")
            use_latent_diffusion = False
            vae = None
    
    if use_cfg:
        print(f"Using CFG with scale {cfg_scale}")
    else:
        print(f"Not using CFG (cfg_scale={cfg_scale})")
    
    batch_size = max(1, batch_size)
    
    for radius in tqdm(radius_values, desc="Generating samples"):
        samples_list = []
        
        sample_index = 0
        while sample_index < num_samples:
            current_batch = min(batch_size, num_samples - sample_index)
            
            # Determine input channels
            if config.get('architecture') == 'songunet':
                in_channels = config.get('detected_in_channels', 3)
            else:
                in_channels = getattr(model, 'in_channels', 3)
            
            # Build deterministic noise batch
            noise_batches = []
            for j in range(current_batch):
                gen = torch.Generator(device=device)
                gen.manual_seed(42 + sample_index + j)
                noise_batches.append(
                    torch.randn(
                        (1, in_channels, input_size, input_size),
                        device=device,
                        generator=gen,
                    )
                )
            z = torch.cat(noise_batches, dim=0)
            z_shape = z.shape
            
            # Check if model has forward_with_cfg method
            has_forward_with_cfg = hasattr(model, 'forward_with_cfg')
            
            # Sample with or without CFG
            with torch.no_grad():
                # Check if we're using flow matching
                is_flow_matching = config.get('use_flow_matching', False)
                
                if is_flow_matching:
                    # Flow matching sampling
                    r = torch.full((current_batch,), radius, device=device, dtype=torch.float32)

                    if use_cfg and has_forward_with_cfg:
                        # Flow matching with CFG
                        sample = diffusion.sample_with_cfg(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            cfg_scale=cfg_scale,
                            model_kwargs=dict(r=r),
                            device=device,
                        )
                    else:
                        # Flow matching without CFG
                        sample = diffusion.sample(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            model_kwargs=dict(r=r),
                            device=device,
                        )
                else:
                    # Diffusion sampling
                    if use_cfg and has_forward_with_cfg:
                        # Use the model's forward_with_cfg method
                        # Create doubled radius tensor: [conditional, unconditional]
                        r_cond = torch.full(
                            (current_batch,),
                            radius,
                            device=device,
                            dtype=torch.float32,
                        )
                        r_uncond = torch.full(
                            (current_batch,),
                            config['null_radius'],
                            device=device,
                            dtype=torch.float32,
                        )
                        r_doubled = torch.cat([r_cond, r_uncond], 0)
                        
                        # Custom forward function that uses forward_with_cfg
                        def cfg_forward(x, t, **kwargs):
                            # Note: x is NOT doubled, forward_with_cfg handles that
                            return model.forward_with_cfg(x, t, r_doubled, cfg_scale)
                        
                        sample = diffusion.p_sample_loop(
                            cfg_forward,
                            z_shape,
                            noise=z,
                            clip_denoised=True,
                            model_kwargs={},
                            progress=False,
                            device=device,
                        )
                    else:
                        # No CFG - standard sampling
                        r = torch.full(
                            (current_batch,),
                            radius,
                            device=device,
                            dtype=torch.float32,
                        )
                        sample = diffusion.p_sample_loop(
                            model,
                            z_shape,
                            noise=z,
                            clip_denoised=True,
                            model_kwargs=dict(r=r),
                            progress=False,
                            device=device,
                        )
                
                # Handle VAE decoding if using latent diffusion
                if use_latent_diffusion and vae is not None:
                    # Decode from latent space to pixel space
                    sample = sample / 0.18215  # Reverse the scaling applied during training
                    sample = vae.decode(sample).sample
            
            # Convert to image - handle channel conversion
            if sample.shape[1] == 4:
                sample = sample[:, :3, :, :]
            elif sample.shape[1] != 3:
                raise ValueError(f"Unexpected number of output channels: {sample.shape[1]}")
            
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255)
            sample = sample.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            for img in sample:
                samples_list.append(img)
                sample_index += 1
                if sample_index >= num_samples:
                    break
        
        samples_dict[radius] = samples_list
    
    return samples_dict


def visualize_results(results_interp, results_extrap, save_path, config):
    """Create comprehensive visualizations of all three metrics."""
    # Create figure with subplots for all three metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prepare data
    all_results = {**results_interp, **results_extrap}
    radius_values = sorted(all_results.keys())
    
    # Separate interpolation and extrapolation radii
    interp_radii = sorted(results_interp.keys())
    extrap_radii = sorted(results_extrap.keys())
    
    # Filter extrapolation radii to only include those outside training range
    train_min = config['train_min_radius']
    train_max = config['train_max_radius']
    extrap_radii_filtered = [r for r in extrap_radii if r < train_min or r > train_max]
    
    # Calculate overall averages for display
    all_interp_ious = [iou for r in interp_radii for iou in results_interp[r]['ious']]
    all_extrap_ious = [iou for r in extrap_radii_filtered for iou in results_extrap[r]['ious']]
    all_interp_raw_errors = [err for r in interp_radii for err in results_interp[r]['radius_errors_raw']]
    all_extrap_raw_errors = [err for r in extrap_radii_filtered for err in results_extrap[r]['radius_errors_raw']]
    all_interp_pct_errors = [err for r in interp_radii for err in results_interp[r]['radius_errors_percentage']]
    all_extrap_pct_errors = [err for r in extrap_radii_filtered for err in results_extrap[r]['radius_errors_percentage']]
    
    avg_interp_iou = np.mean(all_interp_ious) if all_interp_ious else 0
    avg_extrap_iou = np.mean(all_extrap_ious) if all_extrap_ious else 0
    avg_interp_raw_error = np.mean(np.abs(all_interp_raw_errors)) if all_interp_raw_errors else 0
    avg_extrap_raw_error = np.mean(np.abs(all_extrap_raw_errors)) if all_extrap_raw_errors else 0
    avg_interp_pct_error = np.mean(np.abs(all_interp_pct_errors)) if all_interp_pct_errors else 0
    avg_extrap_pct_error = np.mean(np.abs(all_extrap_pct_errors)) if all_extrap_pct_errors else 0
    
    # Prepare left/right extrapolation splits
    extrap_lower = [r for r in extrap_radii_filtered if r < train_min]
    extrap_upper = [r for r in extrap_radii_filtered if r > train_max]

    # Metric 1: IoU
    ax = axes[0]
    extrap_color = 'tab:orange'
    if interp_radii:
        interp_ious = [np.mean(results_interp[r]['ious']) for r in interp_radii]
        interp_ious_std = [np.std(results_interp[r]['ious']) for r in interp_radii]
        ax.errorbar(interp_radii, interp_ious, yerr=interp_ious_std, 
                    marker='o', label='Interpolation', capsize=5, markersize=8)
    
    if extrap_lower:
        extrap_ious_l = [np.mean(results_extrap[r]['ious']) for r in extrap_lower]
        extrap_ious_l_std = [np.std(results_extrap[r]['ious']) for r in extrap_lower]
        ax.errorbar(extrap_lower, extrap_ious_l, yerr=extrap_ious_l_std,
                    marker='s', label='Extrapolation', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    if extrap_upper:
        extrap_ious_u = [np.mean(results_extrap[r]['ious']) for r in extrap_upper]
        extrap_ious_u_std = [np.std(results_extrap[r]['ious']) for r in extrap_upper]
        ax.errorbar(extrap_upper, extrap_ious_u, yerr=extrap_ious_u_std,
                    marker='s', label='_nolegend_', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.axvspan(train_min, train_max, alpha=0.2, color='gray', label='Training Range')
    ax.set_xlabel('Radius')
    ax.set_ylabel('IoU')
    ax.set_title('IoU vs Radius', pad=22)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add average metrics text above the axes to avoid blocking
    # Place header just above this axes using figure coordinates (no axis shrinking)
    bbox0 = ax.get_position()
    fig.text(bbox0.x0, min(0.995, bbox0.y1 + 0.045),
             f'Avg IoU  |  Interp: {avg_interp_iou:.3f}   Extrap: {avg_extrap_iou:.3f}',
             ha='left', va='bottom', fontsize=10)
    
    # Metric 2: Raw Radius Error
    ax = axes[1]
    if interp_radii:
        interp_errors = [np.mean(results_interp[r]['radius_errors_raw']) for r in interp_radii]
        interp_errors_std = [np.std(results_interp[r]['radius_errors_raw']) for r in interp_radii]
        ax.errorbar(interp_radii, interp_errors, yerr=interp_errors_std, 
                    marker='o', label='Interpolation', capsize=5, markersize=8)
    
    if extrap_lower:
        extrap_errors_l = [np.mean(results_extrap[r]['radius_errors_raw']) for r in extrap_lower]
        extrap_errors_l_std = [np.std(results_extrap[r]['radius_errors_raw']) for r in extrap_lower]
        ax.errorbar(extrap_lower, extrap_errors_l, yerr=extrap_errors_l_std,
                    marker='s', label='Extrapolation', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    if extrap_upper:
        extrap_errors_u = [np.mean(results_extrap[r]['radius_errors_raw']) for r in extrap_upper]
        extrap_errors_u_std = [np.std(results_extrap[r]['radius_errors_raw']) for r in extrap_upper]
        ax.errorbar(extrap_upper, extrap_errors_u, yerr=extrap_errors_u_std,
                    marker='s', label='_nolegend_', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axvspan(train_min, train_max, alpha=0.2, color='gray', label='Training Range')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Raw Radius Error (pixels)')
    ax.set_title('Raw Radius Error vs Radius', pad=22)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add average metrics text above the axes to avoid blocking
    bbox1 = ax.get_position()
    fig.text(bbox1.x0, min(0.995, bbox1.y1 + 0.045),
             f'Avg |Raw Error|  |  Interp: {avg_interp_raw_error:.2f}px   Extrap: {avg_extrap_raw_error:.2f}px',
             ha='left', va='bottom', fontsize=10)
    
    # Metric 3: Percentage Radius Error
    ax = axes[2]
    if interp_radii:
        interp_pct_errors = [np.mean(results_interp[r]['radius_errors_percentage']) for r in interp_radii]
        interp_pct_errors_std = [np.std(results_interp[r]['radius_errors_percentage']) for r in interp_radii]
        ax.errorbar(interp_radii, interp_pct_errors, yerr=interp_pct_errors_std, 
                    marker='o', label='Interpolation', capsize=5, markersize=8)
    
    if extrap_lower:
        extrap_pct_errors_l = [np.mean(results_extrap[r]['radius_errors_percentage']) for r in extrap_lower]
        extrap_pct_errors_l_std = [np.std(results_extrap[r]['radius_errors_percentage']) for r in extrap_lower]
        ax.errorbar(extrap_lower, extrap_pct_errors_l, yerr=extrap_pct_errors_l_std,
                    marker='s', label='Extrapolation', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    if extrap_upper:
        extrap_pct_errors_u = [np.mean(results_extrap[r]['radius_errors_percentage']) for r in extrap_upper]
        extrap_pct_errors_u_std = [np.std(results_extrap[r]['radius_errors_percentage']) for r in extrap_upper]
        ax.errorbar(extrap_upper, extrap_pct_errors_u, yerr=extrap_pct_errors_u_std,
                    marker='s', label='_nolegend_', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axvspan(train_min, train_max, alpha=0.2, color='gray', label='Training Range')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Radius Error (%)')
    ax.set_title('Percentage Radius Error vs Radius', pad=22)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add average metrics text above the axes to avoid blocking
    bbox2 = ax.get_position()
    fig.text(bbox2.x0, min(0.995, bbox2.y1 + 0.045),
             f'Avg |% Error|  |  Interp: {avg_interp_pct_error:.1f}%   Extrap: {avg_extrap_pct_error:.1f}%',
             ha='left', va='bottom', fontsize=10)
    
    cfg_str = f" (CFG={config.get('eval_cfg_scale', 1.0)})" if config.get('use_cfg', False) else ""
    plt.suptitle(f"Model: {config['folder_name']}{cfg_str}", fontsize=14, y=0.97)
    # Keep full-height plots and add generous top margin for the header texts
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.savefig(os.path.join(save_path, 'all_metrics_comparison.png'), dpi=150)
    plt.close()
    
    # Create detailed heatmap for each metric
    create_metric_heatmaps(results_interp, results_extrap, save_path)


def create_metric_heatmaps(results_interp, results_extrap, save_path):
    """Create detailed heatmaps for each metric."""
    metrics = ['ious', 'radius_errors_raw', 'radius_errors_percentage']
    metric_names = ['IoU', 'Raw Radius Error', 'Percentage Radius Error']
    
    for metric, metric_name in zip(metrics, metric_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Interpolation heatmap
        interp_radii = sorted(results_interp.keys())
        if interp_radii:
            max_samples = max(len(results_interp[r][metric]) for r in interp_radii)
            interp_matrix = np.full((len(interp_radii), max_samples), np.nan)
            
            for i, r in enumerate(interp_radii):
                values = results_interp[r][metric]
                interp_matrix[i, :len(values)] = values
            
            sns.heatmap(interp_matrix, ax=ax1, xticklabels=False, yticklabels=interp_radii,
                        cmap='RdYlGn' if metric == 'ious' else 'RdBu_r', 
                        vmin=0 if metric == 'ious' else None,
                        vmax=1 if metric == 'ious' else None,
                        cbar_kws={'label': metric_name})
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Radius')
            ax1.set_title(f'{metric_name} - Interpolation')
        
        # Extrapolation heatmap
        extrap_radii = sorted(results_extrap.keys())
        if extrap_radii:
            max_samples = max(len(results_extrap[r][metric]) for r in extrap_radii)
            extrap_matrix = np.full((len(extrap_radii), max_samples), np.nan)
            
            for i, r in enumerate(extrap_radii):
                values = results_extrap[r][metric]
                extrap_matrix[i, :len(values)] = values
            
            sns.heatmap(extrap_matrix, ax=ax2, xticklabels=False, yticklabels=extrap_radii,
                        cmap='RdYlGn' if metric == 'ious' else 'RdBu_r',
                        vmin=0 if metric == 'ious' else None,
                        vmax=1 if metric == 'ious' else None,
                        cbar_kws={'label': metric_name})
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Radius')
            ax2.set_title(f'{metric_name} - Extrapolation')
        
        plt.suptitle(f'{metric_name} Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{metric}_heatmap.png'), dpi=150)
        plt.close()


def save_size_comparisons_simple(results, samples_dict, save_path):
    """Save simple GT vs Generated for size."""
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    for radius in sorted(results.keys()):
        fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.25))
        
        # Ground truth
        gt_img = results[radius]['gt_images'][0]
        axes[0].imshow(gt_img)
        axes[0].axis('off')
        axes[0].set_title('GT', fontsize=7)
        
        # Generated
        axes[1].imshow(samples_dict[radius][0])
        axes[1].axis('off')
        axes[1].set_title('Gen', fontsize=7)
        
        plt.tight_layout(pad=0.1)
        plt.savefig(os.path.join(save_path, 'comparisons', f'radius_{radius}_simple.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()


def save_example_comparisons(results, samples_dict, save_path, num_examples=3):
    """Save visual comparisons of generated vs ground truth."""
    # If no examples requested, skip gracefully
    if num_examples is None or num_examples <= 0:
        return
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    for radius in sorted(results.keys()):
        fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_examples, len(samples_dict[radius]))):
            # Generated image
            axes[i, 0].imshow(samples_dict[radius][i])
            axes[i, 0].set_title(f'Generated (r={radius})')
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_img = results[radius]['gt_images'][i]
            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Masks overlay
            gen_mask = results[radius]['gen_masks'][i]
            gt_mask = results[radius]['gt_masks'][i]
            
            overlay = np.zeros((*gen_mask.shape, 3))
            overlay[gen_mask == 1] = [0, 1, 0]  # Green for generated
            overlay[gt_mask == 1] = [1, 0, 0]   # Red for ground truth
            overlay[np.logical_and(gen_mask == 1, gt_mask == 1)] = [1, 1, 0]  # Yellow for overlap
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'IoU={results[radius]["ious"][i]:.3f}')
            axes[i, 2].axis('off')
            
            # Radius info
            est_r = results[radius]['estimated_radii'][i]
            err_raw = results[radius]['radius_errors_raw'][i]
            err_pct = results[radius]['radius_errors_percentage'][i]
            
            axes[i, 3].text(0.1, 0.5, f'Est. R: {est_r:.1f}\nError: {err_raw:.1f}px\nError: {err_pct:.1f}%',
                           transform=axes[i, 3].transAxes, fontsize=12, verticalalignment='center')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Radius {radius} - Examples')
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(save_path, 'comparisons', f'radius_{radius}_simple.png'), dpi=100, bbox_inches='tight')
        plt.close()


def evaluate_radius_set(radius_values, samples_dict, config, label="", actual_radii=None):
    """Evaluate a set of radius values and return results."""
    results = {}
    
    # If actual_radii is provided, create a mapping from labels to actual radii
    if actual_radii is not None:
        label_to_actual = dict(zip(radius_values, actual_radii))
        print(f"Using label-to-actual mapping: {label_to_actual}")
    else:
        label_to_actual = {r: r for r in radius_values}  # No bias
    
    for i, radius in enumerate(radius_values):
        results[radius] = {
            'ious': [],
            'precisions': [],
            'recalls': [],
            'estimated_radii': [],
            'gt_radii': [],
            'radius_errors_raw': [],
            'radius_errors_percentage': [],
            'gen_masks': [],
            'gt_masks': [],
            'gt_images': []
        }
        
        # Get actual radius for ground truth generation
        actual_radius = label_to_actual[radius]
        
        # Generate ground truth circles
        for i, gen_img in enumerate(samples_dict[radius]):
            # Generate ground truth using actual radius
            gt_img = generate_ground_truth_circle(
                radius=int(actual_radius),
                image_size=config['image_size'],
                circle_color=(255, 128, 100),
                background_color=(255, 255, 255)
            )
            
            # Calculate metrics
            metrics = calculate_circle_metrics(gen_img, gt_img, actual_radius)
            
            # Store results
            results[radius]['ious'].append(metrics['iou'])
            results[radius]['precisions'].append(metrics['precision'])
            results[radius]['recalls'].append(metrics['recall'])
            results[radius]['estimated_radii'].append(metrics['estimated_radius'])
            results[radius]['gt_radii'].append(actual_radius)  # Store the actual radius
            results[radius]['radius_errors_raw'].append(metrics['radius_error_raw'])
            results[radius]['radius_errors_percentage'].append(metrics['radius_error_percentage'])
            results[radius]['gen_masks'].append(metrics['gen_mask'])
            results[radius]['gt_masks'].append(metrics['gt_mask'])
            results[radius]['gt_images'].append(gt_img)
    
    return results


def calculate_summary_statistics(results, label=""):
    """Calculate summary statistics for a set of results."""
    summary = {}
    all_ious = []
    all_radius_errors_raw = []
    all_radius_errors_percentage = []
    
    for radius in results.keys():
        summary[radius] = {
            'mean_iou': np.mean(results[radius]['ious']),
            'std_iou': np.std(results[radius]['ious']),
            'mean_precision': np.mean(results[radius]['precisions']),
            'mean_recall': np.mean(results[radius]['recalls']),
            'mean_estimated_radius': np.mean(results[radius]['estimated_radii']),
            'mean_radius_error_raw': np.mean(results[radius]['radius_errors_raw']),
            'std_radius_error_raw': np.std(results[radius]['radius_errors_raw']),
            'mean_radius_error_percentage': np.mean(results[radius]['radius_errors_percentage']),
            'std_radius_error_percentage': np.std(results[radius]['radius_errors_percentage'])
        }
        
        all_ious.extend(results[radius]['ious'])
        all_radius_errors_raw.extend(results[radius]['radius_errors_raw'])
        all_radius_errors_percentage.extend(results[radius]['radius_errors_percentage'])
    
    # Overall statistics
    overall = {
        'mean_iou': np.mean(all_ious),
        'std_iou': np.std(all_ious),
        'median_iou': np.median(all_ious),
        'min_iou': np.min(all_ious),
        'max_iou': np.max(all_ious),
        'mean_radius_error_raw': np.mean(all_radius_errors_raw),
        'std_radius_error_raw': np.std(all_radius_errors_raw),
        'mean_abs_radius_error_raw': np.mean(np.abs(all_radius_errors_raw)),
        'mean_radius_error_percentage': np.mean(all_radius_errors_percentage),
        'std_radius_error_percentage': np.std(all_radius_errors_percentage),
        'mean_abs_radius_error_percentage': np.mean(np.abs(all_radius_errors_percentage))
    }
    
    return summary, overall


def print_results_table_enhanced(summary_train, overall_train, summary_interp, overall_interp, summary_extrap, overall_extrap, cfg_scale=1.0):
    """Print a comprehensive results table with training results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    if cfg_scale > 1.0:
        print(f"CFG Scale: {cfg_scale}")
    print("="*80)
    
    # Exact training results
    print("\nEXACT TRAINING RESULTS:")
    print(f"{'Radius':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Raw Error':<15} {'% Error':<15}")
    print("-"*70)
    for radius in sorted(summary_train.keys()):
        s = summary_train[radius]
        print(f"{radius:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_radius_error_raw']:<15.2f} {s['mean_radius_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_train['mean_iou']:<12.4f} {overall_train['std_iou']:<12.4f} "
          f"{overall_train['mean_radius_error_raw']:<15.2f} {overall_train['mean_radius_error_percentage']:<15.1f}")
    
    # Interpolation results
    print("\nINTERPOLATION RESULTS:")
    print(f"{'Radius':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Raw Error':<15} {'% Error':<15}")
    print("-"*70)
    for radius in sorted(summary_interp.keys()):
        s = summary_interp[radius]
        print(f"{radius:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_radius_error_raw']:<15.2f} {s['mean_radius_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_interp['mean_iou']:<12.4f} {overall_interp['std_iou']:<12.4f} "
          f"{overall_interp['mean_radius_error_raw']:<15.2f} {overall_interp['mean_radius_error_percentage']:<15.1f}")
    
    # Extrapolation results
    print("\nEXTRAPOLATION RESULTS:")
    print(f"{'Radius':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Raw Error':<15} {'% Error':<15}")
    print("-"*70)
    for radius in sorted(summary_extrap.keys()):
        s = summary_extrap[radius]
        print(f"{radius:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_radius_error_raw']:<15.2f} {s['mean_radius_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_extrap['mean_iou']:<12.4f} {overall_extrap['std_iou']:<12.4f} "
          f"{overall_extrap['mean_radius_error_raw']:<15.2f} {overall_extrap['mean_radius_error_percentage']:<15.1f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'Exact Training':<20} {'Interpolation':<20} {'Extrapolation':<20}")
    print("-"*90)
    print(f"{'Mean IoU':<30} {overall_train['mean_iou']:<20.4f} {overall_interp['mean_iou']:<20.4f} {overall_extrap['mean_iou']:<20.4f}")
    print(f"{'Mean |Raw Error| (pixels)':<30} {overall_train['mean_abs_radius_error_raw']:<20.2f} {overall_interp['mean_abs_radius_error_raw']:<20.2f} {overall_extrap['mean_abs_radius_error_raw']:<20.2f}")
    print(f"{'Mean |% Error|':<30} {overall_train['mean_abs_radius_error_percentage']:<20.1f} {overall_interp['mean_abs_radius_error_percentage']:<20.1f} {overall_extrap['mean_abs_radius_error_percentage']:<20.1f}")

def print_results_table(summary_interp, overall_interp, summary_extrap, overall_extrap, cfg_scale=1.0):
    """Print a comprehensive results table (backward compatibility)."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    if cfg_scale > 1.0:
        print(f"CFG Scale: {cfg_scale}")
    print("="*80)
    
    # Interpolation results
    print("\nINTERPOLATION RESULTS:")
    print(f"{'Radius':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Raw Error':<15} {'% Error':<15}")
    print("-"*70)
    for radius in sorted(summary_interp.keys()):
        s = summary_interp[radius]
        print(f"{radius:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_radius_error_raw']:<15.2f} {s['mean_radius_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_interp['mean_iou']:<12.4f} {overall_interp['std_iou']:<12.4f} "
          f"{overall_interp['mean_radius_error_raw']:<15.2f} {overall_interp['mean_radius_error_percentage']:<15.1f}")
    
    # Extrapolation results
    print("\nEXTRAPOLATION RESULTS:")
    print(f"{'Radius':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Raw Error':<15} {'% Error':<15}")
    print("-"*70)
    for radius in sorted(summary_extrap.keys()):
        s = summary_extrap[radius]
        print(f"{radius:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_radius_error_raw']:<15.2f} {s['mean_radius_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_extrap['mean_iou']:<12.4f} {overall_extrap['std_iou']:<12.4f} "
          f"{overall_extrap['mean_radius_error_raw']:<15.2f} {overall_extrap['mean_radius_error_percentage']:<15.1f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'Interpolation':<20} {'Extrapolation':<20}")
    print("-"*70)
    print(f"{'Mean IoU':<30} {overall_interp['mean_iou']:<20.4f} {overall_extrap['mean_iou']:<20.4f}")
    print(f"{'Mean |Raw Error| (pixels)':<30} {overall_interp['mean_abs_radius_error_raw']:<20.2f} {overall_extrap['mean_abs_radius_error_raw']:<20.2f}")
    print(f"{'Mean |% Error|':<30} {overall_interp['mean_abs_radius_error_percentage']:<20.1f} {overall_extrap['mean_abs_radius_error_percentage']:<20.1f}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse configuration from checkpoint path
    config = parse_checkpoint_path(args.ckpt)
    config['ckpt_path'] = args.ckpt
    config['eval_cfg_scale'] = args.cfg_scale  # Store the evaluation CFG scale
    
    # Create output directory with CFG info if applicable
    cfg_suffix = f"_cfg{args.cfg_scale}" if config.get('use_cfg', False) and args.cfg_scale > 1.0 else ""
    output_dir = os.path.join("eval_resume", f"{config['folder_name']}{cfg_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("\nInferred Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Additional CFG info
    if config.get('use_cfg', False):
        print(f"\nModel was trained with CFG (dropout_prob={config.get('radius_dropout_prob', 0.1)})")
        print(f"Evaluating with cfg_scale={args.cfg_scale}")
    else:
        print("\nModel was trained without CFG")
        if args.cfg_scale > 1.0:
            print("Warning: cfg_scale > 1.0 but model wasn't trained with CFG - results may be suboptimal")
    
    # Get evaluation radius values
    train_labels, interp_labels, extrap_labels, train_radii, interp_radii, extrap_radii = get_evaluation_radii(
        config['train_min_radius'], 
        config['train_max_radius'],
        config['bias'],
        interp_mode=args.interp_mode
    )
    
    print(f"\nExact training labels: {train_labels}")
    print(f"Interpolation labels: {interp_labels}")
    print(f"Extrapolation labels: {extrap_labels}")
    if config['bias'] != 0:
        print(f"Exact training actual radii: {train_radii}")
        print(f"Interpolation actual radii: {interp_radii}")
        print(f"Extrapolation actual radii: {extrap_radii}")
    
    # Load model
    model = load_model(config, device)
    
    # Create diffusion or flow matching with appropriate settings
    if config.get('use_flow_matching', False):
        print("Using Flow Matching sampler")
        diffusion = FlowMatching(sigma_min=0.0, sigma_data=1.0, use_sigmoid_time=True)
    elif config.get('architecture') == 'songunet':
        print("Using Diffusion sampler (SongUNet)")
        diffusion = create_diffusion(timestep_respacing="", learn_sigma=False)
    else:
        print("Using Diffusion sampler")
        diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))
    
    # Generate samples for all radii (use labels for generation)
    all_labels = train_labels + interp_labels + extrap_labels
    print(f"\nGenerating samples for all label values...")
    samples_dict = generate_samples(
        model, diffusion, all_labels, config, device, 
        num_samples=args.num_samples_per_radius,
        cfg_scale=args.cfg_scale,
        batch_size=args.eval_batch_size,
        num_sampling_steps=args.num_sampling_steps,
    )
    
    # Evaluate exact training
    print("\nEvaluating exact training performance...")
    results_train = evaluate_radius_set(
        train_labels, 
        {r: samples_dict[r] for r in train_labels}, 
        config, 
        label="exact training",
        actual_radii=train_radii  # Pass actual radii for ground truth
    )
    
    # Evaluate interpolation
    print("\nEvaluating interpolation performance...")
    results_interp = evaluate_radius_set(
        interp_labels, 
        {r: samples_dict[r] for r in interp_labels}, 
        config, 
        label="interpolation",
        actual_radii=interp_radii  # Pass actual radii for ground truth
    )
    
    # Evaluate extrapolation
    print("\nEvaluating extrapolation performance...")
    results_extrap = evaluate_radius_set(
        extrap_labels, 
        {r: samples_dict[r] for r in extrap_labels}, 
        config, 
        label="extrapolation",
        actual_radii=extrap_radii  # Pass actual radii for ground truth
    )
    
    # Calculate summary statistics
    summary_train, overall_train = calculate_summary_statistics(results_train, "Exact Training")
    summary_interp, overall_interp = calculate_summary_statistics(results_interp, "Interpolation")
    summary_extrap, overall_extrap = calculate_summary_statistics(results_extrap, "Extrapolation")
    
    # Print results
    print_results_table_enhanced(summary_train, overall_train, summary_interp, overall_interp, summary_extrap, overall_extrap, cfg_scale=args.cfg_scale)
    
    # Save results
    results_data = {
        'config': config,
        'cfg_scale': args.cfg_scale,
        'exact_training': {
            'radii': train_labels,
            'summary': summary_train,
            'overall': overall_train,
            'detailed_results': {
                str(r): {
                    'ious': results_train[r]['ious'],
                    'precisions': results_train[r]['precisions'],
                    'recalls': results_train[r]['recalls'],
                    'estimated_radii': [float(x) for x in results_train[r]['estimated_radii']],
                    'radius_errors_raw': [float(x) for x in results_train[r]['radius_errors_raw']],
                    'radius_errors_percentage': [float(x) for x in results_train[r]['radius_errors_percentage']]
                }
                for r in train_labels
            }
        },
        'interpolation': {
            'radii': interp_labels,
            'summary': summary_interp,
            'overall': overall_interp,
            'detailed_results': {
                str(r): {
                    'ious': results_interp[r]['ious'],
                    'precisions': results_interp[r]['precisions'],
                    'recalls': results_interp[r]['recalls'],
                    'estimated_radii': [float(x) for x in results_interp[r]['estimated_radii']],
                    'radius_errors_raw': [float(x) for x in results_interp[r]['radius_errors_raw']],
                    'radius_errors_percentage': [float(x) for x in results_interp[r]['radius_errors_percentage']]
                }
                for r in interp_labels
            }
        },
        'extrapolation': {
            'radii': extrap_labels,
            'summary': summary_extrap,
            'overall': overall_extrap,
            'detailed_results': {
                str(r): {
                    'ious': results_extrap[r]['ious'],
                    'precisions': results_extrap[r]['precisions'],
                    'recalls': results_extrap[r]['recalls'],
                    'estimated_radii': [float(x) for x in results_extrap[r]['estimated_radii']],
                    'radius_errors_raw': [float(x) for x in results_extrap[r]['radius_errors_raw']],
                    'radius_errors_percentage': [float(x) for x in results_extrap[r]['radius_errors_percentage']]
                }
                for r in extrap_labels
            }
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_results(results_interp, results_extrap, output_dir, config)
    
    # Save example comparisons
    # Use simple comparison saving
    save_size_comparisons_simple(
        {**results_train, **results_interp, **results_extrap}, 
        samples_dict, 
        output_dir
    )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate continuous radius DiT model with auto-configuration and CFG support')
    
    # Only required argument is checkpoint path
    parser.add_argument("--ckpt", type=str, default=f"{VGL_ROOT}/circle_model_32_radii_linear_5to20_64x64_4workers/003-DiT-S-2-continuous/checkpoints/final_0234000.pt",
                        help="Path to model checkpoint (will infer all settings from path)")
    
    # CFG parameter
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger guidance)")
    
    # Optional sampling parameters
    parser.add_argument("--num-samples-per-radius", type=int, default=3,
                        help="Number of samples to generate per radius")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of DDPM sampling steps")
    parser.add_argument("--num-visual-examples", type=int, default=1,
                        help="Number of visual examples to save per radius")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of samples to generate in parallel during evaluation")
    
    parser.add_argument("--interp-mode", type=str, default="odd", choices=["odd","midpoints"],
                        help="Interpolation radii selection: odd integers or half-step midpoints")

    args = parser.parse_args()
    main(args)
