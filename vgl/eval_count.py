#!/usr/bin/env python
"""
Enhanced evaluation script for continuous count DiT model.
Automatically infers model parameters from checkpoint path and evaluates 
interpolation vs extrapolation performance with count detection metrics.
Includes proper CFG (Classifier-Free Guidance) support.
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
from skimage import measure, morphology
from scipy import ndimage
# Inline simple comparison function
def save_count_comparisons_simple(results, samples_dict, save_path):
    """Save generated-only for count."""
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    for count in sorted(results.keys()):
        fig, ax = plt.subplots(1, 1, figsize=(1.25, 1.25))
        
        ax.imshow(samples_dict[count][0])
        ax.axis('off')
        
        plt.tight_layout(pad=0.1)
        plt.savefig(os.path.join(save_path, 'comparisons', f'count_{count}_simple.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()

# Import model components
from vgl.models import DiT_models_continuous as DiT_models
from vgl.unet_models import UNet_models
from vgl.unet_models_song import SongUNet_models
from vgl.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

def parse_checkpoint_path(ckpt_path):
    """Parse model configuration from checkpoint path."""
    # Get the folder name from the path - be more flexible
    path_parts = Path(ckpt_path).parts
    folder_name = None
    
    # Look for various patterns in the path
    for part in path_parts:
        if any(keyword in part.lower() for keyword in ['count_model', 'songunet', 'count_results', 'dit-s', 'unet']):
            folder_name = part
            break
    
    # If no specific folder found, use the parent directory name
    if not folder_name:
        # Use the results directory name (e.g., count_results_dataset1_grid)
        for part in reversed(path_parts):
            if 'count_results' in part:
                folder_name = part
                break
        
        # Fallback to using the full path as folder name
        if not folder_name:
            folder_name = Path(ckpt_path).parent.name
    
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
    elif 'rotary' in name_lower or 'rot' in name_lower:
        config['radius_embedding_type'] = 'rotary'
    elif 'sinusoidal' in name_lower or 'sin' in name_lower:
        config['radius_embedding_type'] = 'sinusoidal'
    elif 'linear' in name_lower:
        config['radius_embedding_type'] = 'linear'
    else:
        config['radius_embedding_type'] = 'linear'  # default for your count experiments
    
    # Parse conditioning method
    if 'concat' in name_lower:
        config['conditioning_method'] = 'concat'
    else:
        config['conditioning_method'] = 'concat'  # default for your count experiments
    
    # Parse CFG settings (respect explicit no_cfg)
    if 'no_cfg' in name_lower or 'nocfg' in name_lower:
        config['radius_dropout_prob'] = 0.0
        config['use_cfg'] = False
        config['null_embedding_type'] = 'none'
    elif 'cfg' in name_lower:
        config['radius_dropout_prob'] = 0.1
        config['use_cfg'] = True  # Flag to indicate CFG was used in training
        
        # Check for null embedding type including "none"
        if 'none' in name_lower or 'no_emb' in name_lower or 'noemb' in name_lower:
            config['null_embedding_type'] = 'none'
        elif 'zero' in name_lower:
            config['null_embedding_type'] = 'zero'
        elif 'learnable' in name_lower or 'learn' in name_lower:
            config['null_embedding_type'] = 'learnable'
        else:
            # Default for CFG
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
    
    # Parse training count range
    range_match = re.search(r'(\d+)to(\d+)', folder_name)
    if range_match:
        config['train_min_count'] = int(range_match.group(1))
        config['train_max_count'] = int(range_match.group(2))
    else:
        # Try alternative patterns
        range_match = re.search(r'_(\d+)_(\d+)_', folder_name)
        if range_match:
            config['train_min_count'] = int(range_match.group(1))
            config['train_max_count'] = int(range_match.group(2))
        else:
            # Parse from dataset name patterns
            if 'dataset1' in folder_name.lower():
                config['train_min_count'] = 3
                config['train_max_count'] = 25
            elif 'dataset2' in folder_name.lower():
                config['train_min_count'] = 2
                config['train_max_count'] = 7
            elif 'dataset3' in folder_name.lower():
                config['train_min_count'] = 3
                config['train_max_count'] = 25
            else:
                # Default to even counts 4-96
                config['train_min_count'] = 4
                config['train_max_count'] = 96
    
    # Parse bias information
    bias_match = re.search(r'bias(\d+)', folder_name)
    if bias_match:
        config['bias'] = int(bias_match.group(1))
        print(f"Detected bias: {config['bias']}")
        # Adjust training range to actual count values
        config['actual_min_count'] = config['train_min_count'] - config['bias']
        config['actual_max_count'] = config['train_max_count'] - config['bias']
        print(f"Actual training counts: {config['actual_min_count']} to {config['actual_max_count']} objects")
    else:
        config['bias'] = 0
        config['actual_min_count'] = config['train_min_count']
        config['actual_max_count'] = config['train_max_count']
    
    # Set other defaults
    config['use_latent_diffusion'] = False
    config['null_radius'] = 0.0  # We'll use this as null_count
    
    # Parse architecture (DiT, UNet, or SongUNet)
    if 'songunet' in name_lower:
        config['architecture'] = 'songunet'
        # SongUNet models use linear embedding and concat conditioning
        config['radius_embedding_type'] = 'linear'
        config['conditioning_method'] = 'concat'
    elif 'unet' in name_lower:
        config['architecture'] = 'unet'
    else:
        config['architecture'] = 'dit'
    
    # Parse model name/size from path components
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


def get_evaluation_counts(train_min, train_max, bias=0, interp_mode='odd'):
    """Get interpolation and extrapolation count values, accounting for bias."""
    # If there's a bias, we need to test specific ACTUAL counts and map them to the correct labels
    if bias != 0:
        print(f"Using biased evaluation: labels {train_min} to {train_max} (actual counts {train_min-bias} to {train_max-bias} objects)")
        
        # For biased models, we want to test specific ACTUAL counts and map them to the correct labels
        
        # Interpolation set inside the training range
        actual_train_min = train_min - bias
        actual_train_max = train_max - bias

        if interp_mode == 'midpoints':
            # Half-steps between integers, e.g., 4.5, 6.5, ..., (max-0.5)
            interpolation_actual_counts = [c + 0.5 for c in range(actual_train_min, actual_train_max, 2)]
        else:
            # Odd integers inside the range (since training was on even)
            interpolation_actual_counts = [float(c) for c in range(actual_train_min+1, actual_train_max, 2)]
        
        # Convert actual counts to labels for generation
        interpolation_labels = [c + bias for c in interpolation_actual_counts]
        
        # Extrapolation: Test actual counts 1, 2, 3 for lower, and higher values for upper
        extrapolation_actual_counts = []
        
        # Lower extrapolation: Test actual counts 1, 2, 3
        for actual_count in range(1, 4):
            extrapolation_actual_counts.append(float(actual_count))
        
        # Upper extrapolation: Test actual counts beyond training range (even values)
        for i in range(2, 8, 2):  # 2, 4, 6 steps beyond
            actual_count = actual_train_max + i
            extrapolation_actual_counts.append(float(actual_count))
        
        # Convert actual counts to labels for generation
        extrapolation_labels = [c + bias for c in extrapolation_actual_counts]
        
        print(f"Interpolation: actual counts {interpolation_actual_counts} -> generation labels {interpolation_labels}")
        print(f"Lower extrapolation: actual counts {extrapolation_actual_counts[:3]} -> generation labels {extrapolation_labels[:3]}")
        print(f"Upper extrapolation: actual counts {extrapolation_actual_counts[3:]} -> generation labels {extrapolation_labels[3:]}")
        
        return interpolation_labels, extrapolation_labels, interpolation_actual_counts, extrapolation_actual_counts
    else:
        # No bias - standard evaluation
        # Interpolation set inside the training range (odd counts since training was on even)
        if interp_mode == 'midpoints':
            interpolation_counts = [c + 0.5 for c in range(train_min, train_max, 2)]
        else:
            interpolation_counts = [float(c) for c in range(train_min+1, train_max, 2)]  # Odd counts
        
        # Extrapolation: values below (down to 1) and values above
        extrapolation_counts = []
        
        # Below range
        for c in [1, 2, 3]:
            if c < train_min:
                extrapolation_counts.append(float(c))
        
        # Above range (even values to match training pattern)
        for i in range(2, 8, 2):  # 2, 4, 6 steps beyond
            c = train_max + i
            extrapolation_counts.append(float(c))
        
        return interpolation_counts, extrapolation_counts, interpolation_counts, extrapolation_counts


def generate_ground_truth_count_image(count, image_size=64, min_radius=2, max_radius=8,
                                     circle_color_range=((100, 255), (100, 255), (100, 255)),
                                     background_color=(255, 255, 255), max_attempts=1000):
    """Generate a ground truth count image (same as training data)."""
    img = Image.new('RGB', (image_size, image_size), background_color)
    draw = ImageDraw.Draw(img)
    
    circles = []  # Store (x, y, radius) for overlap checking
    placed_count = 0
    attempts = 0
    
    while placed_count < count and attempts < max_attempts:
        # Generate random circle properties
        radius = np.random.randint(min_radius, max_radius + 1)
        x = np.random.randint(radius, image_size - radius - 1)
        y = np.random.randint(radius, image_size - radius - 1)
        
        # Check for overlap with existing circles
        overlap = False
        for cx, cy, cr in circles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < (radius + cr + 1):  # +1 for small buffer
                overlap = True
                break
        
        if not overlap:
            # Generate random color within range
            color = (
                np.random.randint(*circle_color_range[0]),
                np.random.randint(*circle_color_range[1]),
                np.random.randint(*circle_color_range[2])
            )
            
            # Draw circle
            left = x - radius
            top = y - radius
            right = x + radius
            bottom = y + radius
            
            draw.ellipse([left, top, right, bottom], fill=color, outline=color)
            circles.append((x, y, radius))
            placed_count += 1
        
        attempts += 1
    
    return img


def detect_and_count_objects(image, min_area=30, max_area=None):
    """
    Detect and count circular objects in an image.
    
    Returns:
        count: Number of detected objects
        centers: List of (x, y) centers
        radii: List of estimated radii
        mask: Binary mask of detected objects
    """
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
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get binary image
    # Use Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (objects should be white)
    if np.mean(binary) > 127:  # If mostly white, invert
        binary = 255 - binary
    
    # Clean up the binary image using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    labeled = measure.label(binary > 0)
    regions = measure.regionprops(labeled)
    
    # Filter regions by area and extract features
    centers = []
    radii = []
    valid_regions = []
    
    for region in regions:
        if min_area <= region.area <= max_area:
            # Estimate radius from area (assuming circular objects)
            estimated_radius = np.sqrt(region.area / np.pi)
            
            # Use centroid as center
            y, x = region.centroid
            centers.append((x, y))
            radii.append(estimated_radius)
            valid_regions.append(region)
    
    count = len(centers)
    
    # Create mask of detected objects
    mask = np.zeros_like(binary)
    for region in valid_regions:
        mask[labeled == region.label] = 1
    
    return count, centers, radii, mask


def calculate_count_metrics(generated_img, ground_truth_img, gt_count):
    """
    Calculate various metrics between generated and ground truth count images.
    
    Returns:
        dict with estimated count, count error, detection accuracy, etc.
    """
    # Detect objects in both images
    gen_count, gen_centers, gen_radii, gen_mask = detect_and_count_objects(generated_img)
    gt_count_detected, gt_centers, gt_radii, gt_mask = detect_and_count_objects(ground_truth_img)
    
    # Calculate count error
    count_error = gen_count - gt_count
    count_error_percentage = (count_error / gt_count * 100) if gt_count > 0 else 0
    
    # Calculate absolute errors
    abs_count_error = abs(count_error)
    abs_count_error_percentage = abs(count_error_percentage)
    
    # Calculate IoU between detection masks
    intersection = np.logical_and(gen_mask, gt_mask).sum()
    union = np.logical_or(gen_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    
    # Calculate precision and recall for detection
    precision = intersection / gen_mask.sum() if gen_mask.sum() > 0 else 0
    recall = intersection / gt_mask.sum() if gt_mask.sum() > 0 else 0
    
    return {
        'estimated_count': gen_count,
        'gt_count': gt_count,
        'gt_count_detected': gt_count_detected,  # What we actually detected in GT
        'count_error': count_error,
        'count_error_percentage': count_error_percentage,
        'abs_count_error': abs_count_error,
        'abs_count_error_percentage': abs_count_error_percentage,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'gen_mask': gen_mask,
        'gt_mask': gt_mask,
        'gen_centers': gen_centers,
        'gt_centers': gt_centers
    }


def load_model(config, device):
    """Load the trained DiT model."""
    # Determine input size and channels
    input_size = config['image_size']
    in_channels = 3  # Default
    
    # Load checkpoint first to check model configuration
    print(f"Loading checkpoint...")
    checkpoint = torch.load(config['ckpt_path'], map_location=device)
    
    # Get the state dict to check
    state_dict = None
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # For SongUNet, detect channels from checkpoint (concat conditioning adds channels)
    if config.get('architecture') == 'songunet':
        config['use_latent_diffusion'] = False
        found_channels = False
        checkpoint_channels = None
        for key in state_dict.keys():
            if 'enc.' in key and 'conv.weight' in key:
                checkpoint_channels = state_dict[key].shape[1]
                print(f"SongUNet checkpoint has {checkpoint_channels} input channels (from {key})")
                found_channels = True
                break
        if not found_channels:
            checkpoint_channels = 3
            print("SongUNet detected but could not infer input channels; using default 3-channel RGB")

        # Infer conditioning method and base channels
        extra_channels = 1  # count uses radius-style scalar conditioning
        if checkpoint_channels > 3:
            if config.get('conditioning_method') != 'concat':
                print("Detected extra input channels; overriding conditioning_method to 'concat'")
            config['conditioning_method'] = 'concat'
            base_channels = checkpoint_channels - extra_channels
        else:
            # No extra channels -> adaln-style conditioning
            if config.get('conditioning_method') != 'adaln':
                print("Detected 3-channel input; overriding conditioning_method to 'adaln'")
            config['conditioning_method'] = 'adaln'
            base_channels = checkpoint_channels

        in_channels = max(1, base_channels)
        out_channels = in_channels
        
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
    
    # Check positional embedding size to infer conditioning method
    if 'pos_embed' in state_dict:
        pos_embed_shape = state_dict['pos_embed'].shape
        num_patches = (input_size // 2) ** 2  # For patch size 2
        if config['model_name'] == 'DiT-S/1':
            num_patches = input_size ** 2  # For patch size 1
        elif config['model_name'] == 'DiT-S/4':
            num_patches = (input_size // 4) ** 2  # For patch size 4
        elif config['model_name'] == 'DiT-S/8':
            num_patches = (input_size // 8) ** 2  # For patch size 8
        
        expected_adaln_positions = num_patches
        expected_concat_positions = num_patches + 2  # +2 for time and count tokens
        
        actual_positions = pos_embed_shape[1]
        
        if actual_positions == expected_concat_positions:
            print(f"Detected concat conditioning from pos_embed shape: {pos_embed_shape}")
            config['conditioning_method'] = 'concat'
        elif actual_positions == expected_adaln_positions:
            print(f"Detected adaln conditioning from pos_embed shape: {pos_embed_shape}")
            config['conditioning_method'] = 'adaln'
        else:
            print(f"Warning: Unexpected pos_embed shape {pos_embed_shape}. Expected {expected_adaln_positions} (adaln) or {expected_concat_positions} (concat)")
            print(f"Keeping original conditioning method: {config['conditioning_method']}")
    
    # Initialize model - select by inferred model name and architecture
    model_key = config.get('model_name', 'DiT-S/2')
    architecture = config.get('architecture', 'dit')
    
    if architecture == 'songunet':
        if model_key not in SongUNet_models:
            print(f"Warning: model '{model_key}' not found. Falling back to 'SongUNet-S'.")
            model_key = "SongUNet-S"
        
        # SongUNet handles concat conditioning internally, so pass the base input channels
        print(f"Model initialization: base_channels={in_channels}, conditioning={config['conditioning_method']}")
        
        model = SongUNet_models[model_key](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=out_channels,  # Use detected output channels
            conditioning_type='radius',  # We'll use the same embedding for counts
            radius_embedding_type=config['radius_embedding_type'],
            conditioning_method=config['conditioning_method'],
            radius_dropout_prob=config['radius_dropout_prob'],
        ).to(device)
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
            null_embedding_type=config['null_embedding_type']
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
            null_embedding_type=config['null_embedding_type']
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


def generate_samples(model, diffusion, count_values, config, device, num_samples=3, cfg_scale=1.0, batch_size=1):
    """Generate samples for given count values with optional CFG."""
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
            latent_size = input_size // 8
        except Exception as e:
            print(f"Warning: Could not load VAE: {e}. Falling back to direct pixel sampling.")
            use_latent_diffusion = False
            vae = None
    
    if use_cfg:
        print(f"Using CFG with scale {cfg_scale}")
    else:
        print(f"Not using CFG (cfg_scale={cfg_scale})")
    
    batch_size = max(1, batch_size)
    
    for count in tqdm(count_values, desc="Generating samples"):
        samples_list = []
        sample_index = 0
        
        while sample_index < num_samples:
            current_batch = min(batch_size, num_samples - sample_index)
            
            # Sample noise - use detected input channels
            if config.get('architecture') == 'songunet':
                # For SongUNet, use the detected base input channels from config
                detected_in_channels = config.get('detected_in_channels', 3)
                in_channels = detected_in_channels
            else:
                # For DiT/UNet, use model.in_channels if available
                in_channels = getattr(model, 'in_channels', 3)

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
            
            # Check if model has forward_with_cfg method
            has_forward_with_cfg = hasattr(model, 'forward_with_cfg')
            
            # Sample with or without CFG
            with torch.no_grad():
                if use_cfg and has_forward_with_cfg:
                    # Use the model's forward_with_cfg method
                    # Create doubled count tensor: [conditional, unconditional]
                    r_cond = torch.full((current_batch,), count, device=device, dtype=torch.float32)
                    r_uncond = torch.full((current_batch,), config['null_radius'], device=device, dtype=torch.float32)
                    r_doubled = torch.cat([r_cond, r_uncond], 0)
                    
                    # Custom forward function that uses forward_with_cfg
                    def cfg_forward(x, t, **kwargs):
                        # Note: x is NOT doubled, forward_with_cfg handles that
                        return model.forward_with_cfg(x, t, r_doubled, cfg_scale)
                    
                    sample = diffusion.p_sample_loop(
                        cfg_forward, z.shape, clip_denoised=True,
                        model_kwargs={}, progress=False, device=device
                    )
                else:
                    # No CFG - standard sampling
                    r = torch.full((current_batch,), count, device=device, dtype=torch.float32)
                    sample = diffusion.p_sample_loop(
                        model, z.shape, clip_denoised=True,
                        model_kwargs=dict(r=r), progress=False, device=device
                    )
                
                # Handle VAE decoding if using latent diffusion
                if use_latent_diffusion and vae is not None:
                    # Decode from latent space to pixel space
                    sample = sample / 0.18215  # Reverse the scaling applied during training
                    sample = vae.decode(sample).sample
            
            # Convert to image - handle channel conversion
            if sample.shape[1] == 3:
                # Standard 3-channel RGB output
                sample = torch.clamp(127.5 * sample + 128.0, 0, 255)
                sample = sample.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            elif sample.shape[1] == 4:
                # 4-channel output - take first 3 channels as RGB
                sample = sample[:, :3, :, :]
                sample = torch.clamp(127.5 * sample + 128.0, 0, 255)
                sample = sample.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            else:
                raise ValueError(f"Unexpected number of output channels: {sample.shape[1]}")
            
            for img in sample:
                samples_list.append(img)
                sample_index += 1
                if sample_index >= num_samples:
                    break
        
        samples_dict[count] = samples_list
    
    return samples_dict


def visualize_results(results_interp, results_extrap, save_path, config):
    """Create comprehensive visualizations of all three metrics."""
    # Create figure with subplots for all three metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prepare data
    all_results = {**results_interp, **results_extrap}
    count_values = sorted(all_results.keys())
    
    # Separate interpolation and extrapolation counts
    interp_counts = sorted(results_interp.keys())
    extrap_counts = sorted(results_extrap.keys())
    
    # Filter extrapolation counts to only include those outside training range
    train_min = config['train_min_count']
    train_max = config['train_max_count']
    extrap_counts_filtered = [c for c in extrap_counts if c < train_min or c > train_max]
    
    # Calculate overall averages for display
    all_interp_ious = [iou for c in interp_counts for iou in results_interp[c]['ious']]
    all_extrap_ious = [iou for c in extrap_counts_filtered for iou in results_extrap[c]['ious']]
    all_interp_count_errors = [err for c in interp_counts for err in results_interp[c]['abs_count_errors']]
    all_extrap_count_errors = [err for c in extrap_counts_filtered for err in results_extrap[c]['abs_count_errors']]
    all_interp_pct_errors = [err for c in interp_counts for err in results_interp[c]['abs_count_errors_percentage']]
    all_extrap_pct_errors = [err for c in extrap_counts_filtered for err in results_extrap[c]['abs_count_errors_percentage']]
    
    avg_interp_iou = np.mean(all_interp_ious) if all_interp_ious else 0
    avg_extrap_iou = np.mean(all_extrap_ious) if all_extrap_ious else 0
    avg_interp_count_error = np.mean(all_interp_count_errors) if all_interp_count_errors else 0
    avg_extrap_count_error = np.mean(all_extrap_count_errors) if all_extrap_count_errors else 0
    avg_interp_pct_error = np.mean(all_interp_pct_errors) if all_interp_pct_errors else 0
    avg_extrap_pct_error = np.mean(all_extrap_pct_errors) if all_extrap_pct_errors else 0
    
    # Prepare left/right extrapolation splits
    extrap_lower = [c for c in extrap_counts_filtered if c < train_min]
    extrap_upper = [c for c in extrap_counts_filtered if c > train_max]

    # Metric 1: IoU
    ax = axes[0]
    extrap_color = 'tab:orange'
    if interp_counts:
        interp_ious = [np.mean(results_interp[c]['ious']) for c in interp_counts]
        interp_ious_std = [np.std(results_interp[c]['ious']) for c in interp_counts]
        ax.errorbar(interp_counts, interp_ious, yerr=interp_ious_std, 
                    marker='o', label='Interpolation', capsize=5, markersize=8)
    
    if extrap_lower:
        extrap_ious_l = [np.mean(results_extrap[c]['ious']) for c in extrap_lower]
        extrap_ious_l_std = [np.std(results_extrap[c]['ious']) for c in extrap_lower]
        ax.errorbar(extrap_lower, extrap_ious_l, yerr=extrap_ious_l_std,
                    marker='s', label='Extrapolation', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    if extrap_upper:
        extrap_ious_u = [np.mean(results_extrap[c]['ious']) for c in extrap_upper]
        extrap_ious_u_std = [np.std(results_extrap[c]['ious']) for c in extrap_upper]
        ax.errorbar(extrap_upper, extrap_ious_u, yerr=extrap_ious_u_std,
                    marker='s', label='_nolegend_', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.axvspan(train_min, train_max, alpha=0.2, color='gray', label='Training Range')
    ax.set_xlabel('Count')
    ax.set_ylabel('IoU')
    ax.set_title('IoU vs Count', pad=22)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add average metrics text above the axes
    bbox0 = ax.get_position()
    fig.text(bbox0.x0, min(0.995, bbox0.y1 + 0.045),
             f'Avg IoU  |  Interp: {avg_interp_iou:.3f}   Extrap: {avg_extrap_iou:.3f}',
             ha='left', va='bottom', fontsize=10)
    
    # Metric 2: Absolute Count Error
    ax = axes[1]
    if interp_counts:
        interp_errors = [np.mean(results_interp[c]['abs_count_errors']) for c in interp_counts]
        interp_errors_std = [np.std(results_interp[c]['abs_count_errors']) for c in interp_counts]
        ax.errorbar(interp_counts, interp_errors, yerr=interp_errors_std, 
                    marker='o', label='Interpolation', capsize=5, markersize=8)
    
    if extrap_lower:
        extrap_errors_l = [np.mean(results_extrap[c]['abs_count_errors']) for c in extrap_lower]
        extrap_errors_l_std = [np.std(results_extrap[c]['abs_count_errors']) for c in extrap_lower]
        ax.errorbar(extrap_lower, extrap_errors_l, yerr=extrap_errors_l_std,
                    marker='s', label='Extrapolation', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    if extrap_upper:
        extrap_errors_u = [np.mean(results_extrap[c]['abs_count_errors']) for c in extrap_upper]
        extrap_errors_u_std = [np.std(results_extrap[c]['abs_count_errors']) for c in extrap_upper]
        ax.errorbar(extrap_upper, extrap_errors_u, yerr=extrap_errors_u_std,
                    marker='s', label='_nolegend_', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axvspan(train_min, train_max, alpha=0.2, color='gray', label='Training Range')
    ax.set_xlabel('Count')
    ax.set_ylabel('Absolute Count Error')
    ax.set_title('Absolute Count Error vs Count', pad=22)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add average metrics text above the axes
    bbox1 = ax.get_position()
    fig.text(bbox1.x0, min(0.995, bbox1.y1 + 0.045),
             f'Avg |Count Error|  |  Interp: {avg_interp_count_error:.1f}   Extrap: {avg_extrap_count_error:.1f}',
             ha='left', va='bottom', fontsize=10)
    
    # Metric 3: Percentage Count Error
    ax = axes[2]
    if interp_counts:
        interp_pct_errors = [np.mean(results_interp[c]['abs_count_errors_percentage']) for c in interp_counts]
        interp_pct_errors_std = [np.std(results_interp[c]['abs_count_errors_percentage']) for c in interp_counts]
        ax.errorbar(interp_counts, interp_pct_errors, yerr=interp_pct_errors_std, 
                    marker='o', label='Interpolation', capsize=5, markersize=8)
    
    if extrap_lower:
        extrap_pct_errors_l = [np.mean(results_extrap[c]['abs_count_errors_percentage']) for c in extrap_lower]
        extrap_pct_errors_l_std = [np.std(results_extrap[c]['abs_count_errors_percentage']) for c in extrap_lower]
        ax.errorbar(extrap_lower, extrap_pct_errors_l, yerr=extrap_pct_errors_l_std,
                    marker='s', label='Extrapolation', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    if extrap_upper:
        extrap_pct_errors_u = [np.mean(results_extrap[c]['abs_count_errors_percentage']) for c in extrap_upper]
        extrap_pct_errors_u_std = [np.std(results_extrap[c]['abs_count_errors_percentage']) for c in extrap_upper]
        ax.errorbar(extrap_upper, extrap_pct_errors_u, yerr=extrap_pct_errors_u_std,
                    marker='s', label='_nolegend_', capsize=5, markersize=8, linestyle='-', color=extrap_color)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axvspan(train_min, train_max, alpha=0.2, color='gray', label='Training Range')
    ax.set_xlabel('Count')
    ax.set_ylabel('Count Error (%)')
    ax.set_title('Percentage Count Error vs Count', pad=22)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add average metrics text above the axes
    bbox2 = ax.get_position()
    fig.text(bbox2.x0, min(0.995, bbox2.y1 + 0.045),
             f'Avg |% Error|  |  Interp: {avg_interp_pct_error:.1f}%   Extrap: {avg_extrap_pct_error:.1f}%',
             ha='left', va='bottom', fontsize=10)
    
    cfg_str = f" (CFG={config.get('eval_cfg_scale', 1.0)})" if config.get('use_cfg', False) else ""
    plt.suptitle(f"Model: {config['folder_name']}{cfg_str}", fontsize=14, y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.savefig(os.path.join(save_path, 'all_metrics_comparison.png'), dpi=150)
    plt.close()


def save_example_comparisons(results, samples_dict, save_path, num_examples=3):
    """Save visual comparisons of generated vs ground truth."""
    if num_examples is None or num_examples <= 0:
        return
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    for count in sorted(results.keys()):
        fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_examples, len(samples_dict[count]))):
            # Generated image
            axes[i, 0].imshow(samples_dict[count][i])
            axes[i, 0].set_title(f'Generated (count={count})')
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_img = results[count]['gt_images'][i]
            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Masks overlay
            gen_mask = results[count]['gen_masks'][i]
            gt_mask = results[count]['gt_masks'][i]
            
            overlay = np.zeros((*gen_mask.shape, 3))
            overlay[gen_mask == 1] = [0, 1, 0]  # Green for generated
            overlay[gt_mask == 1] = [1, 0, 0]   # Red for ground truth
            overlay[np.logical_and(gen_mask == 1, gt_mask == 1)] = [1, 1, 0]  # Yellow for overlap
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'IoU={results[count]["ious"][i]:.3f}')
            axes[i, 2].axis('off')
            
            # Count info
            est_count = results[count]['estimated_counts'][i]
            err_count = results[count]['count_errors'][i]
            err_pct = results[count]['abs_count_errors_percentage'][i]
            
            axes[i, 3].text(0.1, 0.5, f'Est. Count: {est_count}\nError: {err_count:+.0f}\nError: {err_pct:.1f}%',
                           transform=axes[i, 3].transAxes, fontsize=12, verticalalignment='center')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Count {count} - Examples')
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(save_path, 'comparisons', f'count_{count}_simple.png'), dpi=100, bbox_inches='tight')
        plt.close()


def evaluate_count_set(count_values, samples_dict, config, label="", actual_counts=None):
    """Evaluate a set of count values and return results."""
    results = {}
    
    # If actual_counts is provided, create a mapping from labels to actual counts
    if actual_counts is not None:
        label_to_actual = dict(zip(count_values, actual_counts))
        print(f"Using label-to-actual mapping: {label_to_actual}")
    else:
        label_to_actual = {c: c for c in count_values}  # No bias
    
    for i, count in enumerate(count_values):
        results[count] = {
            'ious': [],
            'precisions': [],
            'recalls': [],
            'estimated_counts': [],
            'gt_counts': [],
            'count_errors': [],
            'abs_count_errors': [],
            'abs_count_errors_percentage': [],
            'gen_masks': [],
            'gt_masks': [],
            'gt_images': []
        }
        
        # Get actual count for ground truth generation
        actual_count = label_to_actual[count]
        
        # Generate ground truth images
        for i, gen_img in enumerate(samples_dict[count]):
            # Generate ground truth using actual count
            gt_img = generate_ground_truth_count_image(
                count=int(actual_count),
                image_size=config['image_size'],
                min_radius=2,
                max_radius=8
            )
            
            # Calculate metrics
            metrics = calculate_count_metrics(gen_img, gt_img, actual_count)
            
            # Store results
            results[count]['ious'].append(metrics['iou'])
            results[count]['precisions'].append(metrics['precision'])
            results[count]['recalls'].append(metrics['recall'])
            results[count]['estimated_counts'].append(metrics['estimated_count'])
            results[count]['gt_counts'].append(actual_count)  # Store the actual count
            results[count]['count_errors'].append(metrics['count_error'])
            results[count]['abs_count_errors'].append(metrics['abs_count_error'])
            results[count]['abs_count_errors_percentage'].append(metrics['abs_count_error_percentage'])
            results[count]['gen_masks'].append(metrics['gen_mask'])
            results[count]['gt_masks'].append(metrics['gt_mask'])
            results[count]['gt_images'].append(gt_img)
    
    return results


def calculate_summary_statistics(results, label=""):
    """Calculate summary statistics for a set of results."""
    summary = {}
    all_ious = []
    all_count_errors = []
    all_abs_count_errors = []
    all_abs_count_errors_percentage = []
    
    for count in results.keys():
        summary[count] = {
            'mean_iou': np.mean(results[count]['ious']),
            'std_iou': np.std(results[count]['ious']),
            'mean_precision': np.mean(results[count]['precisions']),
            'mean_recall': np.mean(results[count]['recalls']),
            'mean_estimated_count': np.mean(results[count]['estimated_counts']),
            'mean_count_error': np.mean(results[count]['count_errors']),
            'std_count_error': np.std(results[count]['count_errors']),
            'mean_abs_count_error': np.mean(results[count]['abs_count_errors']),
            'std_abs_count_error': np.std(results[count]['abs_count_errors']),
            'mean_abs_count_error_percentage': np.mean(results[count]['abs_count_errors_percentage']),
            'std_abs_count_error_percentage': np.std(results[count]['abs_count_errors_percentage'])
        }
        
        all_ious.extend(results[count]['ious'])
        all_count_errors.extend(results[count]['count_errors'])
        all_abs_count_errors.extend(results[count]['abs_count_errors'])
        all_abs_count_errors_percentage.extend(results[count]['abs_count_errors_percentage'])
    
    # Overall statistics
    overall = {
        'mean_iou': np.mean(all_ious),
        'std_iou': np.std(all_ious),
        'median_iou': np.median(all_ious),
        'min_iou': np.min(all_ious),
        'max_iou': np.max(all_ious),
        'mean_count_error': np.mean(all_count_errors),
        'std_count_error': np.std(all_count_errors),
        'mean_abs_count_error': np.mean(all_abs_count_errors),
        'std_abs_count_error': np.std(all_abs_count_errors),
        'mean_abs_count_error_percentage': np.mean(all_abs_count_errors_percentage),
        'std_abs_count_error_percentage': np.std(all_abs_count_errors_percentage)
    }
    
    return summary, overall


def print_results_table(summary_interp, overall_interp, summary_extrap, overall_extrap, cfg_scale=1.0):
    """Print a comprehensive results table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    if cfg_scale > 1.0:
        print(f"CFG Scale: {cfg_scale}")
    print("="*80)
    
    # Interpolation results
    print("\nINTERPOLATION RESULTS:")
    print(f"{'Count':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Count Error':<15} {'% Error':<15}")
    print("-"*70)
    for count in sorted(summary_interp.keys()):
        s = summary_interp[count]
        print(f"{count:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_abs_count_error']:<15.1f} {s['mean_abs_count_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_interp['mean_iou']:<12.4f} {overall_interp['std_iou']:<12.4f} "
          f"{overall_interp['mean_abs_count_error']:<15.1f} {overall_interp['mean_abs_count_error_percentage']:<15.1f}")
    
    # Extrapolation results
    print("\nEXTRAPOLATION RESULTS:")
    print(f"{'Count':<10} {'Mean IoU':<12} {'Std IoU':<12} {'Count Error':<15} {'% Error':<15}")
    print("-"*70)
    for count in sorted(summary_extrap.keys()):
        s = summary_extrap[count]
        print(f"{count:<10.1f} {s['mean_iou']:<12.4f} {s['std_iou']:<12.4f} "
              f"{s['mean_abs_count_error']:<15.1f} {s['mean_abs_count_error_percentage']:<15.1f}")
    
    print("-"*70)
    print(f"Overall:    {overall_extrap['mean_iou']:<12.4f} {overall_extrap['std_iou']:<12.4f} "
          f"{overall_extrap['mean_abs_count_error']:<15.1f} {overall_extrap['mean_abs_count_error_percentage']:<15.1f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'Interpolation':<20} {'Extrapolation':<20}")
    print("-"*70)
    print(f"{'Mean IoU':<30} {overall_interp['mean_iou']:<20.4f} {overall_extrap['mean_iou']:<20.4f}")
    print(f"{'Mean |Count Error|':<30} {overall_interp['mean_abs_count_error']:<20.1f} {overall_extrap['mean_abs_count_error']:<20.1f}")
    print(f"{'Mean |% Error|':<30} {overall_interp['mean_abs_count_error_percentage']:<20.1f} {overall_extrap['mean_abs_count_error_percentage']:<20.1f}")


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
    output_dir = os.path.join("eval_results", f"{config['folder_name']}{cfg_suffix}")
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
    
    # Get evaluation count values
    interp_labels, extrap_labels, interp_counts, extrap_counts = get_evaluation_counts(
        config['train_min_count'], 
        config['train_max_count'],
        config['bias'],
        interp_mode=args.interp_mode
    )
    
    print(f"\nInterpolation labels: {interp_labels}")
    print(f"Extrapolation labels: {extrap_labels}")
    if config['bias'] != 0:
        print(f"Interpolation actual counts: {interp_counts}")
        print(f"Extrapolation actual counts: {extrap_counts}")
    
    # Load model
    model = load_model(config, device)
    # Create diffusion with appropriate settings
    if config.get('architecture') == 'songunet':
        diffusion = create_diffusion(timestep_respacing="", learn_sigma=False)
    else:
        diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))
    
    # Generate samples for all counts (use labels for generation)
    all_labels = interp_labels + extrap_labels
    print(f"\nGenerating samples for all label values...")
    samples_dict = generate_samples(
        model, diffusion, all_labels, config, device, 
        num_samples=args.num_samples_per_count,
        cfg_scale=args.cfg_scale,
        batch_size=args.eval_batch_size
    )
    
    # Evaluate interpolation
    print("\nEvaluating interpolation performance...")
    results_interp = evaluate_count_set(
        interp_labels, 
        {c: samples_dict[c] for c in interp_labels}, 
        config, 
        label="interpolation",
        actual_counts=interp_counts  # Pass actual counts for ground truth
    )
    
    # Evaluate extrapolation
    print("\nEvaluating extrapolation performance...")
    results_extrap = evaluate_count_set(
        extrap_labels, 
        {c: samples_dict[c] for c in extrap_labels}, 
        config, 
        label="extrapolation",
        actual_counts=extrap_counts  # Pass actual counts for ground truth
    )
    
    # Calculate summary statistics
    summary_interp, overall_interp = calculate_summary_statistics(results_interp, "Interpolation")
    summary_extrap, overall_extrap = calculate_summary_statistics(results_extrap, "Extrapolation")
    
    # Print results
    print_results_table(summary_interp, overall_interp, summary_extrap, overall_extrap, cfg_scale=args.cfg_scale)
    
    # Save results
    results_data = {
        'config': config,
        'cfg_scale': args.cfg_scale,
        'interpolation': {
            'counts': interp_labels,
            'summary': summary_interp,
            'overall': overall_interp,
            'detailed_results': {
                str(c): {
                    'ious': results_interp[c]['ious'],
                    'precisions': results_interp[c]['precisions'],
                    'recalls': results_interp[c]['recalls'],
                    'estimated_counts': [int(x) for x in results_interp[c]['estimated_counts']],
                    'count_errors': [int(x) for x in results_interp[c]['count_errors']],
                    'abs_count_errors': [int(x) for x in results_interp[c]['abs_count_errors']],
                    'abs_count_errors_percentage': [float(x) for x in results_interp[c]['abs_count_errors_percentage']]
                }
                for c in interp_labels
            }
        },
        'extrapolation': {
            'counts': extrap_labels,
            'summary': summary_extrap,
            'overall': overall_extrap,
            'detailed_results': {
                str(c): {
                    'ious': results_extrap[c]['ious'],
                    'precisions': results_extrap[c]['precisions'],
                    'recalls': results_extrap[c]['recalls'],
                    'estimated_counts': [int(x) for x in results_extrap[c]['estimated_counts']],
                    'count_errors': [int(x) for x in results_extrap[c]['count_errors']],
                    'abs_count_errors': [int(x) for x in results_extrap[c]['abs_count_errors']],
                    'abs_count_errors_percentage': [float(x) for x in results_extrap[c]['abs_count_errors_percentage']]
                }
                for c in extrap_labels
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
    save_count_comparisons_simple(
        {**results_interp, **results_extrap}, 
        samples_dict, 
        output_dir
    )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate continuous count DiT model with auto-configuration and CFG support')
    
    # Required argument is checkpoint path
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (will infer all settings from path)")
    
    # CFG parameter
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger guidance)")
    
    # Optional sampling parameters
    parser.add_argument("--num-samples-per-count", type=int, default=3,
                        help="Number of samples to generate per count")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of DDPM sampling steps")
    parser.add_argument("--num-visual-examples", type=int, default=1,
                        help="Number of visual examples to save per count")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of samples to generate in parallel during evaluation")
    
    parser.add_argument("--interp-mode", type=str, default="odd", choices=["odd","midpoints"],
                        help="Interpolation counts selection: odd integers or half-step midpoints")

    args = parser.parse_args()
    main(args)
