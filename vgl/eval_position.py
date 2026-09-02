#!/usr/bin/env python
"""
Enhanced evaluation script for continuous position DiT model.
Now includes nearest training point calculation, improved coordinate debugging,
and comprehensive visualizations similar to eval_radius.py.
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
# Inline simple comparison function
def save_position_comparisons_simple(results, samples_dict, save_path):
    """Save simple GT vs Generated for position."""
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    for pos in sorted(results.keys()):
        fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.25))
        
        # Create ground truth - use smaller circle (radius 4) to match generated appearance
        img_size = 64
        gt_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        img_center = img_size // 2
        center_x = int(img_center + pos[0])
        center_y = int(img_center + pos[1])
        # Use same color as generated for consistency
        cv2.circle(gt_img, (center_x, center_y), 4, (150, 150, 150), -1)
        
        axes[0].imshow(gt_img)
        axes[0].axis('off')
        axes[0].set_title('GT', fontsize=7)
        
        # Generated
        axes[1].imshow(samples_dict[pos][0])
        axes[1].axis('off')
        axes[1].set_title('Gen', fontsize=7)
        
        plt.tight_layout(pad=0.1)
        
        pos_str = f"{pos[0]:.1f}_{pos[1]:.1f}".replace('.', 'p').replace('-', 'neg')
        plt.savefig(os.path.join(save_path, 'comparisons', f'position_{pos_str}_simple.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()
from scipy import ndimage
from scipy.spatial.distance import cdist

# Import model components
from vgl.models_position import DiT_models_position as DiT_models
from vgl.unet_models_position import UNet_models_position as UNet_models
from vgl.unet_models_song_position import SongUNet_Position_models as SongUNet_models
from vgl.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from vgl.flow_matching import FlowMatching


def generate_training_positions(square_size, density):
    """
    Generate the training position grid based on square size and density.
    This should match exactly how positions were generated during dataset creation.
    """
    half_size = square_size / 2
    coords = np.linspace(-half_size, half_size, density)
    
    positions = []
    for x in coords:
        for y in coords:
            positions.append((x, y))
    
    return positions


def find_nearest_training_position(test_pos, training_positions):
    """
    Find the nearest training position to a test position.
    Returns the nearest position and the distance.
    """
    test_pos_array = np.array([test_pos])
    train_pos_array = np.array(training_positions)
    
    distances = cdist(test_pos_array, train_pos_array)[0]
    nearest_idx = np.argmin(distances)
    
    return training_positions[nearest_idx], distances[nearest_idx]


def parse_checkpoint_path(ckpt_path):
    """
    Parse model configuration from checkpoint path.
    Expected folder format: circle_model_position_36_8x8_linear_concat_no_cfg_4workers
    """
    # Get the folder name from the path - handle both old and new folder structures
    path_parts = Path(ckpt_path).parts
    folder_name = None
    
    # First try to find 'circle_model_position' (old structure)
    for part in path_parts:
        if 'circle_model_position' in part.lower():
            folder_name = part
            break
    
    # If not found, look for ablation folder structure (new structure)
    if not folder_name:
        for part in path_parts:
            if 'position_' in part:
                folder_name = part
                break
    
    if not folder_name:
        raise ValueError(f"Could not find model folder name in path: {ckpt_path}")
    
    print(f"Parsing configuration from folder: {folder_name}")
    
    config = {
        'folder_name': folder_name,
        'ckpt_path': ckpt_path
    }
    
    # Detect architecture and model type from path
    # Look for patterns like "DiT-S", "DiT-B", "DiT-L", "DiT-XL", "UNet-S", etc. in the entire path
    full_path_str = str(ckpt_path).lower()
    
    # Check architecture (check for songunet BEFORE checking for unet, since songunet contains "unet")
    if 'songunet' in full_path_str:
        config['architecture'] = 'songunet'
        config['model_type'] = 'SongUNet-Position-B'  # Will be overridden later if needed (default to -B for NEW models)
    elif 'unet' in full_path_str:
        config['architecture'] = 'unet'
        # Parse UNet model type
        if 'unet-xl' in full_path_str:
            config['model_type'] = 'UNet-XL'
        elif 'unet-l' in full_path_str:
            config['model_type'] = 'UNet-L'
        elif 'unet-b' in full_path_str:
            config['model_type'] = 'UNet-B'
        elif 'unet-s' in full_path_str:
            config['model_type'] = 'UNet-S'
        else:
            print("Warning: Could not detect UNet model type from path, defaulting to UNet-S")
            config['model_type'] = 'UNet-S'
    else:
        config['architecture'] = 'dit'
        # Parse DiT model type
        if 'dit-xl' in full_path_str:
            config['model_type'] = 'DiT-XL/2'
        elif 'dit-l' in full_path_str:
            config['model_type'] = 'DiT-L/2'
        elif 'dit-b' in full_path_str:
            config['model_type'] = 'DiT-B/2'
        elif 'dit-s' in full_path_str:
            config['model_type'] = 'DiT-S/2'
        else:
            # Default to DiT-S/2 if not found
            print("Warning: Could not detect model type from path, defaulting to DiT-S/2")
            config['model_type'] = 'DiT-S/2'
    
    print(f"Detected architecture: {config['architecture']}, model type: {config['model_type']}")
    
    # Parse from pattern: circle_model_position_36_8x8_linear_concat_no_cfg_4workers
    parts = folder_name.split('_')
    
    # Find the square size (first number after "position")
    square_size = None
    density = None
    for i, part in enumerate(parts):
        if part == 'position' and i + 1 < len(parts):
            try:
                square_size = int(parts[i + 1])
                # Next part should be density (e.g., "8x8")
                if i + 2 < len(parts) and 'x' in parts[i + 2]:
                    density_match = re.match(r'(\d+)x(\d+)', parts[i + 2])
                    if density_match:
                        density = int(density_match.group(1))
            except ValueError:
                pass
    
    if square_size is None:
        square_size = 36  # default from examples
    
    # Set training range based on square size
    config['square_size'] = square_size
    config['train_min_pos'] = -square_size / 2
    config['train_max_pos'] = square_size / 2
    config['density'] = density if density else 5  # default density
    
    config['image_size'] = 64  # default
    
    # Parse position embedding type
    if 'rotary' in folder_name.lower():
        config['position_embedding_type'] = 'rotary'
    elif 'linear' in folder_name.lower():
        config['position_embedding_type'] = 'linear'
    elif 'sinusoidal' in folder_name.lower() or "sinu" in folder_name.lower():
        config['position_embedding_type'] = 'sinusoidal'
    else:
        config['position_embedding_type'] = 'linear'  # default
    
    # Parse conditioning method
    if 'concat' in folder_name.lower():
        config['conditioning_method'] = 'concat'
    elif 'adaln' in folder_name.lower():
        config['conditioning_method'] = 'adaln'
    else:
        config['conditioning_method'] = 'concat'  # default for position
    
    # Parse CFG settings
    if 'no_cfg' in folder_name.lower() or 'cfg' not in folder_name.lower():
        config['position_dropout_prob'] = 0.0
        config['use_cfg'] = False
        config['null_embedding_type'] = 'learnable'
    else:
        config['position_dropout_prob'] = 0.1
        config['use_cfg'] = True
        if 'zero' in folder_name.lower():
            config['null_embedding_type'] = 'zero'
        elif 'learnable' in folder_name.lower():
            config['null_embedding_type'] = 'learnable'
        else:
            config['null_embedding_type'] = 'learnable'
    
    # Parse flow matching
    if 'flow' in folder_name.lower():
        config['use_flow_matching'] = True
        print(f"Detected flow matching model")
    else:
        config['use_flow_matching'] = False
    
    # Parse architecture
    if 'songunet' in folder_name.lower():
        config['architecture'] = 'songunet'
        # Default to Position-B for NEW models (they use 256 channels)
        config['model_name'] = 'SongUNet-Position-B'  # Correct model key for SongUNet Position NEW
        # NEW SongUNet models use sinusoidal embedding and adaln conditioning
        # Override generic parsing unless explicitly specified in folder name
        if 'linear' not in folder_name.lower() and 'sinusoidal' not in folder_name.lower() and 'rotary' not in folder_name.lower():
            config['position_embedding_type'] = 'sinusoidal'
        if 'adaln' not in folder_name.lower() and 'concat' not in folder_name.lower():
            config['conditioning_method'] = 'adaln'
    else:
        config['architecture'] = 'dit'  # default
    
    # Set other defaults
    config['null_position'] = (0.0, 0.0)
    
    # Calculate valid range for extrapolation
    config['circle_radius'] = 8  # from dataset generation
    config['valid_min'] = -config['image_size'] / 2 + config['circle_radius']
    config['valid_max'] = config['image_size'] / 2 - config['circle_radius']
    
    return config


def get_evaluation_positions(train_min, train_max, square_size, valid_min, valid_max, density=5):
    """
    Get exact training, interpolation, and extrapolation position values.
    """
    # Training positions are on a grid within the square
    train_coords = np.linspace(train_min, train_max, density)
    train_positions = [(x, y) for x in train_coords for y in train_coords]
    
    # Select a subset of training positions for evaluation
    # Choose strategic positions: corners, center, edges, and some random ones
    exact_train_positions = []
    
    # Add corners
    exact_train_positions.extend([
        (train_coords[0], train_coords[0]),      # Top-left
        (train_coords[-1], train_coords[-1]),    # Bottom-right
        (train_coords[0], train_coords[-1]),     # Top-right
        (train_coords[-1], train_coords[0]),     # Bottom-left
    ])
    
    # Add center if it exists
    if density % 2 == 1:
        center_idx = density // 2
        exact_train_positions.append((train_coords[center_idx], train_coords[center_idx]))
    
    # Add middle of edges
    mid_idx = density // 2
    exact_train_positions.extend([
        (train_coords[0], train_coords[mid_idx]),    # Left edge middle
        (train_coords[-1], train_coords[mid_idx]),   # Right edge middle
        (train_coords[mid_idx], train_coords[0]),    # Top edge middle
        (train_coords[mid_idx], train_coords[-1]),   # Bottom edge middle
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_exact = []
    for pos in exact_train_positions:
        if pos not in seen and pos in train_positions:
            seen.add(pos)
            unique_exact.append(pos)
    
    # Limit to reasonable number
    unique_exact = unique_exact[:10]
    
    # Interpolation: positions within training square but not in training set
    interp_coords = []
    for i in range(len(train_coords) - 1):
        mid = (train_coords[i] + train_coords[i+1]) / 2
        interp_coords.append(mid)
    
    interpolation_positions = []
    if len(interp_coords) >= 2:
        interpolation_positions.extend([
            (interp_coords[0], interp_coords[0]),
            (interp_coords[-1], interp_coords[-1]),
            (0.0, interp_coords[0]),
            (interp_coords[0], 0.0),
            (interp_coords[1], interp_coords[1]) if len(interp_coords) > 1 else (interp_coords[0], -interp_coords[0])
        ])
    
    interpolation_positions = [(x, y) for (x, y) in interpolation_positions 
                              if (x, y) not in train_positions]
    
    # Extrapolation positions
    extrapolation_positions = []
    
    margin = 2.0
    extrap_near = train_max + margin
    extrap_far = min(valid_max, train_max + 8.0)
    
    if extrap_near <= valid_max:
        extrapolation_positions.extend([
            (extrap_near, 0.0),
            (0.0, extrap_near),
            (extrap_near, extrap_near),
            (-extrap_near, 0.0),
            (0.0, -extrap_near),
            (-extrap_near, -extrap_near),
        ])
    
    if extrap_far <= valid_max:
        extrapolation_positions.extend([
            (extrap_far, 0.0),
            (0.0, extrap_far),
            (extrap_far, extrap_far),
            (-extrap_far, -extrap_far),
        ])
    
    if extrap_near <= valid_max:
        extrapolation_positions.extend([
            (train_min / 2, extrap_near),
            (extrap_near, train_min / 2),
        ])
    
    extrapolation_positions = [(x, y) for (x, y) in extrapolation_positions 
                              if valid_min <= x <= valid_max and valid_min <= y <= valid_max]
    
    # Remove duplicates for interpolation and extrapolation
    seen = set()
    unique_interp = []
    for pos in interpolation_positions:
        if pos not in seen:
            seen.add(pos)
            unique_interp.append(pos)
    
    seen = set()
    unique_extrap = []
    for pos in extrapolation_positions:
        if pos not in seen:
            seen.add(pos)
            unique_extrap.append(pos)
    
    return unique_exact, unique_interp, unique_extrap, train_positions


def extract_circle_center_debug(image, expected_radius=8, debug=False):
    """
    Extract the center position of a circle in the image with optional debugging.
    Returns (x, y) coordinates relative to image center and debug info.
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
    
    # Apply thresholding to get binary mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (ensure circle is white)
    if np.mean(mask) > 127:
        mask = 255 - mask
    
    # Find center of mass
    mask_binary = (mask > 127).astype(np.float32)
    
    debug_info = {
        'mask_pixels': np.sum(mask_binary),
        'mask': mask_binary if debug else None
    }
    
    if np.sum(mask_binary) > 10:  # Minimum pixels to consider valid
        # Calculate center of mass
        # Note: ndimage returns (row, col) which is (y, x) in image coordinates
        cy, cx = ndimage.center_of_mass(mask_binary)
        
        # Convert to coordinates relative to image center
        img_center_x = img_array.shape[1] / 2  # width / 2
        img_center_y = img_array.shape[0] / 2  # height / 2
        
        # x_rel: positive = right of center, negative = left of center
        # y_rel: positive = below center, negative = above center
        x_rel = cx - img_center_x
        y_rel = cy - img_center_y
        
        debug_info.update({
            'cx_pixel': cx,
            'cy_pixel': cy,
            'img_center_x': img_center_x,
            'img_center_y': img_center_y,
            'x_rel': x_rel,
            'y_rel': y_rel
        })
        
        return x_rel, y_rel, True, debug_info
    else:
        return 0.0, 0.0, False, debug_info


def calculate_position_metrics_enhanced(generated_img, expected_pos, training_positions, image_size=64, debug=False):
    """
    Calculate position-specific metrics with nearest training point info.
    """
    # Extract circle center from generated image
    detected_x, detected_y, valid, debug_info = extract_circle_center_debug(generated_img, debug=debug)
    
    # Find nearest training position
    nearest_train_pos, distance_to_nearest = find_nearest_training_position(expected_pos, training_positions)
    
    if not valid:
        return {
            'position_error': float('inf'),
            'position_error_x': float('inf'),
            'position_error_y': float('inf'),
            'relative_error': float('inf'),
            'detection_success': False,
            'detected_position': (0.0, 0.0),
            'expected_position': expected_pos,
            'nearest_train_position': nearest_train_pos,
            'distance_to_nearest_train': distance_to_nearest,
            'debug_info': debug_info
        }
    
    # Calculate errors
    expected_x, expected_y = expected_pos
    error_x = detected_x - expected_x
    error_y = detected_y - expected_y
    position_error = np.sqrt(error_x**2 + error_y**2)
    
    # Calculate relative error
    max_distance = np.sqrt(2) * (image_size / 2)
    relative_error = (position_error / max_distance) * 100
    
    return {
        'position_error': position_error,
        'position_error_x': error_x,
        'position_error_y': error_y,
        'relative_error': relative_error,
        'detection_success': True,
        'detected_position': (detected_x, detected_y),
        'expected_position': expected_pos,
        'nearest_train_position': nearest_train_pos,
        'distance_to_nearest_train': distance_to_nearest,
        'debug_info': debug_info
    }


def visualize_results(results_exact, results_interp, results_extrap, save_path, config):
    """Create comprehensive visualizations including 2D position plots and distance analysis."""
    
    # 1. Create main 3-metric comparison plot
    create_main_metrics_plot(results_exact, results_interp, results_extrap, save_path, config)
    
    # 2. Create 2D position scatter plots
    create_2d_position_plots(results_exact, results_interp, results_extrap, save_path, config)
    
    # 3. Create distance-from-training analysis
    create_distance_analysis_plots(results_exact, results_interp, results_extrap, save_path, config)
    
    # 4. Create 2D heatmaps of performance across space
    create_2d_performance_heatmaps(results_exact, results_interp, results_extrap, save_path, config)
    
    # 5. Create detailed metric heatmaps
    create_metric_heatmaps(results_exact, results_interp, results_extrap, save_path)


def create_main_metrics_plot(results_exact, results_interp, results_extrap, save_path, config):
    """Create the main 3-subplot metrics comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Separate by category
    exact_positions = sorted(results_exact.keys()) if results_exact else []
    interp_positions = sorted(results_interp.keys()) if results_interp else []
    extrap_positions = sorted(results_extrap.keys()) if results_extrap else []
    
    # Helper function to extract x-coordinates for plotting
    def get_x_coords(positions):
        return [pos[0] for pos in positions]
    
    # Metric 1: Position Error vs Distance from Center
    ax = axes[0]
    
    def get_distances_from_center(positions):
        return [np.sqrt(pos[0]**2 + pos[1]**2) for pos in positions]
    
    if exact_positions:
        exact_dist = get_distances_from_center(exact_positions)
        exact_errors = []
        exact_errors_std = []
        for pos in exact_positions:
            valid_errors = [e for e, success in zip(results_exact[pos]['position_errors'], 
                                                   results_exact[pos]['detection_successes']) if success]
            exact_errors.append(np.mean(valid_errors) if valid_errors else 0)
            exact_errors_std.append(np.std(valid_errors) if valid_errors else 0)
        
        ax.errorbar(exact_dist, exact_errors, yerr=exact_errors_std, 
                    marker='o', label='Exact Training', capsize=5, markersize=8, alpha=0.8)
    
    if interp_positions:
        interp_dist = get_distances_from_center(interp_positions)
        interp_errors = []
        interp_errors_std = []
        for pos in interp_positions:
            valid_errors = [e for e, success in zip(results_interp[pos]['position_errors'], 
                                                   results_interp[pos]['detection_successes']) if success]
            interp_errors.append(np.mean(valid_errors) if valid_errors else 0)
            interp_errors_std.append(np.std(valid_errors) if valid_errors else 0)
            
        ax.errorbar(interp_dist, interp_errors, yerr=interp_errors_std, 
                    marker='s', label='Interpolation', capsize=5, markersize=8, alpha=0.8)
    
    if extrap_positions:
        extrap_dist = get_distances_from_center(extrap_positions)
        extrap_errors = []
        extrap_errors_std = []
        for pos in extrap_positions:
            valid_errors = [e for e, success in zip(results_extrap[pos]['position_errors'], 
                                                   results_extrap[pos]['detection_successes']) if success]
            extrap_errors.append(np.mean(valid_errors) if valid_errors else 0)
            extrap_errors_std.append(np.std(valid_errors) if valid_errors else 0)
            
        ax.errorbar(extrap_dist, extrap_errors, yerr=extrap_errors_std, 
                    marker='^', label='Extrapolation', capsize=5, markersize=8, alpha=0.8, linestyle='None')
    
    ax.set_xlabel('Distance from Center (pixels)')
    ax.set_ylabel('Position Error (pixels)')
    ax.set_title('Position Error vs Distance from Center')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Metric 2: Relative Error vs Distance from Center
    ax = axes[1]
    
    if exact_positions:
        exact_rel_errors = []
        exact_rel_errors_std = []
        for pos in exact_positions:
            valid_errors = [e for e, success in zip(results_exact[pos]['relative_errors'], 
                                                   results_exact[pos]['detection_successes']) if success]
            exact_rel_errors.append(np.mean(valid_errors) if valid_errors else 0)
            exact_rel_errors_std.append(np.std(valid_errors) if valid_errors else 0)
            
        ax.errorbar(exact_dist, exact_rel_errors, yerr=exact_rel_errors_std, 
                    marker='o', label='Exact Training', capsize=5, markersize=8, alpha=0.8)
    
    if interp_positions:
        interp_rel_errors = []
        interp_rel_errors_std = []
        for pos in interp_positions:
            valid_errors = [e for e, success in zip(results_interp[pos]['relative_errors'], 
                                                   results_interp[pos]['detection_successes']) if success]
            interp_rel_errors.append(np.mean(valid_errors) if valid_errors else 0)
            interp_rel_errors_std.append(np.std(valid_errors) if valid_errors else 0)
            
        ax.errorbar(interp_dist, interp_rel_errors, yerr=interp_rel_errors_std, 
                    marker='s', label='Interpolation', capsize=5, markersize=8, alpha=0.8)
    
    if extrap_positions:
        extrap_rel_errors = []
        extrap_rel_errors_std = []
        for pos in extrap_positions:
            valid_errors = [e for e, success in zip(results_extrap[pos]['relative_errors'], 
                                                   results_extrap[pos]['detection_successes']) if success]
            extrap_rel_errors.append(np.mean(valid_errors) if valid_errors else 0)
            extrap_rel_errors_std.append(np.std(valid_errors) if valid_errors else 0)
            
        ax.errorbar(extrap_dist, extrap_rel_errors, yerr=extrap_rel_errors_std, 
                    marker='^', label='Extrapolation', capsize=5, markersize=8, alpha=0.8, linestyle='None')
    
    ax.set_xlabel('Distance from Center (pixels)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Relative Error vs Distance from Center')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Metric 3: Detection Rate vs Distance from Center
    ax = axes[2]
    
    if exact_positions:
        exact_det_rates = [np.mean(results_exact[pos]['detection_successes']) * 100 
                          for pos in exact_positions]
        ax.scatter(exact_dist, exact_det_rates, marker='o', label='Exact Training', s=80, alpha=0.8)
    
    if interp_positions:
        interp_det_rates = [np.mean(results_interp[pos]['detection_successes']) * 100 
                           for pos in interp_positions]
        ax.scatter(interp_dist, interp_det_rates, marker='s', label='Interpolation', s=80, alpha=0.8)
    
    if extrap_positions:
        extrap_det_rates = [np.mean(results_extrap[pos]['detection_successes']) * 100 
                           for pos in extrap_positions]
        ax.scatter(extrap_dist, extrap_det_rates, marker='^', label='Extrapolation', s=80, alpha=0.8)
    
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Perfect Detection')
    ax.set_xlabel('Distance from Center (pixels)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate vs Distance from Center')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 105)
    
    cfg_str = f" (CFG={config.get('eval_cfg_scale', 1.0)})" if config.get('use_cfg', False) else ""
    plt.suptitle(f"Model: {config['folder_name']}{cfg_str}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_metrics_comparison.png'), dpi=150)
    plt.close()


def create_2d_position_plots(results_exact, results_interp, results_extrap, save_path, config):
    """Create 2D scatter plots showing position performance across the 2D space."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Combine all results for plotting
    all_categories = [
        (results_exact, 'Exact Training', 'o'),
        (results_interp, 'Interpolation', 's'),
        (results_extrap, 'Extrapolation', '^')
    ]
    
    metrics = [
        ('position_errors', 'Position Error (pixels)', 'viridis_r'),
        ('relative_errors', 'Relative Error (%)', 'plasma_r'),
        ('detection_successes', 'Detection Rate', 'RdYlGn')
    ]
    
    for i, (metric_key, metric_name, cmap) in enumerate(metrics):
        ax = axes[i]
        
        # Plot training region boundary
        train_square = plt.Rectangle(
            (config['train_min_pos'], config['train_min_pos']),
            config['train_max_pos'] - config['train_min_pos'],
            config['train_max_pos'] - config['train_min_pos'],
            fill=False, edgecolor='gray', linewidth=2, alpha=0.5, label='Training Region'
        )
        ax.add_patch(train_square)
        
        # Collect all data for consistent color scaling
        all_values = []
        for results, _, _ in all_categories:
            for pos, data in results.items():
                if metric_key == 'detection_successes':
                    value = np.mean(data[metric_key])
                else:
                    valid_values = [v for v, success in zip(data[metric_key], data['detection_successes']) if success]
                    value = np.mean(valid_values) if valid_values else 0
                all_values.append(value)
        
        if all_values:
            vmin, vmax = min(all_values), max(all_values)
            if metric_key == 'detection_successes':
                vmin, vmax = 0, 1
        else:
            vmin, vmax = 0, 1
        
        # Plot each category
        for results, label, marker in all_categories:
            if not results:
                continue
                
            x_coords = [pos[0] for pos in results.keys()]
            y_coords = [pos[1] for pos in results.keys()]
            values = []
            
            for pos, data in results.items():
                if metric_key == 'detection_successes':
                    value = np.mean(data[metric_key])
                else:
                    valid_values = [v for v, success in zip(data[metric_key], data['detection_successes']) if success]
                    value = np.mean(valid_values) if valid_values else 0
                values.append(value)
            
            scatter = ax.scatter(x_coords, y_coords, c=values, marker=marker, 
                               s=100, alpha=0.8, cmap=cmap, vmin=vmin, vmax=vmax, 
                               edgecolor='black', linewidth=0.5, label=label)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(metric_name)
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'{metric_name} - 2D View')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.suptitle(f"2D Position Performance Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '2d_position_analysis.png'), dpi=150)
    plt.close()


def create_distance_analysis_plots(results_exact, results_interp, results_extrap, save_path, config):
    """Create plots analyzing performance vs distance from nearest training position."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    all_categories = [
        (results_exact, 'Exact Training', 'o'),
        (results_interp, 'Interpolation', 's'),
        (results_extrap, 'Extrapolation', '^')
    ]
    
    metrics = [
        ('position_errors', 'Position Error (pixels)'),
        ('relative_errors', 'Relative Error (%)'),
        ('detection_successes', 'Detection Rate (%)')
    ]
    
    for i, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[i]
        
        for results, label, marker in all_categories:
            if not results:
                continue
                
            distances = []
            values = []
            
            for pos, data in results.items():
                # Get distance to nearest training position
                dist_to_train = np.mean(data['distances_to_nearest_train'])
                distances.append(dist_to_train)
                
                if metric_key == 'detection_successes':
                    value = np.mean(data[metric_key]) * 100  # Convert to percentage
                else:
                    valid_values = [v for v, success in zip(data[metric_key], data['detection_successes']) if success]
                    value = np.mean(valid_values) if valid_values else 0
                values.append(value)
            
            ax.scatter(distances, values, marker=marker, s=80, alpha=0.8, label=label)
        
        ax.set_xlabel('Distance to Nearest Training Position (pixels)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Training Distance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if metric_key == 'detection_successes':
            ax.set_ylim(0, 105)
    
    plt.suptitle('Performance vs Distance from Training Positions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'distance_analysis.png'), dpi=150)
    plt.close()


def create_2d_performance_heatmaps(results_exact, results_interp, results_extrap, save_path, config):
    """Create 2D heatmaps showing performance across the position space."""
    # Combine all results
    all_results = {**results_exact, **results_interp, **results_extrap}
    
    if not all_results:
        return
    
    # Create a grid for interpolation
    positions = list(all_results.keys())
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Create metrics data
    metrics_data = {
        'Position Error': [],
        'Relative Error': [], 
        'Detection Rate': []
    }
    
    for pos in positions:
        data = all_results[pos]
        
        # Position Error
        valid_errors = [e for e, success in zip(data['position_errors'], data['detection_successes']) if success]
        metrics_data['Position Error'].append(np.mean(valid_errors) if valid_errors else np.nan)
        
        # Relative Error
        valid_rel_errors = [e for e, success in zip(data['relative_errors'], data['detection_successes']) if success]
        metrics_data['Relative Error'].append(np.mean(valid_rel_errors) if valid_rel_errors else np.nan)
        
        # Detection Rate
        metrics_data['Detection Rate'].append(np.mean(data['detection_successes']))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        # Create scatter plot with color coding
        if metric_name == 'Detection Rate':
            cmap = 'RdYlGn'
            vmin, vmax = 0, 1
        else:
            cmap = 'viridis_r'
            vmin, vmax = None, None
        
        scatter = ax.scatter(x_coords, y_coords, c=values, s=100, cmap=cmap, 
                           vmin=vmin, vmax=vmax, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add training region boundary
        train_square = plt.Rectangle(
            (config['train_min_pos'], config['train_min_pos']),
            config['train_max_pos'] - config['train_min_pos'],
            config['train_max_pos'] - config['train_min_pos'],
            fill=False, edgecolor='red', linewidth=2, alpha=0.8, label='Training Region'
        )
        ax.add_patch(train_square)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        if metric_name == 'Detection Rate':
            cbar.set_label('Detection Rate')
        elif metric_name == 'Position Error':
            cbar.set_label('Position Error (pixels)')
        else:
            cbar.set_label('Relative Error (%)')
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'{metric_name} Heatmap')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.suptitle('2D Performance Heatmaps', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '2d_performance_heatmaps.png'), dpi=150)
    plt.close()


def create_metric_heatmaps(results_exact, results_interp, results_extrap, save_path):
    """Create detailed heatmaps for each metric (similar to eval_radius.py)."""
    metrics = ['position_errors', 'relative_errors', 'detection_successes']
    metric_names = ['Position Error (pixels)', 'Relative Error (%)', 'Detection Success']
    
    for metric, metric_name in zip(metrics, metric_names):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Helper function to create heatmap for a result set
        def create_heatmap(results, ax, title):
            if not results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
                
            positions = sorted(results.keys())
            max_samples = max(len(results[pos][metric]) for pos in positions)
            matrix = np.full((len(positions), max_samples), np.nan)
            
            for i, pos in enumerate(positions):
                values = results[pos][metric]
                if metric == 'detection_successes':
                    # Convert boolean to float for detection success
                    values = [float(v) for v in values]
                matrix[i, :len(values)] = values
            
            # Create position labels
            pos_labels = [f"({pos[0]:.1f},{pos[1]:.1f})" for pos in positions]
            
            if metric == 'detection_successes':
                cmap = 'RdYlGn'
                vmin, vmax = 0, 1
            elif metric == 'relative_errors':
                cmap = 'RdBu_r'
                vmin, vmax = None, None
            else:  # position_errors
                cmap = 'RdBu_r'
                vmin, vmax = None, None
            
            sns.heatmap(matrix, ax=ax, xticklabels=False, yticklabels=pos_labels,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        cbar_kws={'label': metric_name})
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Position')
            ax.set_title(title)
        
        # Create heatmaps for each category
        create_heatmap(results_exact, axes[0], f'{metric_name} - Exact Training')
        create_heatmap(results_interp, axes[1], f'{metric_name} - Interpolation')
        create_heatmap(results_extrap, axes[2], f'{metric_name} - Extrapolation')
        
        plt.suptitle(f'{metric_name} Heatmaps', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{metric}_heatmap.png'), dpi=150)
        plt.close()


def save_example_comparisons_enhanced(results, samples_dict, save_path, training_positions, num_examples=3):
    """
    Save visual comparisons with enhanced debugging information.
    """
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    for pos in sorted(results.keys()):
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, 3*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_examples, len(samples_dict[pos]))):
            # Generated image
            img = samples_dict[pos][i]
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Generated\nTarget: ({pos[0]:.1f}, {pos[1]:.1f})')
            axes[i, 0].axis('off')
            
            # Generated image with markers
            img_with_markers = img.copy()
            img_center = img.shape[0] // 2
            
            # Draw expected position (green cross)
            expected_x_px = int(img_center + pos[0])
            expected_y_px = int(img_center + pos[1])
            cv2.drawMarker(img_with_markers, (expected_x_px, expected_y_px), 
                          (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
            
            # Draw detected position (red cross) if valid
            if results[pos]['detection_successes'][i]:
                detected_x, detected_y = results[pos]['detected_positions'][i]
                detected_x_px = int(img_center + detected_x)
                detected_y_px = int(img_center + detected_y)
                cv2.drawMarker(img_with_markers, (detected_x_px, detected_y_px), 
                              (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            
            axes[i, 1].imshow(img_with_markers)
            axes[i, 1].set_title('Green=Expected, Red=Detected')
            axes[i, 1].axis('off')
            
            # Debug visualization showing mask
            if 'debug_info' in results[pos] and i < len(results[pos]['debug_info']):
                debug_info = results[pos]['debug_info'][i]
                if debug_info and 'mask' in debug_info and debug_info['mask'] is not None:
                    axes[i, 2].imshow(debug_info['mask'], cmap='gray')
                    axes[i, 2].set_title(f'Detected Mask\n({debug_info["mask_pixels"]} pixels)')
                else:
                    axes[i, 2].text(0.5, 0.5, 'No mask', ha='center', va='center', 
                                   transform=axes[i, 2].transAxes)
                    axes[i, 2].set_title('Detection Failed')
            else:
                axes[i, 2].text(0.5, 0.5, 'No debug info', ha='center', va='center', 
                               transform=axes[i, 2].transAxes)
                axes[i, 2].set_title('No Debug Info')
            axes[i, 2].axis('off')
            
            # Detailed metrics with nearest training point
            error = results[pos]['position_errors'][i]
            rel_error = results[pos]['relative_errors'][i]
            nearest_train = results[pos]['nearest_train_positions'][i]
            dist_to_train = results[pos]['distances_to_nearest_train'][i]
            
            if results[pos]['detection_successes'][i]:
                detected_x, detected_y = results[pos]['detected_positions'][i]
                text = (f'Target: ({pos[0]:.1f}, {pos[1]:.1f})\n'
                       f'Detected: ({detected_x:.1f}, {detected_y:.1f})\n'
                       f'Position Error: {error:.2f}px\n'
                       f'Relative Error: {rel_error:.1f}%\n'
                       f'X Error: {results[pos]["position_errors_x"][i]:.2f}px\n'
                       f'Y Error: {results[pos]["position_errors_y"][i]:.2f}px\n\n'
                       f'Nearest Train: ({nearest_train[0]:.1f}, {nearest_train[1]:.1f})\n'
                       f'Dist to Train: {dist_to_train:.2f}px')
            else:
                text = (f'Target: ({pos[0]:.1f}, {pos[1]:.1f})\n'
                       f'Detection Failed\n\n'
                       f'Nearest Train: ({nearest_train[0]:.1f}, {nearest_train[1]:.1f})\n'
                       f'Dist to Train: {dist_to_train:.2f}px')
            
            axes[i, 3].text(0.05, 0.95, text, transform=axes[i, 3].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Position ({pos[0]:.1f}, {pos[1]:.1f}) - Detailed Analysis')
        plt.tight_layout()
        pos_str = f"{pos[0]:.1f}_{pos[1]:.1f}".replace('.', 'p').replace('-', 'neg')
        plt.savefig(os.path.join(save_path, 'comparisons', f'position_{pos_str}_simple.png'), dpi=100, bbox_inches='tight')
        plt.close()


def evaluate_position_set_enhanced(position_values, samples_dict, config, training_positions, label="", debug_first=True):
    """
    Evaluate a set of position values with enhanced metrics.
    """
    results = {}
    
    for pos_idx, pos in enumerate(tqdm(position_values, desc=f"Evaluating {label}")):
        results[pos] = {
            'position_errors': [],
            'position_errors_x': [],
            'position_errors_y': [],
            'relative_errors': [],
            'detection_successes': [],
            'detected_positions': [],
            'expected_positions': [],
            'nearest_train_positions': [],
            'distances_to_nearest_train': [],
            'debug_info': []
        }
        
        # Evaluate each sample
        for i, gen_img in enumerate(samples_dict[pos]):
            # Enable debug for first position's first sample
            debug = debug_first and pos_idx == 0 and i == 0
            
            # Calculate metrics
            metrics = calculate_position_metrics_enhanced(
                gen_img, pos, training_positions, config['image_size'], debug=debug
            )
            
            # Store results
            results[pos]['position_errors'].append(metrics['position_error'])
            results[pos]['position_errors_x'].append(metrics['position_error_x'])
            results[pos]['position_errors_y'].append(metrics['position_error_y'])
            results[pos]['relative_errors'].append(metrics['relative_error'])
            results[pos]['detection_successes'].append(metrics['detection_success'])
            results[pos]['detected_positions'].append(metrics['detected_position'])
            results[pos]['expected_positions'].append(metrics['expected_position'])
            results[pos]['nearest_train_positions'].append(metrics['nearest_train_position'])
            results[pos]['distances_to_nearest_train'].append(metrics['distance_to_nearest_train'])
            results[pos]['debug_info'].append(metrics.get('debug_info', {}))
    
    return results


def print_results_table_enhanced(summary_exact, overall_exact, summary_interp, overall_interp, 
                                summary_extrap, overall_extrap, cfg_scale=1.0):
    """
    Print enhanced results table with nearest training point info.
    """
    print("\n" + "="*110)
    print("POSITION EVALUATION RESULTS")
    if cfg_scale > 1.0:
        print(f"CFG Scale: {cfg_scale}")
    print("="*110)
    
    # Exact training results
    if summary_exact:
        print("\nEXACT TRAINING POSITION RESULTS:")
        print(f"{'Position':<20} {'Nearest Train':<20} {'Train Dist':<12} {'Mean Err':<12} {'Rel Err%':<12} {'Det Rate%':<12}")
        print("-"*105)
        for pos in sorted(summary_exact.keys()):
            s = summary_exact[pos]
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            nearest_str = f"({s['nearest_train_pos'][0]:.1f}, {s['nearest_train_pos'][1]:.1f})"
            print(f"{pos_str:<20} {nearest_str:<20} {s['mean_dist_to_train']:<12.2f} "
                  f"{s['mean_position_error']:<12.2f} {s['mean_relative_error']:<12.1f} "
                  f"{s['detection_rate']*100:<12.1f}")
        
        print("-"*105)
        print(f"{'Overall:':<20} {'':<20} {overall_exact['mean_dist_to_train']:<12.2f} "
              f"{overall_exact['mean_position_error']:<12.2f} "
              f"{overall_exact['mean_relative_error']:<12.1f} "
              f"{overall_exact['detection_rate']*100:<12.1f}")
    
    # Interpolation results
    if summary_interp:
        print("\nINTERPOLATION RESULTS:")
        print(f"{'Position':<20} {'Nearest Train':<20} {'Train Dist':<12} {'Mean Err':<12} {'Rel Err%':<12} {'Det Rate%':<12}")
        print("-"*105)
        for pos in sorted(summary_interp.keys()):
            s = summary_interp[pos]
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            nearest_str = f"({s['nearest_train_pos'][0]:.1f}, {s['nearest_train_pos'][1]:.1f})"
            print(f"{pos_str:<20} {nearest_str:<20} {s['mean_dist_to_train']:<12.2f} "
                  f"{s['mean_position_error']:<12.2f} {s['mean_relative_error']:<12.1f} "
                  f"{s['detection_rate']*100:<12.1f}")
        
        print("-"*105)
        print(f"{'Overall:':<20} {'':<20} {overall_interp['mean_dist_to_train']:<12.2f} "
              f"{overall_interp['mean_position_error']:<12.2f} "
              f"{overall_interp['mean_relative_error']:<12.1f} "
              f"{overall_interp['detection_rate']*100:<12.1f}")
    
    # Extrapolation results
    if summary_extrap:
        print("\nEXTRAPOLATION RESULTS:")
        print(f"{'Position':<20} {'Nearest Train':<20} {'Train Dist':<12} {'Mean Err':<12} {'Rel Err%':<12} {'Det Rate%':<12}")
        print("-"*105)
        for pos in sorted(summary_extrap.keys()):
            s = summary_extrap[pos]
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            nearest_str = f"({s['nearest_train_pos'][0]:.1f}, {s['nearest_train_pos'][1]:.1f})"
            print(f"{pos_str:<20} {nearest_str:<20} {s['mean_dist_to_train']:<12.2f} "
                  f"{s['mean_position_error']:<12.2f} {s['mean_relative_error']:<12.1f} "
                  f"{s['detection_rate']*100:<12.1f}")
        
        print("-"*105)
        print(f"{'Overall:':<20} {'':<20} {overall_extrap['mean_dist_to_train']:<12.2f} "
              f"{overall_extrap['mean_position_error']:<12.2f} "
              f"{overall_extrap['mean_relative_error']:<12.1f} "
              f"{overall_extrap['detection_rate']*100:<12.1f}")
    
    # Summary comparison
    print("\n" + "="*110)
    print("SUMMARY COMPARISON")
    print("="*110)
    print(f"{'Metric':<35} {'Exact Training':<20} {'Interpolation':<20} {'Extrapolation':<20}")
    print("-"*90)
    
    exact_pos_err = overall_exact.get('mean_position_error', float('inf')) if overall_exact else float('inf')
    interp_pos_err = overall_interp.get('mean_position_error', float('inf')) if overall_interp else float('inf')
    extrap_pos_err = overall_extrap.get('mean_position_error', float('inf')) if overall_extrap else float('inf')
    
    exact_rel_err = overall_exact.get('mean_relative_error', float('inf')) if overall_exact else float('inf')
    interp_rel_err = overall_interp.get('mean_relative_error', float('inf')) if overall_interp else float('inf')
    extrap_rel_err = overall_extrap.get('mean_relative_error', float('inf')) if overall_extrap else float('inf')
    
    exact_det_rate = overall_exact.get('detection_rate', 0) if overall_exact else 0
    interp_det_rate = overall_interp.get('detection_rate', 0) if overall_interp else 0
    extrap_det_rate = overall_extrap.get('detection_rate', 0) if overall_extrap else 0
    
    exact_dist_train = overall_exact.get('mean_dist_to_train', 0) if overall_exact else 0
    interp_dist_train = overall_interp.get('mean_dist_to_train', 0) if overall_interp else 0
    extrap_dist_train = overall_extrap.get('mean_dist_to_train', 0) if overall_extrap else 0
    
    print(f"{'Mean Position Error (pixels)':<35} {exact_pos_err:<20.2f} "
          f"{interp_pos_err:<20.2f} "
          f"{extrap_pos_err:<20.2f}")
    print(f"{'Mean Relative Error (%)':<35} {exact_rel_err:<20.1f} "
          f"{interp_rel_err:<20.1f} "
          f"{extrap_rel_err:<20.1f}")
    print(f"{'Detection Rate (%)':<35} {exact_det_rate*100:<20.1f} "
          f"{interp_det_rate*100:<20.1f} "
          f"{extrap_det_rate*100:<20.1f}")
    print(f"{'Mean Distance to Training':<35} {exact_dist_train:<20.2f} "
          f"{interp_dist_train:<20.2f} "
          f"{extrap_dist_train:<20.2f}")


def calculate_summary_statistics_enhanced(results, label=""):
    """
    Calculate enhanced summary statistics including nearest training point info.
    """
    if not results:
        return {}, {}
        
    summary = {}
    all_position_errors = []
    all_relative_errors = []
    all_detection_successes = []
    all_distances_to_train = []
    
    for pos in results.keys():
        # Filter out failed detections for error statistics
        valid_errors = [e for e, success in zip(results[pos]['position_errors'], 
                                               results[pos]['detection_successes']) if success]
        valid_rel_errors = [e for e, success in zip(results[pos]['relative_errors'], 
                                                   results[pos]['detection_successes']) if success]
        
        # Get nearest training position (should be same for all samples of this position)
        nearest_train = results[pos]['nearest_train_positions'][0]
        mean_dist_to_train = np.mean(results[pos]['distances_to_nearest_train'])
        
        summary[pos] = {
            'mean_position_error': np.mean(valid_errors) if valid_errors else float('inf'),
            'std_position_error': np.std(valid_errors) if valid_errors else 0,
            'mean_relative_error': np.mean(valid_rel_errors) if valid_rel_errors else float('inf'),
            'std_relative_error': np.std(valid_rel_errors) if valid_rel_errors else 0,
            'detection_rate': np.mean(results[pos]['detection_successes']),
            'num_samples': len(results[pos]['detection_successes']),
            'nearest_train_pos': nearest_train,
            'mean_dist_to_train': mean_dist_to_train
        }
        
        all_position_errors.extend(valid_errors)
        all_relative_errors.extend(valid_rel_errors)
        all_detection_successes.extend(results[pos]['detection_successes'])
        all_distances_to_train.extend(results[pos]['distances_to_nearest_train'])
    
    # Overall statistics
    overall = {
        'mean_position_error': np.mean(all_position_errors) if all_position_errors else float('inf'),
        'std_position_error': np.std(all_position_errors) if all_position_errors else 0,
        'median_position_error': np.median(all_position_errors) if all_position_errors else float('inf'),
        'mean_relative_error': np.mean(all_relative_errors) if all_relative_errors else float('inf'),
        'std_relative_error': np.std(all_relative_errors) if all_relative_errors else 0,
        'detection_rate': np.mean(all_detection_successes) if all_detection_successes else 0,
        'total_samples': len(all_detection_successes),
        'mean_dist_to_train': np.mean(all_distances_to_train) if all_distances_to_train else 0
    }
    
    return summary, overall


def load_model(config, device):
    """Load the trained model (DiT or UNet)."""
    input_size = config['image_size']
    in_channels = 3  # Default
    architecture = config.get('architecture', 'dit')
    
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
    
    # Check if this is a VAE model (latent diffusion)
    # Be careful not to match "novae" or "no_vae" (no VAE) - only match "vae" as a standalone word
    ckpt_path_lower = str(config['ckpt_path']).lower()
    # Trust the training run's own flag when the caller supplies it; the path
    # heuristic below only fires when it is absent, and misses directory names
    # like ".../hires128_vae/position/...".
    is_vae_model = bool(config.get('use_latent_diffusion')) or (
                  ('_vae_' in ckpt_path_lower or 
                   ckpt_path_lower.endswith('_vae') or 
                   '/vae_' in ckpt_path_lower or
                   '/vae/' in ckpt_path_lower) and 'novae' not in ckpt_path_lower and 'no_vae' not in ckpt_path_lower)
    # The run config records this explicitly and is authoritative; the path
    # heuristic below is only a fallback for older checkpoints whose run_config
    # predates the flag. Relying on the path alone silently built a pixel-space
    # model for latent checkpoints under directories like "hires128_vae/".
    if config.get('use_latent_diffusion'):
        is_vae_model = True

    if is_vae_model:
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
    else:
        # For non-VAE models, keep standard settings
        config['use_latent_diffusion'] = False
    
    # For SongUNet, detect channels from checkpoint (concat conditioning adds channels)
    if architecture == 'songunet' and not is_vae_model:
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
        extra_channels = 2  # position uses x,y channels when concatenated
        if checkpoint_channels > 3:
            if config.get('conditioning_method') != 'concat':
                print("Detected extra input channels; overriding conditioning_method to 'concat'")
            config['conditioning_method'] = 'concat'
            base_channels = checkpoint_channels - extra_channels
        else:
            if config.get('conditioning_method') != 'adaln':
                print("Detected 3-channel input; overriding conditioning_method to 'adaln'")
            config['conditioning_method'] = 'adaln'
            base_channels = checkpoint_channels

        in_channels = max(1, base_channels)
    
    model_key = config.get('model_name', config.get('model_type', 'DiT-S/2'))
    print(f"DEBUG load_model: model_key={model_key}, model_name={config.get('model_name')}, model_type={config.get('model_type')}")
    
    if architecture == 'songunet':
        # Use standard SongUNet-Position-S
        # The model uses: model_channels=128, channel_mult=[1,2,2]
        model_key = 'SongUNet-Position-S'
        print(f"Using standard {model_key} at {input_size}x{input_size} resolution")
        model = SongUNet_models[model_key](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=in_channels,
            position_embedding_type=config['position_embedding_type'],
            conditioning_method=config['conditioning_method'],
            position_dropout_prob=config['position_dropout_prob']
        ).to(device)
    elif architecture == 'unet':
        model = UNet_models[config['model_type']](
            input_size=input_size,
            in_channels=in_channels,
            position_embedding_type=config['position_embedding_type'],
            conditioning_method=config['conditioning_method'],
            position_dropout_prob=config['position_dropout_prob'],
            max_position_value=config['train_max_pos']  # Max position value for the dataset
        ).to(device)
    else:
        model = DiT_models[config['model_type']](
            input_size=input_size,
            in_channels=in_channels,
            position_embedding_type=config['position_embedding_type'],
            conditioning_method=config['conditioning_method'],
            position_dropout_prob=config['position_dropout_prob'],
            null_position=config['null_position'],
            null_embedding_type=config['null_embedding_type']
        ).to(device)
    
    if 'ema' in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint['ema'], strict=False)
        print("Loaded EMA weights")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    elif 'model' in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print("Loaded model weights")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print("Loaded weights directly")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    model.eval()
    return model


def convert_tuple_keys_to_strings(summary_dict):
    """Convert tuple keys to string representation for JSON serialization."""
    if not summary_dict:
        return {}
    converted = {}
    for key, value in summary_dict.items():
        if isinstance(key, tuple):
            # Convert tuple (x, y) to string "x,y"
            str_key = f"{key[0]},{key[1]}"
        else:
            str_key = str(key)
        converted[str_key] = value
    return converted

def generate_samples(
    model,
    diffusion,
    position_values,
    config,
    device,
    num_samples=3,
    cfg_scale=1.0,
    batch_size=1,
    num_sampling_steps=250,
):
    """Generate samples for given position values with optional CFG."""
    samples_dict = {}
    
    input_size = config['image_size']
    use_cfg = config.get('use_cfg', False) and cfg_scale > 1.0
    use_latent_diffusion = config.get('use_latent_diffusion', False)
    
    # Load VAE if needed
    vae = None
    if use_latent_diffusion:
        try:
            from diffusers.models import AutoencoderKL
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
    
    for pos in tqdm(position_values, desc="Generating samples"):
        samples_list = []
        
        sample_index = 0
        while sample_index < num_samples:
            current_batch = min(batch_size, num_samples - sample_index)
            
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
            z_shape = z.shape
            
            # Check if model has forward_with_cfg method
            has_forward_with_cfg = hasattr(model, 'forward_with_cfg')
            
            # Sample with or without CFG
            with torch.no_grad():
                # Convert position tuple to tensor
                pos_tensor = torch.tensor(
                    [pos] * current_batch, device=device, dtype=torch.float32
                )
                
                # Check if we're using flow matching
                is_flow_matching = config.get('use_flow_matching', False)
                
                if is_flow_matching:
                    # Flow matching sampling
                    if use_cfg and has_forward_with_cfg:
                        # Flow matching with CFG
                        sample = diffusion.sample_with_cfg(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            cfg_scale=cfg_scale,
                            model_kwargs=dict(pos=pos_tensor),
                            device=device,
                        )
                    else:
                        # Flow matching without CFG
                        sample = diffusion.sample(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            model_kwargs=dict(pos=pos_tensor),
                            device=device,
                        )
                else:
                    # Diffusion sampling
                    sample = diffusion.p_sample_loop(
                        model,
                        z_shape,
                        noise=z,
                        clip_denoised=True,
                        model_kwargs=dict(pos=pos_tensor),
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
        
        samples_dict[pos] = samples_list
    
    return samples_dict


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse configuration from checkpoint path
    config = parse_checkpoint_path(args.ckpt)
    config['eval_cfg_scale'] = args.cfg_scale
    
    # Override image size if specified
    if args.image_size is not None:
        print(f"Overriding auto-detected image size {config['image_size']} with {args.image_size}")
        config['image_size'] = args.image_size
        config['valid_min'] = -config['image_size'] / 2 + config['circle_radius']
        config['valid_max'] = config['image_size'] / 2 - config['circle_radius']
    
    # Create output directory
    cfg_suffix = f"_cfg{args.cfg_scale}" if config.get('use_cfg', False) and args.cfg_scale > 1.0 else ""
    output_dir = os.path.join("eval_resume", f"{config['folder_name']}{cfg_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("\nInferred Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Generate training positions
    training_positions = generate_training_positions(config['square_size'], config['density'])
    print(f"\nGenerated {len(training_positions)} training positions")
    
    # Get evaluation position values
    exact_train_positions, interp_positions, extrap_positions, _ = get_evaluation_positions(
        config['train_min_pos'], 
        config['train_max_pos'],
        config['square_size'],
        config['valid_min'],
        config['valid_max'],
        density=config.get('density', 5)
    )
    
    print(f"\nTraining configuration:")
    print(f"  Square size: {config['square_size']} (range: [{config['train_min_pos']:.1f}, {config['train_max_pos']:.1f}])")
    print(f"  Density: {config.get('density', 5)}x{config.get('density', 5)} grid")
    print(f"  Valid range: [{config['valid_min']:.1f}, {config['valid_max']:.1f}]")
    print(f"\nExact training positions ({len(exact_train_positions)}): {exact_train_positions}")
    print(f"Interpolation positions ({len(interp_positions)}): {interp_positions}")
    print(f"Extrapolation positions ({len(extrap_positions)}): {extrap_positions}")
    
    # Load model
    model = load_model(config, device)
    
    # Create diffusion or flow matching with appropriate settings
    if config.get('use_flow_matching', False):
        print("Using Flow Matching sampler")
        diffusion = FlowMatching(sigma_min=0.0, sigma_data=1.0, use_sigmoid_time=True)
    else:
        print("Using Diffusion sampler")
        # SongUNet doesn't learn sigma, so we need to specify that
        if config.get('architecture') == 'songunet':
            diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)
            print("  (SongUNet: learn_sigma=False)")
        else:
            diffusion = create_diffusion(str(args.num_sampling_steps))
            print("  (DiT/UNet: learn_sigma=True)")
    
    # Generate samples for all positions
    all_positions = exact_train_positions + interp_positions + extrap_positions
    print(f"\nGenerating samples for all position values...")
    samples_dict = generate_samples(
        model, diffusion, all_positions, config, device, 
        num_samples=args.num_samples_per_position,
        cfg_scale=args.cfg_scale,
        batch_size=args.eval_batch_size,
        num_sampling_steps=args.num_sampling_steps,
    )
    
    # Evaluate exact training positions
    results_exact = {}
    if exact_train_positions:
        print("\nEvaluating exact training position performance...")
        results_exact = evaluate_position_set_enhanced(
            exact_train_positions, 
            {pos: samples_dict[pos] for pos in exact_train_positions}, 
            config, 
            training_positions,
            label="exact training"
        )
    
    # Evaluate interpolation
    results_interp = {}
    if interp_positions:
        print("\nEvaluating interpolation performance...")
        results_interp = evaluate_position_set_enhanced(
            interp_positions, 
            {pos: samples_dict[pos] for pos in interp_positions}, 
            config, 
            training_positions,
            label="interpolation"
        )
    
    # Evaluate extrapolation
    results_extrap = {}
    if extrap_positions:
        print("\nEvaluating extrapolation performance...")
        results_extrap = evaluate_position_set_enhanced(
            extrap_positions, 
            {pos: samples_dict[pos] for pos in extrap_positions}, 
            config, 
            training_positions,
            label="extrapolation"
        )
    
    # Calculate summary statistics
    summary_exact, overall_exact = calculate_summary_statistics_enhanced(results_exact, "Exact Training")
    summary_interp, overall_interp = calculate_summary_statistics_enhanced(results_interp, "Interpolation")
    summary_extrap, overall_extrap = calculate_summary_statistics_enhanced(results_extrap, "Extrapolation")
    
    # Print results
    print_results_table_enhanced(summary_exact, overall_exact, summary_interp, overall_interp, 
                                summary_extrap, overall_extrap, cfg_scale=args.cfg_scale)
    
    # Create comprehensive visualizations (similar to eval_radius.py)
    print("\nCreating visualizations...")
    visualize_results(results_exact, results_interp, results_extrap, output_dir, config)
    
    # Save example comparisons with enhanced visualizations
    # Use simple comparison saving
    save_position_comparisons_simple(
        {**results_exact, **results_interp, **results_extrap}, 
        samples_dict, 
        output_dir
    )
    
    # Save detailed results to JSON
    results_data = {
        'config': config,
        'cfg_scale': args.cfg_scale,
        'exact_training': {
            'positions': exact_train_positions,
            'summary': convert_tuple_keys_to_strings(summary_exact),
            'overall': overall_exact,
        } if results_exact else {},
        'interpolation': {
            'positions': interp_positions,
            'summary': convert_tuple_keys_to_strings(summary_interp),
            'overall': overall_interp,
        } if results_interp else {},
        'extrapolation': {
            'positions': extrap_positions,
            'summary': convert_tuple_keys_to_strings(summary_extrap),
            'overall': overall_extrap,
        } if results_extrap else {}
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)  # default=str to handle numpy types
    
    print(f"\nResults saved to {output_dir}")
    
    # Print coordinate system verification
    print("\n" + "="*80)
    print("COORDINATE SYSTEM VERIFICATION")
    print("="*80)
    print(f"For a {config['image_size']}x{config['image_size']} image:")
    print("- Image coordinates: (0,0) is top-left")
    print(f"- Center is at pixel ({config['image_size']//2}, {config['image_size']//2})")
    print("- Our relative coordinates: (0,0) is center")
    print("  - Positive X = right of center")
    print("  - Positive Y = below center")
    print("  - Negative X = left of center")
    print("  - Negative Y = above center")
    print("- Example: Position (10, -10) means 10 pixels right and 10 pixels up from center")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced position evaluation with comprehensive visualizations')
    
    parser.add_argument("--ckpt", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num-samples-per-position", type=int, default=3,
                        help="Number of samples to generate per position")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of DDPM sampling steps")
    parser.add_argument("--num-visual-examples", type=int, default=3,
                        help="Number of visual examples to save per position")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of samples to generate in parallel during evaluation")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Override auto-detected image size")
    
    args = parser.parse_args()
    main(args)
