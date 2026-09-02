#!/usr/bin/env python
"""
Evaluation script for continuous rotation angle DiT model.
Fixed with template matching for accurate angle detection.
"""
import os
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
import math
from scipy.optimize import minimize_scalar

# Import model components
from vgl.models_rotation import DiT_models_rotation as DiT_models
from vgl.unet_models_rotation import UNet_models_rotation as UNet_models
from vgl.unet_models_song_rotation import SongUNet_Rotation_models as SongUNet_models
from vgl.diffusion import create_diffusion
from vgl.flow_matching import FlowMatching


def generate_ground_truth_arrow(angle_degrees, image_size=64, arrow_size=20,
                               arrow_color=(255, 100, 100), background_color=(255, 255, 255),
                               antialiasing=4):
    """
    Generate arrow exactly as in the dataset generation script.
    This must match your training data generation exactly.
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_arrow_size = arrow_size * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    center_x = large_size // 2
    center_y = large_size // 2
    
    # This is the exact transformation from your generation script
    angle_rad = math.radians(-angle_degrees + 90)
    
    # Arrow points (asymmetric arrow pointing up initially)
    arrow_points = [
        (0, -large_arrow_size * 0.5),  # Tip
        (-large_arrow_size * 0.25, -large_arrow_size * 0.2),
        (-large_arrow_size * 0.15, -large_arrow_size * 0.2),
        (-large_arrow_size * 0.1, large_arrow_size * 0.4),
        (0, large_arrow_size * 0.5),  # Tail
        (large_arrow_size * 0.1, large_arrow_size * 0.4),
        (large_arrow_size * 0.15, -large_arrow_size * 0.2),
        (large_arrow_size * 0.25, -large_arrow_size * 0.2),
    ]
    
    # Rotate and translate vertices
    rotated_points = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for x, y in arrow_points:
        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a
        final_x = center_x + rotated_x
        final_y = center_y + rotated_y
        rotated_points.append((final_x, final_y))
    
    draw.polygon(rotated_points, fill=arrow_color, outline=arrow_color)
    
    # Resize to final size
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def generate_ground_truth_triangle(angle_degrees, image_size=64, triangle_size=20,
                                  triangle_color=(255, 100, 100), background_color=(255, 255, 255),
                                  antialiasing=4):
    """
    Generate right triangle exactly as in the dataset generation script.
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_triangle_size = triangle_size * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    center_x = large_size // 2
    center_y = large_size // 2
    
    # Convert angle to radians (same convention as arrows)
    angle_rad = math.radians(-angle_degrees + 90)
    
    # Define right triangle vertices (pointing up initially)
    triangle_points = [
        (0, -large_triangle_size * 0.5),  # Top vertex
        (large_triangle_size * 0.433, large_triangle_size * 0.25),  # Bottom right
        (-large_triangle_size * 0.2, large_triangle_size * 0.25),  # Bottom left
    ]
    
    # Rotate and translate vertices
    rotated_points = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for x, y in triangle_points:
        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a
        final_x = center_x + rotated_x
        final_y = center_y + rotated_y
        rotated_points.append((final_x, final_y))
    
    draw.polygon(rotated_points, fill=triangle_color, outline=triangle_color)
    
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def extract_arrow_angle_template_matching(image, initial_step=10, refine_steps=[3, 1]):
    """
    Multi-scale template matching for accurate arrow angle detection.
    
    Args:
        image: Input image containing the arrow
        initial_step: Initial coarse search step in degrees
        refine_steps: List of refinement steps for hierarchical search
    
    Returns:
        tuple: (detected_angle, success)
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Start with coarse search
    best_angle = 0
    best_score = -float('inf')
    current_step = initial_step
    
    # Multi-scale refinement
    for step in [initial_step] + refine_steps:
        # Define search angles for this scale
        if step == initial_step:
            # Initial coarse search over full range
            test_angles = np.arange(0, 360, step)
        else:
            # Refined search around best angle
            start = best_angle - current_step * 2
            end = best_angle + current_step * 2 + 1
            test_angles = np.arange(start, end, step)
        
        # Test each angle
        for test_angle in test_angles:
            test_angle = test_angle % 360
            ref_arrow = generate_ground_truth_arrow(test_angle, image_size=gray.shape[0])
            ref_gray = cv2.cvtColor(np.array(ref_arrow), cv2.COLOR_RGB2GRAY)
            
            result = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score:
                best_score = score
                best_angle = test_angle
        
        current_step = step
    
    return best_angle, True


def extract_triangle_angle_template_matching(image, initial_step=10, refine_steps=[3, 1]):
    """
    Multi-scale template matching for triangle angle detection.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    best_angle = 0
    best_score = -float('inf')
    current_step = initial_step
    
    for step in [initial_step] + refine_steps:
        if step == initial_step:
            test_angles = np.arange(0, 360, step)
        else:
            start = best_angle - current_step * 2
            end = best_angle + current_step * 2 + 1
            test_angles = np.arange(start, end, step)
        
        for test_angle in test_angles:
            test_angle = test_angle % 360
            ref_triangle = generate_ground_truth_triangle(test_angle, image_size=gray.shape[0])
            ref_gray = cv2.cvtColor(np.array(ref_triangle), cv2.COLOR_RGB2GRAY)
            
            result = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score:
                best_score = score
                best_angle = test_angle
        
        current_step = step
    
    return best_angle, True


def extract_arrow_angle_continuous(image, num_initial_points=36):
    """
    Continuous optimization for sub-degree accuracy (slower but more accurate).
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    def compute_match_score(angle):
        angle = angle % 360
        ref_arrow = generate_ground_truth_arrow(angle, image_size=gray.shape[0])
        ref_gray = cv2.cvtColor(np.array(ref_arrow), cv2.COLOR_RGB2GRAY)
        result = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        return -np.max(result)  # Negative for minimization
    
    # Coarse global search
    coarse_angles = np.linspace(0, 360, num_initial_points, endpoint=False)
    coarse_scores = [compute_match_score(angle) for angle in coarse_angles]
    best_coarse_idx = np.argmin(coarse_scores)
    best_coarse_angle = coarse_angles[best_coarse_idx]
    
    # Fine-tune with scipy optimization
    result = minimize_scalar(
        compute_match_score,
        bounds=(best_coarse_angle - 10, best_coarse_angle + 10),
        method='bounded',
        options={'xatol': 0.1}
    )
    
    return result.x % 360, True


# Use template matching as the main detection functions
def extract_arrow_angle(image, expected_size=20):
    """Main arrow angle extraction using template matching."""
    return extract_arrow_angle_template_matching(image)


def extract_arrow_angle_robust(image, expected_size=20):
    """Robust arrow angle extraction using template matching."""
    return extract_arrow_angle_template_matching(image)


def extract_triangle_angle(image, expected_size=20):
    """Triangle angle extraction using template matching."""
    return extract_triangle_angle_template_matching(image)


def generate_ground_truth_shape(angle, shape_type='arrow', image_size=64, shape_color=(255, 128, 100), 
                               background_color=(255, 255, 255), antialiasing=4):
    """Generate a ground truth rotated shape image."""
    if shape_type == 'arrow':
        return generate_ground_truth_arrow(angle, image_size, arrow_color=shape_color,
                                         background_color=background_color, antialiasing=antialiasing)
    else:
        return generate_ground_truth_triangle(angle, image_size, triangle_color=shape_color,
                                             background_color=background_color, antialiasing=antialiasing)


def extract_shape_mask(image, threshold_method='otsu'):
    """Extract binary mask of the shape from an image."""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if threshold_method == 'otsu':
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 127:
            mask = 255 - mask
    else:
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    mask = (mask > 127).astype(np.uint8)
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
    
    return intersection / union


def calculate_rotation_metrics(generated_img, expected_angle, shape_type='arrow', image_size=64):
    """Calculate rotation-specific metrics using angle error."""
    # Use template matching for accurate detection
    if shape_type == 'arrow':
        detected_angle, success = extract_arrow_angle_robust(generated_img)
    else:
        detected_angle, success = extract_triangle_angle(generated_img)
    
    if not success:
        return {
            'angle_error': float('inf'),
            'angle_error_abs': float('inf'),
            'detection_success': False,
            'expected_angle': expected_angle,
            'detected_angle': 0.0
        }
    
    # Calculate angle error (handling wrap-around)
    angle_diff = detected_angle - expected_angle
    
    # Handle wrap-around
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    
    angle_error_abs = abs(angle_diff)
    
    return {
        'angle_error': angle_diff,
        'angle_error_abs': angle_error_abs,
        'detection_success': True,
        'expected_angle': expected_angle,
        'detected_angle': detected_angle
    }


def load_dataset_metadata(data_path):
    """Load dataset metadata to get actual training angles."""
    metadata_path = os.path.join(data_path, 'dataset_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def generate_test_angles_from_data(metadata, num_test_samples=10, extrapolation_step=10):
    """Generate test angles with evenly spaced extrapolation angles."""
    training_angles = metadata['rotation_angles']
    min_angle = metadata['min_angle']
    max_angle = metadata['max_angle']
    
    # Training angles (subset for evaluation)
    if len(training_angles) > num_test_samples:
        indices = np.linspace(0, len(training_angles)-1, num_test_samples, dtype=int)
        train_angles_eval = [training_angles[i] for i in indices]
    else:
        train_angles_eval = training_angles.copy()
    
    # Interpolation angles (between training angles)
    interp_angles = []
    if len(training_angles) > 1:
        for i in range(min(num_test_samples, len(training_angles) - 1)):
            mid_angle = (training_angles[i] + training_angles[i+1]) / 2
            interp_angles.append(mid_angle)
    
    # If we need more interpolation samples
    if len(interp_angles) < num_test_samples and len(training_angles) > 2:
        for i in range(len(interp_angles), num_test_samples):
            idx1 = i % (len(training_angles) - 1)
            idx2 = idx1 + 1
            fraction = 0.3 + (i // len(training_angles)) * 0.2
            angle = training_angles[idx1] + (training_angles[idx2] - training_angles[idx1]) * fraction
            interp_angles.append(angle)
    
    # NEW: Evenly spaced extrapolation angles
    extrap_angles = []
    
    # Lower extrapolation: every step degrees from 0 to min_angle-5
    if min_angle > 5:
        lower_start = 0
        lower_end = min_angle - 5
        lower_angles = np.arange(lower_start, lower_end + extrapolation_step, extrapolation_step)
        extrap_angles.extend(lower_angles.tolist())
    
    # Upper extrapolation: every step degrees from max_angle+5 to 355
    if max_angle < 355:
        upper_start = max_angle + 5
        upper_end = 355
        upper_angles = np.arange(upper_start, upper_end + extrapolation_step, extrapolation_step)
        extrap_angles.extend(upper_angles.tolist())
    
    # Ensure we have enough extrapolation angles
    if len(extrap_angles) < num_test_samples:
        # Fill in more angles if needed
        if min_angle > 15:
            additional_lower = [5, 10, 15]
            extrap_angles.extend([a for a in additional_lower if a not in extrap_angles])
        if max_angle < 345:
            additional_upper = [345, 350, 355]
            extrap_angles.extend([a for a in additional_upper if a not in extrap_angles])
    
    # Remove duplicates and sort
    extrap_angles = sorted(list(set(extrap_angles)))
    
    # Limit to reasonable number
    if len(extrap_angles) > num_test_samples * 2:
        # Take every other angle to reduce number
        extrap_angles = extrap_angles[::2]
    
    return train_angles_eval, interp_angles[:num_test_samples], extrap_angles, training_angles


def find_nearest_training_angle(test_angle, training_angles):
    """Find the nearest training angle."""
    min_distance = float('inf')
    nearest_angle = training_angles[0]
    
    for train_angle in training_angles:
        diff = abs(test_angle - train_angle)
        if diff > 180:
            diff = 360 - diff
        
        if diff < min_distance:
            min_distance = diff
            nearest_angle = train_angle
    
    return nearest_angle, min_distance


def generate_samples(
    model,
    diffusion,
    angle_values,
    config,
    device,
    num_samples=3,
    cfg_scale=1.0,
    batch_size=1,
    num_sampling_steps=250,
):
    """Generate samples for given rotation angles."""
    samples_dict = {}
    
    input_size = config['image_size']
    use_cfg = config.get('use_cfg', False) and cfg_scale > 1.0
    use_latent_diffusion = config.get('use_latent_diffusion', False)
    use_flow_matching = config.get('use_flow_matching', False)
    
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
    
    for angle in tqdm(angle_values, desc="Generating samples"):
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
            
            has_forward_with_cfg = hasattr(model, 'forward_with_cfg')
            
            with torch.no_grad():
                if use_cfg and has_forward_with_cfg:
                    angle_cond = torch.full(
                        (current_batch,),
                        angle,
                        device=device,
                        dtype=torch.float32,
                    )
                    angle_uncond = torch.full(
                        (current_batch,),
                        config['null_angle'],
                        device=device,
                        dtype=torch.float32,
                    )
                    angle_doubled = torch.cat([angle_cond, angle_uncond], 0)
                    
                    def cfg_forward(x, t, **kwargs):
                        return model.forward_with_cfg(x, t, angle_doubled, cfg_scale)
                    
                    if use_flow_matching:
                        # Flow matching with CFG
                        sample = diffusion.sample_with_cfg(
                            model,
                            z_shape,
                            angle_tensor=angle_doubled,
                            cfg_scale=cfg_scale,
                            num_steps=num_sampling_steps,
                            model_kwargs={},
                            device=device,
                        )
                    else:
                        sample = diffusion.p_sample_loop(
                            cfg_forward,
                            z_shape,
                            noise=z,
                            clip_denoised=False,
                            model_kwargs={},
                            progress=False,
                            device=device,
                        )
                else:
                    angle_tensor = torch.full(
                        (current_batch,),
                        angle,
                        device=device,
                        dtype=torch.float32,
                    )
                    if use_flow_matching:
                        # Flow matching without CFG
                        sample = diffusion.sample(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            model_kwargs=dict(rotation=angle_tensor),
                            device=device,
                        )
                    else:
                        sample = diffusion.p_sample_loop(
                            model.forward,
                            z_shape,
                            noise=z,
                            clip_denoised=False,
                            model_kwargs=dict(rotation=angle_tensor),
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
        
        samples_dict[angle] = samples_list
    
    return samples_dict


def evaluate_angle_set(angle_values, samples_dict, config, training_angles, shape_type='arrow', label=""):
    """Evaluate a set of rotation angles using angle error metrics."""
    results = {}
    
    for angle in tqdm(angle_values, desc=f"Evaluating {label}"):
        results[angle] = {
            'angle_errors': [],
            'angle_errors_abs': [],
            'detection_successes': [],
            'expected_angles': [],
            'detected_angles': [],
            'nearest_train_angles': [],
            'distances_to_nearest_train': []
        }
        
        nearest_train, dist_to_train = find_nearest_training_angle(angle, training_angles)
        
        for gen_img in samples_dict[angle]:
            metrics = calculate_rotation_metrics(gen_img, angle, shape_type, config['image_size'])
            
            results[angle]['angle_errors'].append(metrics['angle_error'])
            results[angle]['angle_errors_abs'].append(metrics['angle_error_abs'])
            results[angle]['detection_successes'].append(metrics['detection_success'])
            results[angle]['expected_angles'].append(metrics['expected_angle'])
            results[angle]['detected_angles'].append(metrics['detected_angle'])
            results[angle]['nearest_train_angles'].append(nearest_train)
            results[angle]['distances_to_nearest_train'].append(dist_to_train)
    
    return results


def save_categorized_examples(results_exact, results_interp, results_extrap, samples_dict, save_path, config, num_examples=2):
    """Save visual comparisons categorized by train/interpolation/extrapolation."""
    os.makedirs(os.path.join(save_path, 'comparisons'), exist_ok=True)
    
    shape_type = config.get('shape_type', 'arrow')
    
    categories = [
        ('train', results_exact, 'Training Angles'),
        ('interpolation', results_interp, 'Interpolation Angles'),
        ('extrapolation', results_extrap, 'Extrapolation Angles')
    ]
    
    for cat_name, cat_results, cat_title in categories:
        if not cat_results:
            continue
            
        angles_to_show = list(cat_results.keys())[:min(3, len(cat_results))]
        
        if not angles_to_show:
            continue
            
        fig, axes = plt.subplots(len(angles_to_show), 3, figsize=(12, 4*len(angles_to_show)))
        if len(angles_to_show) == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, angle in enumerate(angles_to_show):
            # Show first sample for this angle
            img = samples_dict[angle][0]
            axes[row_idx, 0].imshow(img)
            axes[row_idx, 0].set_title(f'Generated\nAngle: {angle:.1f}°')
            axes[row_idx, 0].axis('off')
            
            # Angle visualization
            ax = axes[row_idx, 1]
            ax.set_aspect('equal')
            
            # Use the generation convention for visualization
            expected_rad = math.radians(-angle + 90)
            ax.arrow(0, 0, 0.8*math.cos(expected_rad), 0.8*math.sin(expected_rad),
                    head_width=0.1, head_length=0.1, fc='green', ec='green', 
                    linewidth=2, label='Expected')
            
            if cat_results[angle]['detection_successes'][0]:
                detected = cat_results[angle]['detected_angles'][0]
                detected_rad = math.radians(-detected + 90)
                ax.arrow(0, 0, 0.8*math.cos(detected_rad), 0.8*math.sin(detected_rad),
                        head_width=0.1, head_length=0.1, fc='red', ec='red', 
                        linewidth=2, label='Detected', alpha=0.7)
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title('Angle Comparison')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Metrics
            if cat_results[angle]['detection_successes'][0]:
                detected = cat_results[angle]['detected_angles'][0]
                error = cat_results[angle]['angle_errors_abs'][0]
                text = (f'Expected: {angle:.1f}°\n'
                       f'Detected: {detected:.1f}°\n'
                       f'Error: {error:.1f}°')
            else:
                text = f'Expected: {angle:.1f}°\nDetection Failed'
            
            axes[row_idx, 2].text(0.5, 0.5, text, transform=axes[row_idx, 2].transAxes,
                          fontsize=12, verticalalignment='center', horizontalalignment='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            axes[row_idx, 2].axis('off')
        
        plt.suptitle(f'{cat_title} - {shape_type.capitalize()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'comparisons', f'{cat_name}_examples.png'), dpi=150)
        plt.close()


def create_summary_table(summary_exact, summary_interp, summary_extrap, save_path):
    """Create and save a summary table of results."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Category', 'Mean Error (°)', 'Std Error (°)', 'Detection Rate (%)', 'Median Error (°)']
    
    data = []
    
    if summary_exact:
        data.append([
            'Training Angles',
            f"{summary_exact.get('mean_angle_error', 0):.1f}",
            f"{summary_exact.get('std_angle_error', 0):.1f}",
            f"{summary_exact.get('detection_rate', 0)*100:.1f}",
            f"{summary_exact.get('median_angle_error', 0):.1f}"
        ])
    
    if summary_interp:
        data.append([
            'Interpolation',
            f"{summary_interp.get('mean_angle_error', 0):.1f}",
            f"{summary_interp.get('std_angle_error', 0):.1f}",
            f"{summary_interp.get('detection_rate', 0)*100:.1f}",
            f"{summary_interp.get('median_angle_error', 0):.1f}"
        ])
    
    if summary_extrap:
        data.append([
            'Extrapolation',
            f"{summary_extrap.get('mean_angle_error', 0):.1f}",
            f"{summary_extrap.get('std_angle_error', 0):.1f}",
            f"{summary_extrap.get('detection_rate', 0)*100:.1f}",
            f"{summary_extrap.get('median_angle_error', 0):.1f}"
        ])
    
    table = ax.table(cellText=data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.2, 0.15, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    colors = ['#E8F5E9', '#FFF3E0', '#FFEBEE']
    for i, row in enumerate(data, 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(colors[i-1] if i <= 3 else '#FFFFFF')
    
    plt.title('Rotation Evaluation Summary Table (Angle Error)', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_path, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("SUMMARY TABLE (Angle Error)")
    print("="*80)
    print(f"{'Category':<20} {'Mean Error':<12} {'Std Error':<10} {'Detection %':<15} {'Median Error':<15}")
    print("-"*80)
    
    for row in data:
        print(f"{row[0]:<20} {row[1]:<12} {row[2]:<10} {row[3]:<15} {row[4]:<15}")
    print("="*80)


def load_model(config, device):
    """Load the trained model (DiT or UNet)."""
    input_size = config['image_size']
    in_channels = 3  # Default
    architecture = config.get('architecture', 'dit')
    
    # Load checkpoint first to check model configuration
    print(f"Loading checkpoint from {config['ckpt_path']}...")
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
    is_vae_model = ('_vae_' in ckpt_path_lower or 
                   ckpt_path_lower.endswith('_vae') or 
                   '/vae_' in ckpt_path_lower or
                   '/vae/' in ckpt_path_lower) and 'novae' not in ckpt_path_lower and 'no_vae' not in ckpt_path_lower
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
    if architecture == 'songunet' and not config.get('use_latent_diffusion', False):
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
        extra_channels = 1  # rotation uses scalar channel when concatenated
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
    
    if architecture == 'unet':
        model = UNet_models[config['model_type']](
            input_size=input_size,
            in_channels=in_channels,
            rotation_embedding_type=config.get('rotation_embedding_type', 'circular'),
            conditioning_method=config.get('conditioning_method', 'concat'),
            class_dropout_prob=config.get('rotation_dropout_prob', 0.0)
        ).to(device)
    elif architecture == 'songunet':
        model = SongUNet_models[config['model_type']](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=in_channels,  # SongUNet doesn't learn sigma by default
            rotation_embedding_type=config.get('rotation_embedding_type', 'sinusoidal'),
            conditioning_method=config.get('conditioning_method', 'adaln'),
            rotation_dropout_prob=config.get('rotation_dropout_prob', 0.0)
        ).to(device)
    else:
        model = DiT_models[config['model_type']](
            input_size=input_size,
            in_channels=in_channels,
            rotation_embedding_type=config.get('rotation_embedding_type', 'linear'),
            conditioning_method=config.get('conditioning_method', 'concat'),
            rotation_dropout_prob=config.get('rotation_dropout_prob', 0.0),
            null_angle=config.get('null_angle', 180.0),
            null_embedding_type=config.get('null_embedding_type', 'learnable'),
            rotation_text_table=config.get('rotation_text_table')
        ).to(device)
    
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


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset metadata
    metadata = load_dataset_metadata(args.data_path)
    print(f"\nLoaded dataset metadata:")
    print(f"  Shape type: {metadata['shape_type']}")
    print(f"  Rotation range: [{metadata['min_angle']}°, {metadata['max_angle']}°]")
    print(f"  Number of training angles: {metadata['num_angles']}")
    print(f"  Coverage percentage: {metadata['coverage_percent']}%")
    
    # Detect architecture from model name or checkpoint path
    ckpt_path_lower = str(args.ckpt).lower()
    if 'songunet' in args.model.lower() or 'songunet' in ckpt_path_lower:
        architecture = 'songunet'
        model_key = 'SongUNet-Rotation-S'  # Correct model key for SongUNet Rotation
    elif 'unet' in args.model.lower() or 'unet' in ckpt_path_lower:
        architecture = 'unet'
        model_key = args.model
    else:
        architecture = 'dit'
        model_key = args.model
    
    # Parse configuration from checkpoint path
    
    # Parse rotation embedding type
    if 'rotary' in ckpt_path_lower:
        rotation_embedding_type = 'rotary'
    elif 'sinusoidal' in ckpt_path_lower:
        rotation_embedding_type = 'sinusoidal'  
    elif 'linear' in ckpt_path_lower:
        rotation_embedding_type = 'linear'
    elif architecture == 'songunet':
        rotation_embedding_type = 'sinusoidal'  # NEW models use sinusoidal
    else:
        rotation_embedding_type = 'linear'
    
    # Parse conditioning method
    if 'adaln' in ckpt_path_lower:
        conditioning_method = 'adaln'
    elif 'concat' in ckpt_path_lower:
        conditioning_method = 'concat'
    elif architecture == 'songunet':
        # Rotation NEW model uses concat (has 4 channels), but only if not explicitly specified
        if 'adaln' not in ckpt_path_lower and 'concat' not in ckpt_path_lower:
            conditioning_method = 'concat'  # Rotation NEW model uses concat
        else:
            conditioning_method = 'concat'  # default
    else:
        conditioning_method = 'concat'
    
    # Parse flow matching
    if 'flow' in ckpt_path_lower:
        use_flow_matching = True
        print(f"Detected flow matching model")
    else:
        use_flow_matching = False
    
    config = {
        'ckpt_path': args.ckpt,
        'model_type': model_key,
        'architecture': architecture,
        'image_size': args.image_size,
        'min_angle': metadata['min_angle'],
        'max_angle': metadata['max_angle'],
        'coverage_percent': metadata['coverage_percent'],
        'use_cfg': args.cfg_scale > 1.0,
        'null_angle': 180.0,
        'rotation_embedding_type': rotation_embedding_type,
        'conditioning_method': conditioning_method,
        'rotation_dropout_prob': 0.1 if args.cfg_scale > 1.0 else 0.0,
        'null_embedding_type': 'learnable',
        'shape_type': metadata['shape_type'],
        'data_path': args.data_path,
        'use_flow_matching': use_flow_matching
    }
    
    # Create output directory
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    if not folder_name or folder_name in ['checkpoints', '000-DiT-S-2-rotation', '001-DiT-S-2-rotation']:
        folder_name = f"rotation_{config['rotation_embedding_type']}_{config['conditioning_method']}"
        if 'adaln' in str(args.ckpt).lower():
            folder_name += "_adaln"
        ckpt_str_lower = str(args.ckpt).lower()
        if ('_vae_' in ckpt_str_lower or ckpt_str_lower.endswith('_vae') or '/vae_' in ckpt_str_lower or '/vae/' in ckpt_str_lower) and 'novae' not in ckpt_str_lower and 'no_vae' not in ckpt_str_lower:
            folder_name += "_vae"
    
    cfg_suffix = f"_cfg{args.cfg_scale}" if args.cfg_scale > 1.0 else ""
    output_dir = os.path.join("eval_results", f"{folder_name}{cfg_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nEvaluation Configuration:")
    print(f"  Shape Type: {metadata['shape_type']}")
    print(f"  Rotation Range: [{metadata['min_angle']}°, {metadata['max_angle']}°]")
    print(f"  Coverage: {metadata['coverage_percent']}%")
    print(f"  Training angles: {metadata['num_angles']}")
    print(f"  CFG Scale: {args.cfg_scale}")
    print(f"  Extrapolation Step: {args.extrapolation_step}°")
    print(f"  Detection Method: Template Matching (high accuracy)")
    
    # Generate test angles with evenly spaced extrapolation
    train_angles_eval, interp_angles, extrap_angles, all_train_angles = generate_test_angles_from_data(
        metadata, args.num_test_samples, args.extrapolation_step
    )
    
    config['training_angles'] = all_train_angles
    
    print(f"\nTest angles generated:")
    print(f"  Exact training: {len(train_angles_eval)} angles")
    if train_angles_eval:
        print(f"    Examples: {train_angles_eval[:3]}...")
    print(f"  Interpolation: {len(interp_angles)} angles")
    if interp_angles:
        print(f"    Examples: {interp_angles[:3]}...")
    print(f"  Extrapolation: {len(extrap_angles)} angles")
    if extrap_angles:
        print(f"    Examples: {extrap_angles[:3]}...")
    
    model = load_model(config, device)
    # Create diffusion or flow matching with appropriate settings
    if config.get('use_flow_matching', False):
        print("Using Flow Matching sampler")
        diffusion = FlowMatching(sigma_min=0.0, sigma_data=1.0, use_sigmoid_time=True)
    else:
        print("Using Diffusion sampler")
        # SongUNet doesn't learn sigma, so we need to specify that
        if architecture == 'songunet':
            diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)
            print("  (SongUNet: learn_sigma=False)")
        else:
            diffusion = create_diffusion(str(args.num_sampling_steps))
            print("  (DiT/UNet: learn_sigma=True)")
    
    all_angles = train_angles_eval + interp_angles + extrap_angles
    print(f"\nGenerating samples for {len(all_angles)} angles...")
    samples_dict = generate_samples(
        model, diffusion, all_angles, config, device,
        num_samples=args.num_samples_per_angle,
        cfg_scale=args.cfg_scale,
        batch_size=args.eval_batch_size,
        num_sampling_steps=args.num_sampling_steps,
    )
    
    results_exact = evaluate_angle_set(
        train_angles_eval,
        {a: samples_dict[a] for a in train_angles_eval},
        config, all_train_angles, metadata['shape_type'], "exact training"
    ) if train_angles_eval else {}
    
    results_interp = evaluate_angle_set(
        interp_angles,
        {a: samples_dict[a] for a in interp_angles},
        config, all_train_angles, metadata['shape_type'], "interpolation"
    ) if interp_angles else {}
    
    results_extrap = evaluate_angle_set(
        extrap_angles,
        {a: samples_dict[a] for a in extrap_angles},
        config, all_train_angles, metadata['shape_type'], "extrapolation"
    ) if extrap_angles else {}
    
    def calculate_summary(results):
        if not results:
            return {}
        
        all_angle_errors = []
        all_detections = []
        
        for angle, data in results.items():
            successful_errors = [err for err, success in zip(data['angle_errors_abs'], data['detection_successes']) if success]
            all_angle_errors.extend(successful_errors)
            all_detections.extend(data['detection_successes'])
        
        return {
            'mean_angle_error': np.mean(all_angle_errors) if all_angle_errors else float('inf'),
            'std_angle_error': np.std(all_angle_errors) if all_angle_errors else 0,
            'median_angle_error': np.median(all_angle_errors) if all_angle_errors else float('inf'),
            'detection_rate': np.mean(all_detections) if all_detections else 0
        }
    
    summary_exact = calculate_summary(results_exact)
    summary_interp = calculate_summary(results_interp)
    summary_extrap = calculate_summary(results_extrap)
    
    print("\nCreating visualizations...")
    
    save_categorized_examples(results_exact, results_interp, results_extrap, 
                             samples_dict, output_dir, config, args.num_visual_examples)
    
    create_summary_table(summary_exact, summary_interp, summary_extrap, output_dir)
    
    results_data = {
        'config': config,
        'coverage_percent': metadata['coverage_percent'],
        'shape_type': metadata['shape_type'],
        'training_angles': all_train_angles,
        'detection_method': 'template_matching',
        'summary': {
            'exact_training': summary_exact,
            'interpolation': summary_interp,
            'extrapolation': summary_extrap
        },
        'detailed_results': {
            'exact_training': {str(k): v for k, v in results_exact.items()},
            'interpolation': {str(k): v for k, v in results_interp.items()},
            'extrapolation': {str(k): v for k, v in results_extrap.items()}
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")
    print("\nGenerated files:")
    print("  - comparisons/train_examples.png")
    print("  - comparisons/interpolation_examples.png")
    print("  - comparisons/extrapolation_examples.png")
    print("  - summary_table.png")
    print("  - results.json")
    print(f"\nOutput directory: {output_dir}")
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK RESULTS SUMMARY")
    print("="*60)
    if summary_exact:
        print(f"Training Angles: {summary_exact['mean_angle_error']:.2f}° ± {summary_exact['std_angle_error']:.2f}°")
    if summary_interp:
        print(f"Interpolation:   {summary_interp['mean_angle_error']:.2f}° ± {summary_interp['std_angle_error']:.2f}°")
    if summary_extrap:
        print(f"Extrapolation:   {summary_extrap['mean_angle_error']:.2f}° ± {summary_extrap['std_angle_error']:.2f}°")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate rotation conditioning model')
    
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training data directory (contains dataset_metadata.json)")
    parser.add_argument("--model", type=str, default="DiT-S/2",
                       choices=list(DiT_models.keys()) + list(UNet_models.keys()),
                       help="Model type")
    parser.add_argument("--image-size", type=int, default=64,
                       help="Image size")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num-samples-per-angle", type=int, default=3,
                       help="Number of samples to generate per angle")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                       help="Number of DDPM sampling steps")
    parser.add_argument("--num-test-samples", type=int, default=5,
                       help="Number of test samples per category")
    parser.add_argument("--num-visual-examples", type=int, default=2,
                       help="Number of visual examples to save")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of samples to generate in parallel during evaluation")
    parser.add_argument("--use-continuous", action='store_true',
                       help="Use continuous optimization for sub-degree accuracy (slower)")
    parser.add_argument("--extrapolation-step", type=int, default=10,
                       help="Step size in degrees for evenly spaced extrapolation angles")
    
    args = parser.parse_args()
    main(args)
