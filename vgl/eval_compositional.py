#!/usr/bin/env python3
"""
Comprehensive evaluation script for compositional generalization experiments.
Updated to show detected vs ground truth properties in visualizations.
"""

import argparse
import os
import numpy as np

# Helper function to convert numpy types to native Python types for JSON serialization
def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Limit threaded math libraries to avoid shared memory issues on constrained systems
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_USE_SHMEM", "FALSE")

import json
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pathlib import Path
import csv
import itertools

# Import model and diffusion
from vgl.models_compositional import DiT_models_compositional as DiT_models
try:
    from unet_models_compositional import UNet_models_compositional as UNet_models
except ImportError:
    UNet_models = None
    print("Warning: UNet compositional models not found (unet_models_compositional).")

# Add SongUNet compositional support
try:
    from vgl.unet_models_song_compositional import CompositionalSongUNet_models as SongUNet_models
except ImportError:
    SongUNet_models = None
from vgl.diffusion import create_diffusion

SHAPE_NAME_TO_ID = {'circle': 0, 'square': 1, 'triangle': 2, 'diamond': 3}
COLOR_NAME_TO_ID = {
    'red': 0,
    'blue': 1,
    'green': 2,
    'yellow': 3,
    'magenta': 4,
    'cyan': 5,
    'orange': 6,
    'purple': 7,
}
COLOR_RGB_TO_ID = {
    (255, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (255, 255, 0): 3,
    (255, 0, 255): 4,
    (0, 255, 255): 5,
    (255, 128, 0): 6,
    (128, 0, 255): 7,
}

# Import shape generation for consistent ground truth
# Ground-truth renders must come from the same generator that built the dataset.
from scripts.generate_compositional_dataset_coverage import generate_shape_image
# Import proper rotation detection from eval_rotation.py
import math
try:
    from vgl.eval_rotation import (
        generate_ground_truth_arrow, 
        generate_ground_truth_triangle,
        extract_arrow_angle_template_matching,
        extract_triangle_angle_template_matching,
        calculate_rotation_metrics
    )
    ROTATION_TEMPLATE_MATCHING_AVAILABLE = True
    print("Imported template matching rotation detection from eval_rotation.py")
except ImportError:
    print("Warning: Could not import template matching from eval_rotation.py, using fallback rotation detection")
    ROTATION_TEMPLATE_MATCHING_AVAILABLE = False


def parse_checkpoint_path(ckpt_path):
    """Extract configuration from checkpoint path."""
    config = {
        'checkpoint_path': ckpt_path,
        'model_size': 'DiT-S/2',  
        'conditioning_method': 'concat',  
        'property_dropout_prob': 0.0,
        'architecture': 'dit'
    }
    
    # Check architecture
    if 'songunet' in ckpt_path.lower() or 'compsongunet' in ckpt_path.lower():
        config['architecture'] = 'songunet'
        config['model_size'] = 'CompSongUNet-S'
    elif 'unet' in ckpt_path.lower():
        config['architecture'] = 'unet'
        # Parse UNet model size
        if 'UNet-B' in ckpt_path or 'unet-b' in ckpt_path.lower():
            config['model_size'] = 'UNet-B'
        elif 'UNet-L' in ckpt_path or 'unet-l' in ckpt_path.lower():
            config['model_size'] = 'UNet-L'
        elif 'UNet-XL' in ckpt_path or 'unet-xl' in ckpt_path.lower():
            config['model_size'] = 'UNet-XL'
        else:
            config['model_size'] = 'UNet-S'
    else:
        # Parse DiT model size
        if 'DiT-B-2' in ckpt_path or 'DiT-B/2' in ckpt_path:
            config['model_size'] = 'DiT-B/2'
        elif 'DiT-L-2' in ckpt_path or 'DiT-L/2' in ckpt_path:
            config['model_size'] = 'DiT-L/2'
        elif 'DiT-XL-2' in ckpt_path or 'DiT-XL/2' in ckpt_path:
            config['model_size'] = 'DiT-XL/2'
    
    return config


def load_dataset_metadata(dataset_path):
    """Load dataset metadata to understand the compositional structure."""
    metadata_path = os.path.join(dataset_path, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    alt_metadata_path = os.path.join(dataset_path, 'metadata.json')
    if os.path.exists(alt_metadata_path):
        with open(alt_metadata_path, 'r') as f:
            return json.load(f)
    return None


def canonicalize_property_value(prop, value):
    """Convert metadata property values to canonical evaluation format."""
    if prop == 'shape':
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return SHAPE_NAME_TO_ID.get(value.lower(), 0)
    elif prop == 'color':
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return COLOR_NAME_TO_ID.get(value.lower(), 0)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return COLOR_RGB_TO_ID.get(tuple(value), 0)
    elif prop == 'count':
        return int(value)
    elif prop == 'radius':
        return float(value)
    elif prop == 'rotation':
        return float(value)
    elif prop == 'position':
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    return value


def get_property_domain(prop):
    """Return canonical domain of property values for binary compositional datasets."""
    if prop == 'shape':
        return [0, 1, 2, 3]
    if prop == 'color':
        return list(range(8))
    if prop == 'count':
        return [1, 2, 3, 4]
    if prop == 'radius':
        return [float(x) for x in range(6, 21)]
    return None


def build_metadata_based_combinations(metadata, include_properties):
    """Use dataset metadata to derive seen and unseen property combinations."""
    if not metadata or 'properties' not in metadata:
        return set(), set()
    
    training_set = set()
    metadata_combos = metadata.get('combinations', [])
    
    for entry in metadata_combos:
        combo = []
        for prop in include_properties:
            if prop in entry:
                combo.append(canonicalize_property_value(prop, entry[prop]))
            elif prop == 'color' and 'color_rgb' in entry:
                combo.append(canonicalize_property_value(prop, entry['color_rgb']))
            else:
                combo.append(None)
        training_set.add(tuple(combo))
    
    domains = []
    for idx, prop in enumerate(include_properties):
        domain = get_property_domain(prop)
        if domain is None or None in {combo[idx] for combo in training_set}:
            observed = sorted({combo[idx] for combo in training_set if combo[idx] is not None})
            domain = observed
        domains.append(domain)
    
    all_combinations = set()
    if all(domains):
        for values in itertools.product(*domains):
            all_combinations.add(tuple(values))
    
    unseen_combinations = all_combinations - training_set
    return training_set, unseen_combinations


def load_model(config, device, active_properties):
    """Load the model from checkpoint (DiT or UNet)."""
    architecture = config.get('architecture', 'dit')
    
    if architecture == 'songunet':
        assert SongUNet_models is not None, "SongUNet compositional models not available"
        # Detect active_properties order from checkpoint if possible
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        state_dict_ckpt = checkpoint.get('ema') or checkpoint.get('model') or checkpoint
        detected_props = []
        if isinstance(state_dict_ckpt, dict):
            for key in state_dict_ckpt.keys():
                if key.startswith('property_embedders.') and key.endswith('.weight'):
                    prop = key.split('.')[1]
                    if prop not in detected_props:
                        detected_props.append(prop)
        if detected_props:
            if set(detected_props) == set(active_properties):
                print(f"Using active_properties from checkpoint (preserving order): {detected_props}")
                active_properties = detected_props
            else:
                print(f"Warning: CLI active_properties {active_properties} differ from checkpoint props {detected_props}. Using checkpoint order where possible.")
                # Keep intersection in checkpoint order; append any extras at end
                ordered = [p for p in detected_props if p in active_properties]
                extras = [p for p in active_properties if p not in ordered]
                active_properties = ordered + extras
        model = SongUNet_models[config['model_size']](
            img_resolution=64,
            in_channels=3,
            out_channels=3,
            active_properties=active_properties,
            conditioning_method=config.get('conditioning_method', 'concat'),
            property_dropout_prob=0.0,
            num_shapes=4,
            num_colors=8,
        )
    elif architecture == 'unet':
        # Prepare property configs for UNet
        property_configs = {
            'radius': {'embedding_type': 'sinusoidal', 'radius_min': 1.0, 'radius_max': 5.0},
            'position': {'embedding_type': 'sinusoidal', 'max_position_value': 1.0},
            'rotation': {'embedding_type': 'circular'},
            'count': {'max_count': 10},
            'color': {'num_colors': 8},
            'shape': {'num_shapes': 4},
        }
        
        model = UNet_models[config['model_size']](
            input_size=64,
            in_channels=3,
            active_properties=active_properties,
            property_configs=property_configs,
            conditioning_method=config.get('conditioning_method', 'concat'),
            class_dropout_prob=0.0
        )
    else:
        model = DiT_models[config['model_size']](
            input_size=64,
            in_channels=3,
            conditioning_method=config.get('conditioning_method', 'concat'),
            property_dropout_prob=0.0,
            active_properties=active_properties
        )
    
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict (handle DiT/UNet with helper, SongUNet with standard loading)
    loaded = False
    if hasattr(model, 'load_state_dict_with_resize'):
        try:
            missing, unexpected = model.load_state_dict_with_resize(state_dict, strict=False)
            print("Loaded with dynamic pos_embed resizing")
            if missing:
                print(f"Missing keys: {len(missing)} (first 5): {missing[:5]}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
            loaded = True
        except Exception:
            pass
    if not loaded:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded with fallback method")
        if missing:
            print(f"Missing keys: {len(missing)} (first 5): {missing[:5]}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
    
    model = model.to(device)
    model.eval()
    
    return model


def parse_folder_name_comprehensive(folder_name, include_properties):
    """Parse folder name to extract all 6 property types."""
    parts = folder_name.split('_')
    properties = {}
    part_idx = 0
    
    try:
        if part_idx < len(parts) and parts[part_idx].startswith('r'):
            radius_str = parts[part_idx][1:].replace('p', '.')
            properties['radius'] = float(radius_str)
            part_idx += 1
        
        if (part_idx < len(parts) - 1 and 
            parts[part_idx].startswith('x') and parts[part_idx + 1].startswith('y')):
            x_str = parts[part_idx][1:].replace('n', '-').replace('p', '.')
            y_str = parts[part_idx + 1][1:].replace('n', '-').replace('p', '.')
            properties['position'] = (float(x_str), float(y_str))
            part_idx += 2
        
        shape_to_id = {'circle': 0, 'square': 1, 'triangle': 2, 'diamond': 3}
        if part_idx < len(parts) and parts[part_idx] in shape_to_id:
            properties['shape'] = shape_to_id[parts[part_idx]]
            part_idx += 1
        
        color_to_id = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3,
                      'magenta': 4, 'cyan': 5, 'orange': 6, 'purple': 7}
        if part_idx < len(parts) and parts[part_idx] in color_to_id:
            properties['color'] = color_to_id[parts[part_idx]]
            part_idx += 1
        
        if part_idx < len(parts) and parts[part_idx].startswith('cnt'):
            count_str = parts[part_idx][3:]
            properties['count'] = int(count_str)
            part_idx += 1
        
        if part_idx < len(parts) and parts[part_idx].startswith('rot'):
            rot_str = parts[part_idx][3:]
            properties['rotation'] = float(rot_str)
            part_idx += 1
        
        combo = []
        for prop in include_properties:
            if prop in properties:
                combo.append(properties[prop])
            else:
                if prop == 'radius':
                    combo.append(10.0)
                elif prop == 'position':
                    combo.append((0.0, 0.0))
                elif prop == 'shape':
                    combo.append(0)
                elif prop == 'color':
                    combo.append(0)
                elif prop == 'count':
                    combo.append(1)
                elif prop == 'rotation':
                    combo.append(0.0)
        
        return tuple(combo)
        
    except Exception as e:
        print(f"Warning: Could not parse folder '{folder_name}': {e}")
        return None


def load_test_combinations_comprehensive(dataset_path, include_properties, include_train_split=False):
    """Load test combinations from all splits in dataset."""
    combinations = {}
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return combinations

    metadata = load_dataset_metadata(dataset_path)
    metadata_seen, metadata_unseen = build_metadata_based_combinations(metadata, include_properties)
    if metadata_unseen:
        combinations['metadata_unseen'] = sorted(metadata_unseen)
    if include_train_split and metadata_seen:
        combinations['metadata_train'] = sorted(metadata_seen)
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and (item.startswith('test_') or item == 'train'):
            combinations[item] = []
            
            for folder_name in os.listdir(item_path):
                folder_path = os.path.join(item_path, folder_name)
                if os.path.isdir(folder_path):
                    combo = parse_folder_name_comprehensive(folder_name, include_properties)
                    if combo is not None:
                        combinations[item].append(combo)
    
    if not combinations:
        # Fallback: parse directory names directly (legacy datasets)
        direct_combos = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                combo = parse_folder_name_comprehensive(item, include_properties)
                if combo is not None:
                    direct_combos.append(combo)
        if direct_combos:
            combinations['dataset'] = direct_combos
    
    print(f"Loaded test combinations:")
    for split_name, combos in combinations.items():
        print(f"  {split_name}: {len(combos)} combinations")
    
    return combinations


def generate_samples_comprehensive(model, diffusion, test_combinations, include_properties, 
                                 device, num_samples=3, eval_batch_size=8):
    """Generate samples for test combinations with dynamic property handling."""
    samples_dict = {}
    
    print(f"Generating {num_samples} samples for {len(test_combinations)} combinations...")
    print(f"Active properties: {include_properties}")
    
    for combo_idx, combination in enumerate(tqdm(test_combinations, desc="Generating")):
        combo_key = tuple(combination)
        samples_dict[combo_key] = []
        
        total_generated = 0
        while total_generated < num_samples:
            current_batch = min(eval_batch_size, num_samples - total_generated)
            z = torch.randn(current_batch, 3, 64, 64, device=device)
            
            model_kwargs = {}
            for i, prop in enumerate(include_properties):
                if i >= len(combination):
                    continue
                prop_value = combination[i]
                
                if prop == 'radius':
                    model_kwargs['radius'] = torch.full(
                        (current_batch,), float(prop_value), device=device, dtype=torch.float32
                    )
                elif prop == 'position':
                    pos_tensor = torch.tensor(prop_value, device=device, dtype=torch.float32)
                    if pos_tensor.dim() == 1:
                        pos_tensor = pos_tensor.unsqueeze(0)
                    model_kwargs['position'] = pos_tensor.repeat(current_batch, 1)
                elif prop == 'shape':
                    model_kwargs['shape'] = torch.full(
                        (current_batch,), int(prop_value), device=device, dtype=torch.long
                    )
                elif prop == 'color':
                    model_kwargs['color'] = torch.full(
                        (current_batch,), int(prop_value), device=device, dtype=torch.long
                    )
                elif prop == 'count':
                    model_kwargs['count'] = torch.full(
                        (current_batch,), int(prop_value), device=device, dtype=torch.long
                    )
                elif prop == 'rotation':
                    model_kwargs['rotation'] = torch.full(
                        (current_batch,), float(prop_value), device=device, dtype=torch.float32
                    )
            
            with torch.no_grad():
                samples = diffusion.p_sample_loop(
                    model, z.shape, z, clip_denoised=True,
                    model_kwargs=model_kwargs, progress=False
                )
            
            for b in range(current_batch):
                samples_dict[combo_key].append(samples[b])
            total_generated += current_batch
    
    return samples_dict


def evaluate_properties_comprehensive(image, expected_properties, include_properties, debug=False):
    """Evaluate all properties in a generated image."""
    metrics = {
        'detection_success': False,
        'detected_values': {}
    }
    
    for prop in ['radius', 'position', 'shape', 'color', 'count', 'rotation']:
        if prop in include_properties:
            if prop in ['radius', 'position', 'rotation']:
                metrics[f'{prop}_error'] = float('inf')
            else:
                metrics[f'{prop}_accuracy'] = 0.0
    
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
    else:
        image_np = np.array(image)
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Use Otsu thresholding on grayscale - adapts to each image's actual foreground/background
    # This handles noisy backgrounds in canonical datasets better than fixed RGB distance.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert if foreground got classified as white (we want object=white)
    if np.mean(binary) > 127:
        binary = 255 - binary
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # For count detection, we need to handle overlapping objects
    # Use distance transform + watershed to separate touching objects
    expected_count = None
    for i, prop in enumerate(include_properties):
        if prop == 'count' and i < len(expected_properties):
            expected_count = expected_properties[i]
            break
    
    if expected_count and expected_count > 1 and 'count' in include_properties:
        # Use watershed to separate touching objects (shape-agnostic)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

        detected_count = 0
        if binary_clean.max() > 0:
            dist = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            num_markers, markers = cv2.connectedComponents(sure_fg)
            if num_markers > 1:
                sure_bg = cv2.dilate(binary_clean, kernel, iterations=2)
                unknown = cv2.subtract(sure_bg, sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                watershed_input = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
                cv2.watershed(watershed_input, markers)
                for label in np.unique(markers):
                    if label <= 1:
                        continue
                    if int(np.count_nonzero(markers == label)) > 30:
                        detected_count += 1

        metrics['count_accuracy'] = 1.0 if detected_count == expected_count else 0.0
        metrics['detection_success'] = detected_count > 0
        metrics['detected_values']['count'] = detected_count
        
        # For other properties, still use the main contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            main_contour = contours[0]
        else:
            metrics['detected_values'] = {prop: None for prop in include_properties}
            return metrics
    else:
        # Original logic for non-count or count=1 cases
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            metrics['detected_values'] = {prop: None for prop in include_properties}
            return metrics
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = contours[0]
        metrics['detection_success'] = True
        
        if 'count' in include_properties:
            detected_count = len([c for c in contours if cv2.contourArea(c) > 10])
            metrics['count_accuracy'] = 1.0 if detected_count == 1 else 0.0
            metrics['detected_values']['count'] = detected_count
    
    area = cv2.contourArea(main_contour)
    
    # 1. Radius evaluation
    if 'radius' in include_properties:
        expected_radius = None
        for i, prop in enumerate(include_properties):
            if prop == 'radius' and i < len(expected_properties):
                expected_radius = expected_properties[i]
                break
        
        if expected_radius is not None:
            detected_radius = np.sqrt(area / np.pi)
            metrics['radius_error'] = abs(detected_radius - expected_radius) / expected_radius
            metrics['detected_values']['radius'] = detected_radius
    
    # 2. Position evaluation
    if 'position' in include_properties:
        expected_position = None
        for i, prop in enumerate(include_properties):
            if prop == 'position' and i < len(expected_properties):
                expected_position = expected_properties[i]
                break
        
        if expected_position is not None:
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                image_center = image_np.shape[0] // 2
                detected_pos = (cx - image_center, cy - image_center)
                
                pos_error = np.sqrt((detected_pos[0] - expected_position[0])**2 + 
                                  (detected_pos[1] - expected_position[1])**2)
                metrics['position_error'] = pos_error / image_center
                metrics['detected_values']['position'] = detected_pos
    
    # 3. Shape evaluation
    if 'shape' in include_properties:
        expected_shape = None
        for i, prop in enumerate(include_properties):
            if prop == 'shape' and i < len(expected_properties):
                expected_shape = expected_properties[i]
                break
        
        if expected_shape is not None:
            shape_names = ['circle', 'square', 'triangle', 'diamond']
            expected_shape_name = shape_names[expected_shape] if isinstance(expected_shape, int) else expected_shape

            # Template-IoU shape classification with centroid alignment.
            # For each candidate shape, we generate templates at multiple sizes,
            # center-align against the mask, and pick the shape with highest IoU.
            gen_mask = (binary > 0).astype(np.uint8)
            gen_ys, gen_xs = np.where(gen_mask > 0)
            if len(gen_xs) < 3:
                detected_shape_id = 0
                detected_shape = 'circle'
            else:
                gen_cy, gen_cx = int(gen_ys.mean()), int(gen_xs.mean())
                shift_M = np.float32([[1, 0, 32 - gen_cx], [0, 1, 32 - gen_cy]])
                centered = cv2.warpAffine(gen_mask, shift_M, (64, 64))

                best_shape_id = 0
                best_iou = -1.0
                # VGL_SHAPE_METRIC=v2 selects the component-wise, in-place matcher that reads ground
                # truth on every dataset (vgl/shape_metric_v2.py). Default is the published matcher.
                _use_v2 = os.environ.get("VGL_SHAPE_METRIC", "locked") == "v2"
                if _use_v2:
                    from vgl.shape_metric_v2 import classify_shape as _classify_v2
                    _v2_name = _classify_v2(image)
                    best_shape_id = shape_names.index(_v2_name) if _v2_name in shape_names else 0
                    best_iou = 1.0
                # Imported outside the try so a missing module fails loudly instead
                # of silently degrading the shape metric to the contour fallback.
                from scripts.generate_compositional_dataset_coverage import generate_shape_image
                try:
                    for sh_id, sh_name in enumerate(shape_names):
                        if _use_v2:
                            break
                        for size in [8, 10, 12, 14, 16, 18]:
                            t = generate_shape_image(radius=size, position=(0, 0), shape=sh_name,
                                                    color_rgb=(255, 0, 0), image_size=64, rotation=0, count=1)
                            t_np = np.array(t)
                            t_white = np.array([255, 255, 255], dtype=np.float32)
                            t_diff = np.linalg.norm(t_np.astype(np.float32) - t_white, axis=2)
                            t_mask = (t_diff > 30).astype(np.uint8)
                            inter = np.logical_and(centered > 0, t_mask > 0).sum()
                            union = np.logical_or(centered > 0, t_mask > 0).sum()
                            iou = inter / union if union > 0 else 0.0
                            if iou > best_iou:
                                best_iou = iou
                                best_shape_id = sh_id
                except Exception:
                    # Fallback to contour-based if template generation fails
                    perimeter = cv2.arcLength(main_contour, True)
                    approx = cv2.approxPolyDP(main_contour, 0.02 * perimeter, True)
                    best_shape_id = 0
                    if len(approx) == 3:
                        best_shape_id = 2

                detected_shape_id = best_shape_id
                detected_shape = shape_names[best_shape_id]

            metrics['shape_accuracy'] = 1.0 if detected_shape == expected_shape_name else 0.0
            metrics['detected_values']['shape'] = detected_shape_id
    
    # 4. Color evaluation
    if 'color' in include_properties:
        expected_color = None
        for i, prop in enumerate(include_properties):
            if prop == 'color' and i < len(expected_properties):
                expected_color = expected_properties[i]
                break
        
        if expected_color is not None:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [main_contour], -1, 255, -1)
            mean_color = cv2.mean(image_np, mask=mask)[:3]
            
            color_palette = {
                0: [255, 0, 0],    # red
                1: [0, 0, 255],    # blue
                2: [0, 255, 0],    # green
                3: [255, 255, 0],  # yellow
                4: [255, 0, 255],  # magenta
                5: [0, 255, 255],  # cyan
                6: [255, 128, 0],  # orange
                7: [128, 0, 255],  # purple
            }
            
            min_distance = float('inf')
            predicted_color_id = 0
            
            for color_id, color_rgb in color_palette.items():
                distance = np.linalg.norm(np.array(mean_color) - np.array(color_rgb))
                if distance < min_distance:
                    min_distance = distance
                    predicted_color_id = color_id
            
            metrics['color_accuracy'] = 1.0 if predicted_color_id == expected_color else 0.0
            metrics['detected_values']['color'] = predicted_color_id
    
    # 5. Rotation evaluation using the exact function from eval_rotation.py
    if 'rotation' in include_properties:
        expected_rotation = None
        for i, prop in enumerate(include_properties):
            if prop == 'rotation' and i < len(expected_properties):
                expected_rotation = expected_properties[i]
                break
        
        if expected_rotation is not None:
            if ROTATION_TEMPLATE_MATCHING_AVAILABLE:
                # Use the exact calculate_rotation_metrics function from eval_rotation.py
                # Determine shape type - for rotation experiments, typically use triangle
                shape_type = 'triangle'  # Default to triangle as it's more reliable
                rotation_metrics = calculate_rotation_metrics(
                    image_np, expected_rotation, shape_type, image_size=64
                )
                
                if rotation_metrics['detection_success']:
                    # Convert from degrees to normalized [0,1] range for consistency with other metrics
                    metrics['rotation_error'] = rotation_metrics['angle_error_abs'] / 180.0
                    metrics['detected_values']['rotation'] = rotation_metrics['detected_angle']
                else:
                    metrics['rotation_error'] = 1.0  # Maximum error for failed detection
                    metrics['detected_values']['rotation'] = None
            else:
                # Fallback to basic cv2 approach
                rect = cv2.minAreaRect(main_contour)
                detected_angle = rect[2]
                
                if detected_angle < 0:
                    detected_angle += 90
                
                angle_diff = abs(detected_angle - expected_rotation)
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                metrics['rotation_error'] = angle_diff / 180.0
                metrics['detected_values']['rotation'] = detected_angle
    
    return metrics


def evaluate_split_comprehensive(split_combinations, samples_dict, include_properties, split_name):
    """Evaluate a split with comprehensive property handling."""
    results = []
    
    print(f"Evaluating {len(split_combinations)} {split_name} combinations...")
    
    for combo in tqdm(split_combinations, desc=f"Evaluating {split_name}"):
        combo_key = tuple(combo)
        
        if combo_key not in samples_dict:
            continue
        
        combo_results = []
        
        for sample in samples_dict[combo_key]:
            metrics = evaluate_properties_comprehensive(sample, combo, include_properties)
            combo_results.append(metrics)
        
        if combo_results:
            avg_metrics = {}
            for key in combo_results[0].keys():
                if key == 'detection_success':
                    avg_metrics[key] = np.mean([r[key] for r in combo_results])
                elif key == 'detected_values':
                    # Keep the detected values from the first successful detection
                    for result in combo_results:
                        if result['detection_success']:
                            avg_metrics[key] = result[key]
                            break
                    if key not in avg_metrics:
                        avg_metrics[key] = combo_results[0][key]
                else:
                    successful_values = [r[key] for r in combo_results if r['detection_success']]
                    if successful_values:
                        avg_metrics[key] = np.mean(successful_values)
                    else:
                        avg_metrics[key] = float('inf') if 'error' in key else 0.0
            
            for i, prop in enumerate(include_properties):
                if i < len(combo):
                    avg_metrics[f'expected_{prop}'] = combo[i]
            
            avg_metrics['split_name'] = split_name
            results.append(avg_metrics)
    
    return results


def calculate_split_summary(results, split_name):
    """Calculate summary statistics for a split."""
    if not results:
        return {}
    
    detection_values = [1.0 if r.get('detection_success') else 0.0 for r in results]
    detection_rate = float(np.mean(detection_values)) if detection_values else 0.0

    summary = {
        'split_name': split_name,
        'num_combinations': len(results),
        'detection_rate': detection_rate
    }
    
    metric_keys = set()
    for result in results:
        metric_keys.update(result.keys())
    
    for metric_col in metric_keys:
        if metric_col.endswith('_error') or metric_col.endswith('_accuracy'):
            successful_values = []
            for result in results:
                if result.get('detection_success'):
                    value = result.get(metric_col)
                    if isinstance(value, (int, float, np.floating, np.integer)) and not math.isnan(value):
                        successful_values.append(float(value))
            if successful_values:
                summary[f'{metric_col}_mean'] = float(np.mean(successful_values))
                summary[f'{metric_col}_std'] = float(np.std(successful_values))
    
    return summary


def generate_ground_truth_comprehensive(properties_dict, image_size=64, antialiasing=4):
    """Generate ground truth image for given compositional properties."""
    radius = properties_dict.get('radius', 10.0)
    position = properties_dict.get('position', (0.0, 0.0))
    shape = properties_dict.get('shape', 'circle')
    color = properties_dict.get('color', 0)
    count = properties_dict.get('count', 1)
    rotation = properties_dict.get('rotation', 0.0)
    
    # IMPORTANT: Use triangle for rotation experiments
    if 'rotation' in properties_dict and shape == 0:
        shape = 'triangle'
    
    if isinstance(shape, int):
        shape_names = ['circle', 'square', 'triangle', 'diamond']
        shape = shape_names[shape]
    
    color_palette = {
        0: (255, 0, 0),    # red
        1: (0, 0, 255),    # blue
        2: (0, 255, 0),    # green
        3: (255, 255, 0),  # yellow
        4: (255, 0, 255),  # magenta
        5: (0, 255, 255),  # cyan
        6: (255, 128, 0),  # orange
        7: (128, 0, 255),  # purple
    }
    color_rgb = color_palette.get(color, (255, 0, 0))
    
    count_is_active = count > 1
    
    img = generate_shape_image(
        radius, position, shape, color_rgb,
        image_size=image_size, antialiasing=antialiasing,
        rotation=rotation, count=count, count_is_active_property=count_is_active
    )
    
    return img


def create_qualitative_comparison_grid(samples_dict, test_combinations, include_properties, 
                                     output_path, split_name, num_examples=6):
    """Create side-by-side comparison with detected values."""
    
    selected_combinations = test_combinations[:num_examples]
    
    fig, axes = plt.subplots(3, num_examples, figsize=(4*num_examples, 12))
    if num_examples == 1:
        axes = axes.reshape(3, 1)
    
    for i, combination in enumerate(selected_combinations):
        combo_key = tuple(combination)
        
        properties_dict = {}
        for j, prop in enumerate(include_properties):
            if j < len(combination):
                properties_dict[prop] = combination[j]
        
        # Row 1: Generated image
        detected_props = None
        if combo_key in samples_dict and len(samples_dict[combo_key]) > 0:
            sample = samples_dict[combo_key][0]
            
            # Evaluate the sample to get detected values
            metrics = evaluate_properties_comprehensive(sample, combination, include_properties)
            detected_props = metrics.get('detected_values', {})
            
            if isinstance(sample, torch.Tensor):
                if sample.dim() == 4:
                    sample = sample.squeeze(0)
                if sample.shape[0] == 3:
                    sample = sample.permute(1, 2, 0)
                sample = (sample + 1) / 2
                sample = torch.clamp(sample, 0, 1)
                sample_np = sample.cpu().numpy()
            else:
                sample_np = np.array(sample) / 255.0
            
            axes[0, i].imshow(sample_np)
            axes[0, i].set_title('Generated', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
        else:
            axes[0, i].text(0.5, 0.5, 'No Sample', ha='center', va='center', 
                           transform=axes[0, i].transAxes, fontsize=12)
            axes[0, i].axis('off')
        
        # Row 2: Ground truth image
        try:
            gt_img = generate_ground_truth_comprehensive(properties_dict)
            gt_np = np.array(gt_img) / 255.0
            axes[1, i].imshow(gt_np)
            axes[1, i].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, f'GT Error:\n{str(e)[:20]}...', ha='center', va='center',
                           transform=axes[1, i].transAxes, fontsize=10)
            axes[1, i].axis('off')
        
        # Row 3: Property comparison (expected vs detected)
        axes[2, i].axis('off')
        
        # Create comparison text
        comparison_text = []
        shape_names = ['circle', 'square', 'triangle', 'diamond']
        color_names = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
        
        for j, prop in enumerate(include_properties):
            if j < len(combination):
                expected_value = combination[j]
                
                # Format expected value
                if prop == 'radius':
                    expected_str = f"{expected_value:.1f}"
                elif prop == 'position':
                    if isinstance(expected_value, (tuple, list)) and len(expected_value) == 2:
                        expected_str = f"({expected_value[0]:.1f}, {expected_value[1]:.1f})"
                    else:
                        expected_str = str(expected_value)
                elif prop == 'shape':
                    expected_str = shape_names[expected_value] if isinstance(expected_value, int) and expected_value < len(shape_names) else str(expected_value)
                elif prop == 'color':
                    expected_str = color_names[expected_value] if isinstance(expected_value, int) and expected_value < len(color_names) else str(expected_value)
                elif prop == 'count':
                    expected_str = str(expected_value)
                elif prop == 'rotation':
                    expected_str = f"{expected_value:.0f}°"
                else:
                    expected_str = str(expected_value)
                
                # Format detected value
                if detected_props and prop in detected_props:
                    detected_value = detected_props[prop]
                    
                    if detected_value is None:
                        detected_str = "NOT DETECTED"
                    elif prop == 'radius':
                        detected_str = f"{detected_value:.1f}"
                    elif prop == 'position':
                        if isinstance(detected_value, (tuple, list)) and len(detected_value) == 2:
                            detected_str = f"({detected_value[0]:.1f}, {detected_value[1]:.1f})"
                        else:
                            detected_str = str(detected_value)
                    elif prop == 'shape':
                        detected_str = shape_names[detected_value] if isinstance(detected_value, int) and detected_value < len(shape_names) else str(detected_value)
                    elif prop == 'color':
                        detected_str = color_names[detected_value] if isinstance(detected_value, int) and detected_value < len(color_names) else str(detected_value)
                    elif prop == 'count':
                        detected_str = str(detected_value)
                    elif prop == 'rotation':
                        detected_str = f"{detected_value:.0f}°"
                    else:
                        detected_str = str(detected_value)
                else:
                    detected_str = "NOT DETECTED"
                
                # Create comparison line
                prop_display = prop.capitalize()
                comparison_text.append(f"{prop_display}:\n  Expected: {expected_str}\n  Detected: {detected_str}")
        
        # Display comparison text
        comparison_str = '\n'.join(comparison_text)
        axes[2, i].text(0.5, 0.5, comparison_str, ha='center', va='center',
                       transform=axes[2, i].transAxes, fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Qualitative Comparison: {split_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved qualitative comparison: {output_path}")


def save_comprehensive_results(all_results, all_summaries, output_dir, config, include_properties):
    """Save comprehensive results including qualitative comparisons."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results_path = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_results_path, 'w') as f:
        serializable_results = {}
        for split_name, results in all_results.items():
            serializable_results[split_name] = []
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_result[key] = float(value)
                    elif isinstance(value, (int, np.int64)):
                        serializable_result[key] = int(value)
                    elif isinstance(value, (float, np.floating)) and math.isnan(value):
                        serializable_result[key] = None
                    else:
                        serializable_result[key] = value
                serializable_results[split_name].append(serializable_result)
        
        # Use convert_to_serializable to handle all numpy types
        json.dump(convert_to_serializable({
            'config': config,
            'include_properties': include_properties,
            'results': serializable_results
        }), f, indent=2)
    
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    if all_summaries:
        fieldnames = sorted({key for summary in all_summaries for key in summary.keys()})
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for summary in all_summaries:
                row = {}
                for key in fieldnames:
                    value = summary.get(key, "")
                    if isinstance(value, (np.integer, int)):
                        row[key] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        row[key] = float(value)
                    else:
                        row[key] = value
                writer.writerow(row)
    else:
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
    
    performance_summary = {
        'experiment_name': os.path.basename(output_dir),
        'include_properties': include_properties,
        'num_properties': len(include_properties),
        'property_types': {
            'continuous': [p for p in include_properties if p in ['radius', 'position', 'rotation', 'count']],
            'discrete': [p for p in include_properties if p in ['shape', 'color']]
        },
        'splits_evaluated': list(all_results.keys()),
        'overall_performance': {}
    }
    
    for summary in all_summaries:
        split_name = summary['split_name']
        performance_summary['overall_performance'][split_name] = {
            'detection_rate': summary.get('detection_rate', 0),
            'num_combinations': summary.get('num_combinations', 0)
        }
        
        for prop in include_properties:
            if prop in ['radius', 'position', 'rotation']:
                error_key = f'{prop}_error_mean'
                if error_key in summary:
                    performance_summary['overall_performance'][split_name][f'{prop}_error'] = summary[error_key]
            else:
                acc_key = f'{prop}_accuracy_mean'
                if acc_key in summary:
                    performance_summary['overall_performance'][split_name][f'{prop}_accuracy'] = summary[acc_key]
    
    performance_path = os.path.join(output_dir, 'performance_summary.json')
    with open(performance_path, 'w') as f:
        # Convert numpy types to native Python types before JSON serialization
        json.dump(convert_to_serializable(performance_summary), f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"  - Detailed results: detailed_results.json")
    print(f"  - Summary statistics: summary_statistics.csv")
    print(f"  - Performance summary: performance_summary.json")
    print(f"  - Qualitative comparisons: qualitative_*.png")


def print_results_table_comprehensive(summaries):
    """Print a formatted results table for all splits."""
    print(f"\n{'='*120}")
    print(f"COMPREHENSIVE COMPOSITIONAL EVALUATION RESULTS")
    print(f"{'='*120}")
    
    header = f"{'Split':<20} {'N':<5} {'Det%':<6}"
    
    available_metrics = set()
    for summary in summaries:
        for key in summary.keys():
            if key.endswith('_mean'):
                available_metrics.add(key.replace('_mean', ''))
    
    metric_order = ['radius_error', 'position_error', 'rotation_error', 
                   'shape_accuracy', 'color_accuracy', 'count_accuracy']
    
    for metric in metric_order:
        if metric in available_metrics:
            if 'error' in metric:
                header += f" {metric.replace('_error', 'Err'):<8}"
            else:
                header += f" {metric.replace('_accuracy', 'Acc'):<8}"
    
    print(header)
    print("-" * len(header))
    
    for summary in summaries:
        row = f"{summary.get('split_name', 'unknown'):<20} "
        row += f"{summary.get('num_combinations', 0):<5} "
        row += f"{summary.get('detection_rate', 0):.3f}  "
        
        for metric in metric_order:
            if metric in available_metrics:
                mean_key = f'{metric}_mean'
                value = summary.get(mean_key, 0)
                if np.isnan(value):
                    row += f"{'nan':<8}"
                else:
                    row += f"{value:.4f}  "
        
        print(row)
    
    print("-" * len(header))


def main(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = parse_checkpoint_path(args.checkpoint)
    print(f"Model: {config['model_size']}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print(f"Active properties: {args.include_properties}")
    
    print("Loading model...")
    model = load_model(config, device, args.include_properties)
    if config.get('architecture') == 'songunet':
        diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))
    
    print("Loading test combinations...")
    combinations = load_test_combinations_comprehensive(
        args.dataset_path, args.include_properties, include_train_split=args.include_train_split
    )
    if not args.include_train_split:
        if 'train' in combinations:
            print("Skipping training split for evaluation.")
            combinations.pop('train', None)
        if 'metadata_train' in combinations:
            print("Skipping metadata-derived training combinations.")
            combinations.pop('metadata_train', None)
    
    # Optional filtering: restrict radius extrapolation values for prop_a OOD
    if hasattr(args, 'limit_radius_extrap') and args.include_properties and args.include_properties[0] == 'radius':
        radius_filter = None
        try:
            if args.limit_radius_extrap:
                radius_filter = set([float(x.strip()) for x in args.limit_radius_extrap.split(',') if x.strip()])
        except Exception as e:
            print(f"Warning: could not parse --limit-radius-extrap: {e}")
            radius_filter = None
        if radius_filter and 'test_prop_a_ood' in combinations:
            before = len(combinations['test_prop_a_ood'])
            filtered = []
            for combo in combinations['test_prop_a_ood']:
                if len(combo) > 0 and combo[0] in radius_filter:
                    filtered.append(combo)
            combinations['test_prop_a_ood'] = filtered
            after = len(filtered)
            print(f"Applied radius extrapolation filter to prop_a OOD: kept {after}/{before} combos ({sorted(radius_filter)})")
    
    if not combinations:
        print("No test combinations found!")
        return
    
    metadata = load_dataset_metadata(args.dataset_path)
    if metadata:
        print(f"Dataset type: {metadata.get('composition_type', 'unknown')}")
        print(f"Coverage: {metadata.get('coverage', 'unknown')}")
    
    # Optional: add custom prop_a OOD for radius with specific values crossed with colors
    if (hasattr(args, 'custom_radius_prop_a_ood') and args.custom_radius_prop_a_ood 
        and args.include_properties and args.include_properties[0] == 'radius'
        and len(args.include_properties) >= 2 and args.include_properties[1] == 'color'):
        try:
            custom_radii = [float(x.strip()) for x in args.custom_radius_prop_a_ood.split(',') if x.strip()]
            # Colors: 8-class palette [0..7]
            color_ids = list(range(8))
            custom_combos = []
            for r in custom_radii:
                for c in color_ids:
                    custom_combos.append((r, c))
            combinations['custom_prop_a_ood'] = custom_combos
            print(f"Added custom_prop_a_ood combinations for radius x color: {len(custom_combos)} (r in {custom_radii} × 8 colors)")
        except Exception as e:
            print(f"Warning: could not parse --custom-radius-prop-a-ood: {e}")
    
    # Optional: add custom prop_a OOD for position with specific values crossed with shapes
    if (hasattr(args, 'custom_position_prop_a_ood') and args.custom_position_prop_a_ood 
        and args.include_properties and args.include_properties[0] == 'position'
        and len(args.include_properties) >= 2 and args.include_properties[1] == 'shape'):
        try:
            custom_positions_str = [x.strip() for x in args.custom_position_prop_a_ood.split(',') if x.strip()]
            custom_positions = []
            for pos_str in custom_positions_str:
                if ',' in pos_str:
                    # Handle (x,y) format
                    pos_str = pos_str.strip('()')
                    x, y = pos_str.split(',')
                    custom_positions.append((float(x.strip()), float(y.strip())))
                else:
                    # Handle single value - use for both x and y
                    val = float(pos_str)
                    custom_positions.append((val, val))
            
            # Shapes: 4 shapes [0..3]
            shape_ids = list(range(4))
            custom_combos = []
            for pos in custom_positions:
                for s in shape_ids:
                    custom_combos.append((pos, s))
            combinations['custom_prop_a_ood'] = custom_combos
            print(f"Added custom_prop_a_ood combinations for position x shape: {len(custom_combos)} (pos in {custom_positions} × 4 shapes)")
        except Exception as e:
            print(f"Warning: could not parse --custom-position-prop-a-ood: {e}")
    
    # Optional: add custom prop_a OOD and prop_b OOD for radius-rotation
    if (hasattr(args, 'custom_radius_prop_a_ood') and args.custom_radius_prop_a_ood 
        and hasattr(args, 'custom_rotation_prop_b_ood') and args.custom_rotation_prop_b_ood
        and args.include_properties and len(args.include_properties) >= 2
        and args.include_properties[0] == 'radius' and args.include_properties[1] == 'rotation'):
        try:
            custom_radii = [float(x.strip()) for x in args.custom_radius_prop_a_ood.split(',') if x.strip()]
            custom_rotations = [float(x.strip()) for x in args.custom_rotation_prop_b_ood.split(',') if x.strip()]
            
            # Create custom_prop_a_ood (radius OOD x all training rotations)
            # For simplicity, use a few representative training rotation values
            training_rotations = [0.0, 90.0, 180.0, 270.0]  # Representative training rotations
            custom_prop_a_combos = []
            for r in custom_radii:
                for rot in training_rotations:
                    custom_prop_a_combos.append((r, rot))
            combinations['custom_prop_a_ood'] = custom_prop_a_combos
            print(f"Added custom_prop_a_ood combinations for radius x rotation: {len(custom_prop_a_combos)} (r in {custom_radii} × {len(training_rotations)} rotations)")
            
            # Create custom_prop_b_ood (all training radii x rotation OOD)
            # For simplicity, use a few representative training radius values
            training_radii = [5.0, 10.0, 15.0, 20.0]  # Representative training radii
            custom_prop_b_combos = []
            for r in training_radii:
                for rot in custom_rotations:
                    custom_prop_b_combos.append((r, rot))
            combinations['custom_prop_b_ood'] = custom_prop_b_combos
            print(f"Added custom_prop_b_ood combinations for radius x rotation: {len(custom_prop_b_combos)} ({len(training_radii)} radii × rot in {custom_rotations})")
            
            # Create custom_both_ood (radius OOD x rotation OOD)
            custom_both_combos = []
            for r in custom_radii:
                for rot in custom_rotations:
                    custom_both_combos.append((r, rot))
            combinations['custom_both_ood'] = custom_both_combos
            print(f"Added custom_both_ood combinations for radius x rotation: {len(custom_both_combos)} (r in {custom_radii} × rot in {custom_rotations})")
            
        except Exception as e:
            print(f"Warning: could not parse custom radius/rotation OOD arguments: {e}")
    
    # Optional: add custom interpolation and extrapolation for position-shape
    if (hasattr(args, 'custom_position_interp') and args.custom_position_interp 
        and hasattr(args, 'custom_position_extrap') and args.custom_position_extrap
        and args.include_properties and len(args.include_properties) >= 2
        and args.include_properties[0] == 'position' and args.include_properties[1] == 'shape'):
        try:
            # Parse interpolation positions
            interp_positions_str = [x.strip() for x in args.custom_position_interp.split(',') if x.strip()]
            interp_positions = []
            for pos_str in interp_positions_str:
                val = float(pos_str)
                interp_positions.append((val, val))  # Use same value for both x and y
            
            # Parse extrapolation positions  
            extrap_positions_str = [x.strip() for x in args.custom_position_extrap.split(',') if x.strip()]
            extrap_positions = []
            for pos_str in extrap_positions_str:
                val = float(pos_str)
                extrap_positions.append((val, val))  # Use same value for both x and y
            
            # Shapes: 4 shapes [0..3]
            shape_ids = list(range(4))
            
            # Create custom interpolation split (interp positions x all shapes)
            custom_interp_combos = []
            for pos in interp_positions:
                for s in shape_ids:
                    custom_interp_combos.append((pos, s))
            combinations['custom_interpolation'] = custom_interp_combos
            print(f"Added custom_interpolation combinations for position x shape: {len(custom_interp_combos)} (pos in {interp_positions} × 4 shapes)")
            
            # Create custom extrapolation split (extrap positions x all shapes)
            custom_extrap_combos = []
            for pos in extrap_positions:
                for s in shape_ids:
                    custom_extrap_combos.append((pos, s))
            combinations['custom_extrapolation'] = custom_extrap_combos
            print(f"Added custom_extrapolation combinations for position x shape: {len(custom_extrap_combos)} (pos in {extrap_positions} × 4 shapes)")
            
        except Exception as e:
            print(f"Warning: could not parse custom position interp/extrap arguments: {e}")
    
    # Optional: add custom radius-position grid evaluation
    if (hasattr(args, 'custom_radius_position_grid') and args.custom_radius_position_grid 
        and hasattr(args, 'custom_radius_values') and args.custom_radius_values
        and hasattr(args, 'custom_position_values') and args.custom_position_values
        and args.include_properties and len(args.include_properties) >= 2
        and args.include_properties[0] == 'radius' and args.include_properties[1] == 'position'):
        try:
            # Parse radius values
            radius_values = [float(x.strip()) for x in args.custom_radius_values.split(',') if x.strip()]
            
            # Parse position values
            position_values_str = [x.strip() for x in args.custom_position_values.split(',') if x.strip()]
            position_values = []
            for pos_str in position_values_str:
                val = float(pos_str)
                position_values.append((val, val))  # Use same value for both x and y
            
            # Create full grid (all radius x all position combinations)
            custom_grid_combos = []
            for r in radius_values:
                for pos in position_values:
                    custom_grid_combos.append((r, pos))
            
            combinations['custom_radius_position_grid'] = custom_grid_combos
            print(f"Added custom_radius_position_grid combinations: {len(custom_grid_combos)} (radius in {radius_values} × position in {position_values})")
            
        except Exception as e:
            print(f"Warning: could not parse custom radius-position grid arguments: {e}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    all_summaries = []
    all_samples = {}
    
    for split_name, split_combinations in combinations.items():
        if not split_combinations:
            continue
        
        print(f"\nProcessing {split_name}...")
        
        if args.max_combinations_per_split and args.max_combinations_per_split > 0:
            split_combinations = split_combinations[:args.max_combinations_per_split]
        
        samples_dict = generate_samples_comprehensive(
            model, diffusion, split_combinations, args.include_properties, 
            device, args.num_samples, args.eval_batch_size
        )
        
        results = evaluate_split_comprehensive(
            split_combinations, samples_dict, args.include_properties, split_name
        )
        
        summary = calculate_split_summary(results, split_name)
        
        all_results[split_name] = results
        all_summaries.append(summary)
        all_samples[split_name] = (samples_dict, split_combinations)
        
        qualitative_output_path = os.path.join(args.output_dir, f'qualitative_{split_name}.png')
        create_qualitative_comparison_grid(
            samples_dict, split_combinations, args.include_properties,
            qualitative_output_path, split_name, num_examples=min(6, len(split_combinations))
        )
    
    print_results_table_comprehensive(all_summaries)
    
    save_comprehensive_results(all_results, all_summaries, args.output_dir, config, args.include_properties)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of compositional generalization')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to compositional dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--include-properties', type=str, nargs='+',
                       choices=['radius', 'position', 'shape', 'color', 'count', 'rotation'],
                       required=True,
                       help='Properties to evaluate')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples per combination')
    parser.add_argument('--num-sampling-steps', type=int, default=250,
                       help='Number of diffusion sampling steps')
    parser.add_argument('--max-combinations-per-split', type=int, default=0,
                       help='Maximum combinations to evaluate per split (0 = no limit)')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                       help='Number of samples to generate in parallel per combination')
    parser.add_argument('--include-train-split', action='store_true',
                       help='Include the training split in evaluation')
    parser.add_argument('--limit-radius-extrap', type=str, default='',
                       help='Comma-separated radius values to keep for prop_a OOD when first property is radius (e.g., 20,25,30)')
    parser.add_argument('--custom-radius-prop-a-ood', type=str, default='',
                       help='Comma-separated radius values to evaluate as a custom prop_a OOD split (only for radius-color)')
    parser.add_argument('--custom-position-prop-a-ood', type=str, default='',
                       help='Comma-separated position values to evaluate as a custom prop_a OOD split (only for position-shape/color)')
    parser.add_argument('--custom-rotation-prop-b-ood', type=str, default='',
                       help='Comma-separated rotation values to evaluate as a custom prop_b OOD split (only for radius-rotation)')
    parser.add_argument('--custom-position-interp', type=str, default='',
                       help='Comma-separated position values to evaluate as custom interpolation split (only for position-shape)')
    parser.add_argument('--custom-position-extrap', type=str, default='',
                       help='Comma-separated position values to evaluate as custom extrapolation split (only for position-shape)')
    parser.add_argument('--custom-radius-position-grid', action='store_true',
                       help='Create custom radius-position grid evaluation')
    parser.add_argument('--custom-radius-values', type=str, default='',
                       help='Comma-separated radius values for custom grid')
    parser.add_argument('--custom-position-values', type=str, default='',
                       help='Comma-separated position values for custom grid')
    
    args = parser.parse_args()
    main(args)
