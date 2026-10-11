#!/usr/bin/env python3
"""
Sampling and per-skill metrics for the compositional generalization experiments.
"""

import os
import numpy as np


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
from tqdm import tqdm
import cv2
import itertools

# Import model components
from vgl.models_compositional import DiT_models_compositional as DiT_models


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
try:
    from vgl.eval_rotation import calculate_rotation_metrics
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
    
    # Parse DiT model size
    if True:
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
    """Load the model from checkpoint."""
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
    
    # Load state dict (handle DiT/UNet with helper)
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
        # Fallback: parse directory names directly
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
                # Imported outside the try so a missing module fails loudly instead
                # of silently degrading the shape metric to the contour fallback.
                from scripts.generate_compositional_dataset_coverage import generate_shape_image
                try:
                    for sh_id, sh_name in enumerate(shape_names):
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
                shape_type = os.environ.get('VGL_ROTATION_SHAPE', 'triangle')  # 'arrow' for datasets rendered with rotation_shape=arrow
                # The rotation detector correlates against a full-frame template with the object at the
                # center, so it is not translation invariant. When rotation is composed with position,
                # move the object's centroid to the image center before reading the angle; position is
                # scored separately from the original image.
                rot_input = image_np
                if 'position' in include_properties:
                    gray_ = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                    _, mask_ = cv2.threshold(gray_, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    mom_ = cv2.moments(mask_)
                    if mom_['m00'] > 0:
                        cx_, cy_ = mom_['m10'] / mom_['m00'], mom_['m01'] / mom_['m00']
                        h_, w_ = gray_.shape
                        M_ = np.float32([[1, 0, w_ / 2.0 - cx_], [0, 1, h_ / 2.0 - cy_]])
                        bg_ = tuple(int(v) for v in np.median(image_np.reshape(-1, 3), axis=0))
                        rot_input = cv2.warpAffine(image_np, M_, (w_, h_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_)
                rotation_metrics = calculate_rotation_metrics(
                    rot_input, expected_rotation, shape_type, image_size=64
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
