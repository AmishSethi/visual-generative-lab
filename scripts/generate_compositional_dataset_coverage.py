#!/usr/bin/env python3
"""
Coverage-based dataset generation for compositional experiments.
Handles all 2-property combinations with 25%/50%/75% coverage splits.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import random
import json
import itertools
from typing import List, Tuple, Dict, Optional
import math

def generate_shape_image(radius, position, shape, color_rgb, image_size=64, 
                        background_color=(255, 255, 255), antialiasing=4,
                        rotation=0, count=1, count_is_active_property=False):
    """Generate an image with shape(s) of given properties."""
    large_size = image_size * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    # SPECIAL HANDLING: Use smaller fixed size when count is being tested
    if count_is_active_property:
        # Use a fixed smaller radius that allows 4 objects to fit comfortably
        effective_radius = min(radius * 0.5, 5.0)  # Cap at 5 pixels to ensure 4 fit
    else:
        effective_radius = radius
    
    # Handle multiple objects for count > 1 - arrange predictably for extrapolation
    if count > 1:
        # Arrange objects side-by-side horizontally, centered at given position
        base_x, base_y = position[0], position[1]  # Use given position as center
        
        # Calculate spacing based on effective radius to avoid overlap
        spacing = max(effective_radius * 2.5, 10)  # Reduced minimum spacing
        
        # For horizontal arrangement centered on base position
        if count == 2:
            positions = [(base_x - spacing/2, base_y), (base_x + spacing/2, base_y)]
        elif count == 3:
            positions = [(base_x - spacing, base_y), (base_x, base_y), (base_x + spacing, base_y)]
        elif count == 4:
            positions = [(base_x - spacing*1.5, base_y), (base_x - spacing/2, base_y), 
                        (base_x + spacing/2, base_y), (base_x + spacing*1.5, base_y)]
        else:
            positions = [position]
        radii = [effective_radius] * count
    else:
        positions = [position]
        radii = [effective_radius]
    
    # Draw each object
    for obj_idx in range(min(count, len(positions))):
        obj_radius = radii[obj_idx] if obj_idx < len(radii) else radii[-1]
        obj_position = positions[obj_idx]
        
        large_radius = obj_radius * antialiasing
        large_x = obj_position[0] * antialiasing
        large_y = obj_position[1] * antialiasing
        
        center_x = large_size // 2 + large_x
        center_y = large_size // 2 + large_y
        
        # Helper function for rotation
        def rotate_point(px, py, cx, cy, angle_rad):
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            dx = px - cx
            dy = py - cy
            return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        
        angle_rad = np.radians(rotation)
        
        # Draw shape based on type
        if shape == 'circle':
            left = center_x - large_radius
            top = center_y - large_radius
            right = center_x + large_radius
            bottom = center_y + large_radius
            if right > 0 and left < large_size and bottom > 0 and top < large_size:
                draw.ellipse([left, top, right, bottom], fill=color_rgb, outline=color_rgb)
        
        elif shape == 'square':
            if rotation != 0:
                corners = [
                    (center_x - large_radius, center_y - large_radius),
                    (center_x + large_radius, center_y - large_radius),
                    (center_x + large_radius, center_y + large_radius),
                    (center_x - large_radius, center_y + large_radius),
                ]
                rotated_corners = [rotate_point(px, py, center_x, center_y, angle_rad) for px, py in corners]
                draw.polygon(rotated_corners, fill=color_rgb, outline=color_rgb)
            else:
                left = center_x - large_radius
                top = center_y - large_radius
                right = center_x + large_radius
                bottom = center_y + large_radius
                if right > 0 and left < large_size and bottom > 0 and top < large_size:
                    draw.rectangle([left, top, right, bottom], fill=color_rgb, outline=color_rgb)
        
        elif shape == 'triangle':
            height = large_radius * 1.732  # sqrt(3)
            points = [
                (center_x, center_y - height * 2/3),
                (center_x - large_radius, center_y + height/3),
                (center_x + large_radius, center_y + height/3)
            ]
            if rotation != 0:
                points = [rotate_point(px, py, center_x, center_y, angle_rad) for px, py in points]
            draw.polygon(points, fill=color_rgb, outline=color_rgb)
        
        elif shape == 'arrow':
            # Same polygon and angle convention as the single-skill rotation dataset
            # (paper_runs/table2/generate_canonical_datasets.render_arrow), so the paper's
            # arrow detector reads it; arrow length = 2 * radius (20 px at radius 10).
            s_ = large_radius * 2.0
            arrow = [(0.0, -s_ * 0.5), (-s_ * 0.25, -s_ * 0.2), (-s_ * 0.15, -s_ * 0.2), (-s_ * 0.10, s_ * 0.4),
                     (0.0, s_ * 0.5), (s_ * 0.10, s_ * 0.4), (s_ * 0.15, -s_ * 0.2), (s_ * 0.25, -s_ * 0.2)]
            a_ = np.radians(-rotation + 90.0); ca_, sa_ = np.cos(a_), np.sin(a_)
            points = [(center_x + x * ca_ - y * sa_, center_y + x * sa_ + y * ca_) for x, y in arrow]
            draw.polygon(points, fill=color_rgb, outline=color_rgb)

        elif shape == 'diamond':
            points = [
                (center_x, center_y - large_radius),
                (center_x + large_radius, center_y),
                (center_x, center_y + large_radius),
                (center_x - large_radius, center_y)
            ]
            if rotation != 0:
                points = [rotate_point(px, py, center_x, center_y, angle_rad) for px, py in points]
            draw.polygon(points, fill=color_rgb, outline=color_rgb)
    
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img

def add_variations(img, vary_color=True, noise_level=0.02):
    """Add small variations to make dataset more realistic."""
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # if vary_color:
    #     color_shift = np.random.normal(0, 0.03, (1, 1, 3))
    #     img_array = np.clip(img_array + color_shift, 0, 1)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
    
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def generate_property_values_coverage(include_properties=None):
    """Generate discrete property values for coverage-based experiments."""
    if include_properties is None:
        include_properties = ['radius', 'position', 'shape', 'color']
    
    valid_properties = ['radius', 'position', 'shape', 'color', 'count', 'rotation']
    for prop in include_properties:
        if prop not in valid_properties:
            raise ValueError(f"Invalid property '{prop}'. Must be one of {valid_properties}")
    
    if len(include_properties) < 2:
        raise ValueError("Must include at least 2 properties")
    
    property_values = {}
    
    if 'radius' in include_properties:
        property_values['radius'] = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    
    if 'position' in include_properties:
        positions = []
        pos_coords = np.linspace(-14, 14, 5)
        for x in pos_coords:
            for y in pos_coords:
                positions.append((float(x), float(y)))
        property_values['position'] = positions
    
    if 'shape' in include_properties:
        property_values['shape'] = ['circle', 'square', 'triangle', 'diamond']
    
    if 'color' in include_properties:
        property_values['color'] = [
            ('red', (255, 0, 0)),
            ('blue', (0, 0, 255)), 
            ('green', (0, 255, 0)),
            ('yellow', (255, 255, 0)),
            ('magenta', (255, 0, 255)),
            ('cyan', (0, 255, 255)),
            ('orange', (255, 128, 0)),
            ('purple', (128, 0, 255)),
        ]
    
    if 'count' in include_properties:
        property_values['count'] = [1, 2, 3, 4]
    
    if 'rotation' in include_properties:
        property_values['rotation'] = [0, 45, 90, 135, 180, 225, 270, 315]
    
    return property_values, include_properties

def create_coverage_based_splits(property_values, include_properties, coverage=0.75):
    """Create train/test splits based on coverage percentage with proper discrete/continuous handling."""
    print(f"Creating coverage-based splits with {coverage*100:.0f}% coverage")
    
    # Classify properties by type
    continuous_props = ['radius', 'position', 'rotation', 'count']  # Count can extrapolate with linear embedder
    discrete_props = ['shape', 'color']  # Use lookup embeddings, need all values in training
    
    prop_types = {}
    for prop in include_properties:
        if prop in continuous_props:
            prop_types[prop] = 'continuous'
        elif prop in discrete_props:
            prop_types[prop] = 'discrete' 
        else:
            prop_types[prop] = 'unknown'
    
    print(f"Property types: {prop_types}")
    
    # Split properties into train/holdout based on type
    train_property_values = {}
    holdout_property_values = {}
    
    for prop in include_properties:
        prop_values = property_values[prop]
        
        if prop_types[prop] == 'discrete':
            # For discrete properties: ALL values must appear in training
            print(f"  {prop} (discrete): Using ALL values in training")
            train_property_values[prop] = prop_values  # All values
            holdout_property_values[prop] = []  # No holdout values
            
        elif prop_types[prop] == 'continuous':
            # For continuous properties: Can hold out values for extrapolation testing
            if prop == 'radius':
                train_property_values[prop] = prop_values[:-2]  # Hold out 2 highest
                holdout_property_values[prop] = prop_values[-2:]
                print(f"  {prop} (continuous): Training on {train_property_values[prop]}, holdout {holdout_property_values[prop]}")
            elif prop == 'position':
                train_property_values[prop] = prop_values[:-5]  # Hold out 5 positions
                holdout_property_values[prop] = prop_values[-5:]
                print(f"  {prop} (continuous): Training on {len(train_property_values[prop])} positions, holdout {len(holdout_property_values[prop])}")
            elif prop == 'rotation':
                # Hold out diagonal angles
                train_property_values[prop] = [0, 90, 180, 270]
                holdout_property_values[prop] = [45, 135, 225, 315]
                print(f"  {prop} (continuous): Training on {train_property_values[prop]}, holdout {holdout_property_values[prop]}")
            elif prop == 'count':
                # Hold out highest count for extrapolation
                train_property_values[prop] = prop_values[:-1]  # [1, 2, 3]
                holdout_property_values[prop] = prop_values[-1:]  # [4]
                print(f"  {prop} (continuous): Training on {train_property_values[prop]}, holdout {holdout_property_values[prop]}")
        else:
            # Default: use all values
            print(f"  {prop} (unknown type): Using ALL values in training")
            train_property_values[prop] = prop_values
            holdout_property_values[prop] = []
    
    # Generate training combinations (seen properties only)
    train_property_lists = [train_property_values[prop] for prop in include_properties]
    all_train_combinations = list(itertools.product(*train_property_lists))
    
    # Apply coverage percentage to training combinations
    num_train_combinations = int(len(all_train_combinations) * coverage)
    random.seed(42)  # Reproducible sampling
    train_combinations = random.sample(all_train_combinations, num_train_combinations)
    
    print(f"Training combinations: {len(train_combinations)}")
    
    # Every in-range value of EVERY property must appear in at least one training
    # combination; otherwise a held-out combination tests an unseen value rather than an
    # unseen pairing. The check used to run for discrete properties only, which left
    # count and rotation unguarded at 25% coverage.
    for prop in include_properties:
        prop_idx = include_properties.index(prop)
        seen_values = set(combo[prop_idx] for combo in train_combinations)
        all_values = set(train_property_values[prop])
        
        if seen_values != all_values:
            missing_values = all_values - seen_values
            print(f"WARNING: Discrete property '{prop}' missing values in training: {missing_values}")
            
            # Force inclusion of missing values by adding combinations
            for missing_val in missing_values:
                forced_combo = []
                for i, other_prop in enumerate(include_properties):
                    if i == prop_idx:
                        forced_combo.append(missing_val)
                    else:
                        # Use first training value for other properties
                        forced_combo.append(train_property_values[other_prop][0])
                
                train_combinations.append(tuple(forced_combo))
                print(f"  Added forced combination: {tuple(forced_combo)}")
            
            print(f"Updated training combinations: {len(train_combinations)}")
    
    for prop in include_properties:
        prop_idx = include_properties.index(prop)
        seen = set(combo[prop_idx] for combo in train_combinations)
        assert seen == set(train_property_values[prop]), f"{prop}: values never seen in training: {set(train_property_values[prop]) - seen}"

    # Create evaluation splits
    remaining_train_combinations = [c for c in all_train_combinations if c not in train_combinations]
    test_compositional = remaining_train_combinations[:min(len(remaining_train_combinations), 200)]
    
    # OOD scenarios: Only possible if we have continuous properties with holdout values
    test_prop_a_ood = []
    test_prop_b_ood = []
    test_both_ood = []
    
    # Check if we can do OOD testing
    can_do_prop_a_ood = (len(include_properties) >= 1 and 
                         prop_types[include_properties[0]] == 'continuous' and 
                         len(holdout_property_values[include_properties[0]]) > 0)
    can_do_prop_b_ood = (len(include_properties) >= 2 and 
                         prop_types[include_properties[1]] == 'continuous' and 
                         len(holdout_property_values[include_properties[1]]) > 0)
    
    if can_do_prop_a_ood:
        print(f"Creating Property A ({include_properties[0]}) OOD scenarios...")
        prop_a_holdout = holdout_property_values[include_properties[0]]
        prop_b_train = train_property_values[include_properties[1]] if len(include_properties) >= 2 else [None]
        
        for combo in itertools.product(prop_a_holdout, prop_b_train):
            test_prop_a_ood.append(combo)
        
        test_prop_a_ood = test_prop_a_ood[:min(len(test_prop_a_ood), 100)]
    
    if can_do_prop_b_ood:
        print(f"Creating Property B ({include_properties[1]}) OOD scenarios...")
        prop_a_train = train_property_values[include_properties[0]]
        prop_b_holdout = holdout_property_values[include_properties[1]]
        
        for combo in itertools.product(prop_a_train, prop_b_holdout):
            test_prop_b_ood.append(combo)
        
        test_prop_b_ood = test_prop_b_ood[:min(len(test_prop_b_ood), 100)]
    
    if can_do_prop_a_ood and can_do_prop_b_ood:
        print("Creating Both OOD scenarios...")
        prop_a_holdout = holdout_property_values[include_properties[0]]
        prop_b_holdout = holdout_property_values[include_properties[1]]
        
        for combo in itertools.product(prop_a_holdout, prop_b_holdout):
            test_both_ood.append(combo)
        
        test_both_ood = test_both_ood[:min(len(test_both_ood), 50)]
    
    splits = {
        'train': train_combinations,
        'test_compositional': test_compositional,
        'test_prop_a_ood': test_prop_a_ood,
        'test_prop_b_ood': test_prop_b_ood,
        'test_both_ood': test_both_ood
    }
    
    print(f"Dataset splits:")
    for split_name, combinations in splits.items():
        print(f"  {split_name}: {len(combinations)} combinations")
    
    # Add metadata
    splits['_metadata'] = {
        'property_types': prop_types,
        'valid_scenarios': {
            'compositional': True,
            'prop_a_ood': can_do_prop_a_ood,
            'prop_b_ood': can_do_prop_b_ood,
            'both_ood': can_do_prop_a_ood and can_do_prop_b_ood
        }
    }
    
    return splits

def generate_dataset_split(combinations, include_properties, output_dir, split_name, 
                          samples_per_combination=20, image_size=64, add_noise=True):
    """Generate images for a specific split."""
    
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Generating {split_name} split with {len(combinations)} combinations...")
    
    for combo_idx, combination in enumerate(tqdm(combinations, desc=f"Generating {split_name}")):
        # Parse combination based on included properties
        combo_dict = {}
        for i, prop in enumerate(include_properties):
            if i < len(combination):
                combo_dict[prop] = combination[i]
        
        # Set default values and handle type conversion
        try:
            radius = float(combo_dict.get('radius', 10.0)) if 'radius' in combo_dict else 10.0
            
            position = combo_dict.get('position', (0.0, 0.0))
            if isinstance(position, (list, tuple)) and len(position) == 2:
                position = (float(position[0]), float(position[1]))
            else:
                position = (0.0, 0.0)
            
            shape = combo_dict.get('shape', 'circle')
            
            # Use asymmetric shape when rotation is a property
            if 'rotation' in include_properties and 'shape' not in include_properties:
                shape = 'arrow'  # asymmetric under every rotation; matches the single-skill rotation renders
            
            color = combo_dict.get('color', ('red', (255, 0, 0)))
            
            count = int(combo_dict.get('count', 1)) if 'count' in combo_dict else 1
            rotation = float(combo_dict.get('rotation', 0)) if 'rotation' in combo_dict else 0.0
                
        except (ValueError, TypeError) as e:
            print(f"Error processing combination {combo_idx}: {combination}")
            print(f"Error: {e}")
            continue
        
        # Handle color format
        if isinstance(color, tuple) and len(color) == 2:
            color_name, color_rgb = color
        else:
            color_name = str(color)
            color_rgb = (255, 0, 0)
        
        # Create directory name
        name_parts = []
        if 'radius' in include_properties:
            name_parts.append(f"r{radius:.1f}")
        if 'position' in include_properties:
            name_parts.append(f"x{position[0]:.1f}_y{position[1]:.1f}")
        if 'shape' in include_properties:
            name_parts.append(shape)
        if 'color' in include_properties:
            name_parts.append(color_name)
        if 'count' in include_properties:
            name_parts.append(f"cnt{count}")
        if 'rotation' in include_properties:
            name_parts.append(f"rot{rotation:.0f}")
        
        combo_name = "_".join(name_parts)
        combo_name = combo_name.replace('-', 'n').replace('.', 'p')
        combo_dir = os.path.join(split_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)
        
        for sample_idx in range(samples_per_combination):
            # Use exact property values - NO random variations for core properties
            radius_var = radius  # Exact radius value
            pos_var = position   # Exact position value
            rotation_var = rotation  # Exact rotation value
            
            # Keep slight variations only for visual realism (not core properties)
            # Generate slightly varied color
            color_var = tuple(np.clip(np.array(color_rgb) + np.random.randint(-15, 15, 3), 0, 255))
            
            # Generate background color
            bg_colors = [(255, 255, 255), (240, 240, 240), (250, 250, 250)]
            bg_color = random.choice(bg_colors)
            bg_var = tuple(np.clip(np.array(bg_color) + np.random.randint(-10, 10, 3), 200, 255))
            
            # Generate image
            img = generate_shape_image(
                radius_var, pos_var, shape, color_var,
                image_size=image_size, background_color=bg_var,
                rotation=rotation_var, count=count,
                count_is_active_property=('count' in include_properties)
            )
            
            if add_noise:
                img = add_variations(img, vary_color=True, noise_level=0.02)
            
            # Save image
            img_path = os.path.join(combo_dir, f"sample_{sample_idx:03d}.png")
            img.save(img_path)

def generate_compositional_dataset_coverage(output_dir, coverage=0.75, image_size=64, 
                                          include_properties=None, samples_per_combination=20):
    """Generate compositional dataset with coverage-based splits."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Generate property values FIRST
    property_values, include_properties = generate_property_values_coverage(include_properties)
    
    print(f"Included properties: {include_properties}")
    print(f"Using samples per combination: {samples_per_combination}")
    print(f"Property ranges:")
    for prop, values in property_values.items():
        if prop == 'radius':
            print(f"  Radii: {len(values)} values from {min(values):.1f} to {max(values):.1f}")
        elif prop == 'position':
            print(f"  Positions: {len(values)} positions")
        elif prop == 'shape':
            print(f"  Shapes: {values}")
        elif prop == 'color':
            color_names = [name for name, _ in values]
            print(f"  Colors: {color_names}")
        elif prop == 'count':
            print(f"  Counts: {values}")
        elif prop == 'rotation':
            print(f"  Rotations: {values} degrees")
    
    # STEP 3: Create coverage-based splits
    splits = create_coverage_based_splits(property_values, include_properties, coverage)
    
    total_train_samples = len(splits['train']) * samples_per_combination
    print(f"Training combinations: {len(splits['train'])}")
    print(f"Training samples: {total_train_samples}")
    
    # STEP 4: Generate datasets for each split
    for split_name, combinations in splits.items():
        if split_name != '_metadata' and len(combinations) > 0:
            # Use fewer samples for test splits to speed up generation
            split_samples_per_combo = samples_per_combination if split_name == 'train' else min(10, samples_per_combination)
            
            generate_dataset_split(combinations, include_properties, output_dir, split_name, 
                                 split_samples_per_combo, image_size)
    
    # STEP 5: Save metadata
    metadata = {
        'composition_type': 'coverage_based',
        'coverage': coverage,
        'samples_per_combination': samples_per_combination,
        'image_size': image_size,
        'include_properties': include_properties,
        'rotation_shape': 'arrow' if ('rotation' in include_properties and 'shape' not in include_properties) else None,
        'property_ranges': property_values,
        'splits': {name: len(combos) for name, combos in splits.items() if name != '_metadata'},
        'total_combinations': sum(len(combos) for name, combos in splits.items() if name != '_metadata'),
        'train_property_combinations': len(splits['train']),
        'total_samples_generated': len(splits['train']) * samples_per_combination + sum(len(combos) * min(10, samples_per_combination) for name, combos in splits.items() if name != 'train' and name != '_metadata'),
        'property_types': splits['_metadata']['property_types'],
        'valid_scenarios': splits['_metadata']['valid_scenarios'],
        'evaluation_note': 'OOD scenarios only valid for continuous properties. Discrete properties can only test compositional generalization.'
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generated successfully at {output_dir}")
    print(f"Total train combinations: {metadata['train_property_combinations']}")
    print(f"Total samples generated: {metadata['total_samples_generated']}")

def main():
    parser = argparse.ArgumentParser(description='Generate coverage-based compositional dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for the dataset')
    parser.add_argument('--coverage', type=float, default=0.75,
                       help='Coverage percentage for training set (0.25, 0.50, 0.75)')
    parser.add_argument('--samples-per-combination', type=int, default=20,
                       help='Number of samples per property combination')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--include-properties', type=str, nargs='+', 
                       choices=['radius', 'position', 'shape', 'color', 'count', 'rotation'],
                       default=['radius', 'position'],
                       help='Properties to include in the dataset (minimum 2 required)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.include_properties) < 2:
        parser.error("Must include at least 2 properties")
    
    if args.coverage not in [0.25, 0.50, 0.75]:
        parser.error("Coverage must be one of: 0.25, 0.50, 0.75")
    
    generate_compositional_dataset_coverage(
        output_dir=args.output_dir,
        coverage=args.coverage,
        image_size=args.image_size,
        include_properties=args.include_properties,
        samples_per_combination=args.samples_per_combination
    )

if __name__ == "__main__":
    main()