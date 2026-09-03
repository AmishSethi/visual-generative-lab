#!/usr/bin/env python3
"""
Generate color+shape composition datasets with configurable training coverage.

This script creates datasets for studying compositional generalization between
color and shape properties. It supports varying training coverage (25%, 50%, 75%)
and different sample sizes (5k, 10k, 15k).
"""

import os
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import itertools


def add_variations(image, color_shift_range=10, noise_level=5):
    """Add subtle color variations and noise to make dataset more realistic."""
    img_array = np.array(image)
    
    # Add small color shifts
    if color_shift_range > 0:
        color_shift = np.random.randint(-color_shift_range, color_shift_range + 1, size=3)
        img_array = img_array.astype(np.int16)
        img_array[:, :, :3] += color_shift
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Add subtle noise
    if noise_level > 0:
        noise = np.random.randint(-noise_level, noise_level + 1, size=img_array.shape)
        img_array = img_array.astype(np.int16)
        img_array += noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def generate_shape_image(shape, color_rgb, image_size=64, radius=12, 
                        background_color=(255, 255, 255), antialiasing=4):
    """Generate an image with the specified shape and color."""
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_radius = radius * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    # Center the shape
    center_x = large_size // 2
    center_y = large_size // 2
    
    # Draw shape based on type
    if shape == 'circle':
        left = center_x - large_radius
        top = center_y - large_radius
        right = center_x + large_radius
        bottom = center_y + large_radius
        draw.ellipse([left, top, right, bottom], fill=tuple(color_rgb), outline=tuple(color_rgb))
    
    elif shape == 'square':
        left = center_x - large_radius
        top = center_y - large_radius
        right = center_x + large_radius
        bottom = center_y + large_radius
        draw.rectangle([left, top, right, bottom], fill=tuple(color_rgb), outline=tuple(color_rgb))
    
    elif shape == 'triangle':
        height = large_radius * 1.732  # sqrt(3)
        points = [
            (center_x, center_y - height * 2/3),
            (center_x - large_radius, center_y + height/3),
            (center_x + large_radius, center_y + height/3)
        ]
        draw.polygon(points, fill=tuple(color_rgb), outline=tuple(color_rgb))
    
    elif shape == 'diamond':
        points = [
            (center_x, center_y - large_radius),
            (center_x + large_radius, center_y),
            (center_x, center_y + large_radius),
            (center_x - large_radius, center_y)
        ]
        draw.polygon(points, fill=tuple(color_rgb), outline=tuple(color_rgb))
    
    # Resize to final size with antialiasing
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def create_all_combinations():
    """Create all possible color+shape combinations."""
    # Define colors and shapes
    colors = {
        'red': [255, 0, 0],
        'blue': [0, 0, 255], 
        'green': [0, 255, 0],
        'yellow': [255, 255, 0],
        'magenta': [255, 0, 255],
        'cyan': [0, 255, 255],
        'orange': [255, 128, 0],
        'purple': [128, 0, 255]
    }
    
    shapes = ['circle', 'square', 'triangle', 'diamond']
    
    # Create all combinations
    all_combinations = []
    for shape in shapes:
        for color_name, color_rgb in colors.items():
            all_combinations.append((shape, color_name, color_rgb))
    
    return all_combinations, colors, shapes


def create_train_test_splits(all_combinations, coverage_percent=50):
    """
    Create train/test splits for compositional generalization.
    
    Args:
        all_combinations: List of (shape, color_name, color_rgb) tuples
        coverage_percent: Percentage of combinations to include in training (25, 50, or 75)
    
    Returns:
        Dictionary with train, test_composition, test_interpolation, test_extrapolation splits
    """
    random.seed(42)  # For reproducible splits
    np.random.seed(42)
    
    total_combinations = len(all_combinations)
    train_size = int(total_combinations * coverage_percent / 100)
    
    # Shuffle combinations
    shuffled_combinations = all_combinations.copy()
    random.shuffle(shuffled_combinations)
    
    # Split into train and test
    train_combinations = shuffled_combinations[:train_size]
    test_combinations = shuffled_combinations[train_size:]

    # Every shape and every color must appear in at least one training combination; otherwise a
    # held-out combination tests an unseen value rather than an unseen pairing. Swap, rather than
    # append, so the number of training combinations (the coverage) is unchanged. Deterministic.
    def _values(combos, idx): return {c[idx] for c in combos}
    for idx in (0, 1):                                   # 0 = shape, 1 = color name
        missing = sorted(_values(all_combinations, idx) - _values(train_combinations, idx))
        for val in missing:
            counts = {v: sum(1 for c in train_combinations if c[idx] == v) for v in _values(train_combinations, idx)}
            donor = next(c for c in train_combinations if counts[c[idx]] > 1)          # a value seen more than once
            incoming = next(c for c in test_combinations if c[idx] == val)
            train_combinations[train_combinations.index(donor)] = incoming
            test_combinations[test_combinations.index(incoming)] = donor
    for idx in (0, 1):
        assert _values(all_combinations, idx) == _values(train_combinations, idx), "value never seen in training"
    
    # For test combinations, create different generalization types
    # Pure composition: completely unseen combinations
    test_composition = test_combinations
    
    # Interpolation: combinations where individual properties are seen but not together
    # (This is essentially the same as composition for color+shape)
    test_interpolation = []
    
    # Extrapolation: would require properties not seen in training
    # (Not applicable for discrete color+shape combinations)
    test_extrapolation = []
    
    return {
        'train': train_combinations,
        'test_composition': test_composition,
        'test_interpolation': test_interpolation,
        'test_extrapolation': test_extrapolation
    }


def generate_dataset_split(combinations, split_name, output_dir, samples_per_combination, 
                          image_size=64, add_color_variance=True):
    """Generate images for a specific dataset split."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Generating {split_name} split with {len(combinations)} combinations...")
    print(f"Samples per combination: {samples_per_combination}")
    
    total_images = 0
    
    for shape, color_name, color_rgb in tqdm(combinations, desc=f"Processing {split_name}"):
        # Create folder name: shape_color (e.g., "circle_red")
        folder_name = f"{shape}_{color_name}"
        folder_path = os.path.join(split_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Generate images for this combination
        for i in range(samples_per_combination):
            # Generate base image
            img = generate_shape_image(
                shape=shape,
                color_rgb=color_rgb,
                image_size=image_size,
                radius=12  # Fixed radius for color+shape experiment
            )
            
            # Add color variations to make dataset more realistic
            if add_color_variance:
                img = add_variations(img, color_shift_range=8, noise_level=3)
            
            # Save image
            img_path = os.path.join(folder_path, f"{i:04d}.png")
            img.save(img_path)
            total_images += 1
    
    print(f"Generated {total_images} images for {split_name}")
    return total_images


def generate_color_shape_dataset(output_dir, coverage_percent=50, total_train_samples=10000,
                                image_size=64, add_color_variance=True):
    """
    Generate complete color+shape composition dataset.
    
    Args:
        output_dir: Where to save the dataset
        coverage_percent: Training coverage (25, 50, or 75)
        total_train_samples: Total number of training samples (5k, 10k, or 15k)
        image_size: Size of generated images
        add_color_variance: Whether to add color variations
    """
    print(f"Generating color+shape dataset:")
    print(f"  Coverage: {coverage_percent}%")
    print(f"  Total train samples: {total_train_samples:,}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all combinations
    all_combinations, colors, shapes = create_all_combinations()
    print(f"Total possible combinations: {len(all_combinations)}")
    
    # Create train/test splits
    splits = create_train_test_splits(all_combinations, coverage_percent)
    
    # Calculate samples per combination for training
    num_train_combinations = len(splits['train'])
    samples_per_train_combination = total_train_samples // num_train_combinations
    
    print(f"Training combinations: {num_train_combinations}")
    print(f"Samples per train combination: {samples_per_train_combination}")
    print(f"Test combinations: {len(splits['test_composition'])}")
    
    # Generate datasets
    total_images = 0
    
    # Training set
    train_images = generate_dataset_split(
        splits['train'], 'train', output_dir, 
        samples_per_train_combination, image_size, add_color_variance
    )
    total_images += train_images
    
    # Test sets - use fewer samples for evaluation
    test_samples_per_combination = 50  # Fixed for evaluation
    
    if len(splits['test_composition']) > 0:
        test_images = generate_dataset_split(
            splits['test_composition'], 'test_composition', output_dir,
            test_samples_per_combination, image_size, add_color_variance
        )
        total_images += test_images
    
    # Save dataset metadata
    metadata = {
        'dataset_type': 'color_shape_composition',
        'composition_type': 'color+shape',
        'total_combinations': len(all_combinations),
        'coverage_percent': coverage_percent,
        'total_train_samples': total_train_samples,
        'samples_per_train_combination': samples_per_train_combination,
        'test_samples_per_combination': test_samples_per_combination,
        'image_size': image_size,
        'colors': list(colors.keys()),
        'shapes': shapes,
        'splits': {
            'train': len(splits['train']),
            'test_composition': len(splits['test_composition']),
            'test_interpolation': len(splits['test_interpolation']),
            'test_extrapolation': len(splits['test_extrapolation'])
        },
        'train_combinations': [(shape, color) for shape, color, _ in splits['train']],
        'test_combinations': [(shape, color) for shape, color, _ in splits['test_composition']],
        'total_images': total_images,
        'add_color_variance': add_color_variance
    }
    
    metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Total images: {total_images:,}")
    print(f"Metadata saved to: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate color+shape composition dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--coverage', type=int, choices=[25, 50, 75], default=50,
                       help='Training coverage percentage')
    parser.add_argument('--train-samples', type=int, choices=[5000, 10000, 15000], default=10000,
                       help='Total number of training samples')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--no-color-variance', action='store_true',
                       help='Disable color variance in generated images')
    
    args = parser.parse_args()
    
    # Generate dataset
    metadata = generate_color_shape_dataset(
        output_dir=args.output_dir,
        coverage_percent=args.coverage,
        total_train_samples=args.train_samples,
        image_size=args.image_size,
        add_color_variance=not args.no_color_variance
    )
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Location: {args.output_dir}")
    print(f"📊 Training coverage: {args.coverage}%")
    print(f"🎯 Total train samples: {args.train_samples:,}")


if __name__ == "__main__":
    main()
