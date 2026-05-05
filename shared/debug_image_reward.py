#!/usr/bin/env python3
"""
Debug script to check Image Reward calculation with Flickr8K dataset
Investigates why scores are > 1.0
"""

import os
import json
import torch
import ImageReward as RM
from dataset.flickr8k import process_flickr8k
from PIL import Image
import numpy as np

def debug_image_reward():
    """
    Test Image Reward with actual Flickr8K data to understand score distribution
    """
    
    print("=" * 80)
    print("DEBUG: Image Reward Score Analysis")
    print("=" * 80)
    
    # Load Flickr8K captions
    flickr_dir = "flickr8k"
    images_dir = os.path.join(flickr_dir, "Images")
    captions_path = os.path.join(flickr_dir, "captions.txt")
    
    print(f"\n1. Loading Flickr8K dataset from {captions_path}...")
    captions, dimensions = process_flickr8k(images_dir, captions_path, limit=10)
    print(f"   ✓ Loaded {len(captions)} captions")
    
    if not captions:
        print("   ✗ No captions found! Exiting.")
        return
    
    # Load ImageReward model
    print("\n2. Loading ImageReward-v1.0 model...")
    try:
        model = RM.load("ImageReward-v1.0")
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Get model statistics (mean and std used for normalization)
    if hasattr(model, 'mean') and hasattr(model, 'std'):
        print(f"   Model normalization params:")
        print(f"     - Mean: {model.mean}")
        print(f"     - Std: {model.std}")
    
    # Test scoring on a few images
    print(f"\n3. Testing Image Reward scores on {len(captions)} images...")
    print("-" * 80)
    
    scores = []
    raw_scores = []
    valid_count = 0
    error_count = 0
    
    for idx, (filename, caption) in enumerate(list(captions.items())[:10]):
        img_path = os.path.join(images_dir, filename)
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"\n   [{idx+1}] ✗ Image not found: {filename}")
            error_count += 1
            continue
        
        # Check caption validity
        if not caption or caption.strip() == "":
            print(f"\n   [{idx+1}] ⚠ Empty caption for {filename}")
            error_count += 1
            continue
        
        try:
            # Try to open image
            img = Image.open(img_path)
            img_size = img.size
            
            with torch.inference_mode():
                score = model.score(caption, img_path)
            
            scores.append(score)
            valid_count += 1
            
            # Print result
            print(f"\n   [{idx+1}] ✓ {filename}")
            print(f"       Caption: {caption[:60]}...")
            print(f"       Image size: {img_size}")
            print(f"       Image Reward Score: {score:.6f}")
            print(f"       Score > 1.0: {score > 1.0}")
            print(f"       Score < -1.0: {score < -1.0}")
            
        except Exception as e:
            print(f"\n   [{idx+1}] ✗ Error processing {filename}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if scores:
        scores = np.array(scores)
        print(f"\nValid scores: {valid_count}")
        print(f"Error count: {error_count}")
        print(f"\nScore Distribution:")
        print(f"  - Min: {scores.min():.6f}")
        print(f"  - Max: {scores.max():.6f}")
        print(f"  - Mean: {scores.mean():.6f}")
        print(f"  - Median: {np.median(scores):.6f}")
        print(f"  - Std: {scores.std():.6f}")
        print(f"\nScores > 1.0: {(scores > 1.0).sum()} out of {len(scores)} ({(scores > 1.0).sum() / len(scores) * 100:.1f}%)")
        print(f"Scores < -1.0: {(scores < -1.0).sum()} out of {len(scores)} ({(scores < -1.0).sum() / len(scores) * 100:.1f}%)")
        print(f"Scores in [-1, 1]: {((scores >= -1) & (scores <= 1)).sum()} out of {len(scores)} ({((scores >= -1) & (scores <= 1)).sum() / len(scores) * 100:.1f}%)")
        
        print("  EXPECTED BEHAVIOR:")
        print("   - ImageReward follows ~N(0,1) distribution (standard normal)")
        print("   - ~16% of scores should be > 1.0 (normal for standard normal)")
        print("   - ~16% of scores should be < -1.0 (normal for standard normal)")
        print("   - If scores are consistently >> 1.0, check caption quality")
    else:
        print("No valid scores to analyze!")
    
    print("\n" + "=" * 80)

def compare_with_coco():
    """
    Compare ImageReward scores between Flickr8K and COCO if available
    """
    print("\n" + "=" * 80)
    print("Comparing with COCO dataset (if available)")
    print("=" * 80)
    
    coco_captions_path = "coco/annotations/captions_val2017.json"
    
    if not os.path.exists(coco_captions_path):
        print("COCO captions not found. Skipping comparison.")
        return
    
    print(f"\nLoading COCO captions from {coco_captions_path}...")
    with open(coco_captions_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract a few sample captions
    sample_captions = [ann['caption'] for ann in coco_data['annotations'][:5]]
    
    model = RM.load("ImageReward-v1.0")
    
    print(f"Testing {len(sample_captions)} COCO captions (without images, using None)...")
    print("Note: Testing with None image paths - scores will reflect model behavior")

if __name__ == "__main__":
    debug_image_reward()
    # Uncomment to compare with COCO
    # compare_with_coco()
