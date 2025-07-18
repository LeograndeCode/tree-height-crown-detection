#!/usr/bin/env python3
"""
Test tree analysis with specific images (0006-0009) using actual Fast3R inference
"""

import sys
import os
sys.path.append('fast3r')

import torch
import numpy as np
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.utils.checkpoint_utils import load_model
from fast3r.viz.tree_analysis import process_pointcloud_for_trees

def test_with_specific_images():
    """Test tree analysis with images 0006-0009"""
    
    # Use the better images as suggested
    test_images = [
        "images/img_0006.jpg",
        "images/img_0007.jpg", 
        "images/img_0008.jpg",
        "images/img_0009.jpg"
    ]
    
    # Check if images exist
    existing_images = [img for img in test_images if os.path.exists(img)]
    if not existing_images:
        print(f"❌ No test images found from: {test_images}")
        return False
    
    print(f"✅ Found {len(existing_images)} images: {existing_images}")
    
    try:
        # Load model
        print("📦 Loading Fast3R model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model, lit_module = load_model("jedyang97/Fast3R_ViT_Large_512", device=device)
        print("✅ Model loaded successfully")
        
        # Load and process images
        print("🖼️ Loading and processing images...")
        imgs = load_images(existing_images, size=512, verbose=True)
        print(f"✅ Loaded {len(imgs)} images")
        
        # Run Fast3R inference
        print("🔮 Running Fast3R inference...")
        output_dict, profiling_info = inference(
            imgs,
            model,
            device,
            dtype=torch.float32,
            verbose=True,
            profiling=True,
        )
        
        print(f"✅ Fast3R inference completed in {profiling_info['total_time']:.2f}s")
        print(f"   Output keys: {list(output_dict.keys())}")
        print(f"   Number of preds: {len(output_dict.get('preds', []))}")
        print(f"   Number of views: {len(output_dict.get('views', []))}")
        
        # Move tensors to CPU and align points (like in the demo)
        print("🔧 Processing predictions...")
        try:
            for pred in output_dict['preds']:
                for k, v in pred.items():
                    if isinstance(v, torch.Tensor):
                        pred[k] = v.cpu()
            for view in output_dict['views']:
                for k, v in view.items():
                    if isinstance(v, torch.Tensor):
                        view[k] = v.cpu()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning during tensor processing: {e}")
        
        # Align points (like in the demo)
        print("🎯 Aligning points...")
        lit_module.align_local_pts3d_to_global(
            preds=output_dict['preds'],
            views=output_dict['views'],
            min_conf_thr_percentile=85
        )
        print("✅ Points aligned")
        
        # Check scale factor
        scale_factor = None
        scale_file = "output/scale_estimation.json"
        if os.path.exists(scale_file):
            import json
            try:
                with open(scale_file, 'r') as f:
                    scale_data = json.load(f)
                    scale_factor = scale_data.get('scale_factor')
                    print(f"📐 Found scale factor: {scale_factor:.3f} meters/unit")
            except:
                print("⚠️ Could not load scale factor")
        else:
            print("⚠️ No scale factor file found")
        
        # Test tree analysis
        print("🌳 Running tree analysis...")
        results = process_pointcloud_for_trees(
            output_dict, 
            scale_factor=scale_factor,
            output_dir="temp_tree_test"
        )
        
        # Display results
        print("\n" + "="*60)
        print("TREE ANALYSIS RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"❌ Tree analysis failed: {results['error']}")
            return False
        elif results.get("success", False):
            print(f"✅ SUCCESS: {results['message']}")
            measurements = results.get('measurements', [])
            
            if measurements:
                for measurement in measurements:
                    tree_id = measurement.get('tree_id', 'Unknown')
                    height = measurement.get('estimated_height', 0)
                    diameter = measurement.get('crown_diameter', 0)
                    area = measurement.get('area_real', 0)
                    points = measurement.get('points_in_region', 0)
                    
                    print(f"\n🌲 Tree {tree_id}:")
                    if scale_factor:
                        height_m = measurement.get('height_meters', height * scale_factor)
                        diameter_m = measurement.get('crown_diameter_meters', diameter * scale_factor)
                        area_m2 = measurement.get('crown_area_meters2', area * (scale_factor ** 2))
                        print(f"   📏 Height: {height_m:.2f} meters")
                        print(f"   🌿 Crown Diameter: {diameter_m:.2f} meters")
                        print(f"   🍃 Crown Area: {area_m2:.2f} m²")
                    else:
                        print(f"   📏 Height: {height:.3f} units")
                        print(f"   🌿 Crown Diameter: {diameter:.3f} units")
                        print(f"   🍃 Crown Area: {area:.4f} units²")
                    print(f"   🔍 Points Analyzed: {points:,}")
            else:
                print("⚠️ No measurements available")
        else:
            print(f"⚠️ {results.get('message', 'Unknown result')}")
            if 'debug_info' in results:
                print(f"Debug info: {results['debug_info']}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Test failed with exception: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing Tree Analysis with Images 0006-0009")
    print("=" * 60)
    
    success = test_with_specific_images()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed - check errors above")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
