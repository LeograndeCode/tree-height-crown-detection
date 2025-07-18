#!/usr/bin/env python3
"""
Test Fast3R tree analysis with real inference
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

def test_real_fast3r_reconstruction():
    """Test with actual Fast3R reconstruction"""
    try:
        print("🔬 Testing Real Fast3R Reconstruction + Tree Analysis")
        print("=" * 60)
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {device}")
        
        # Load test images
        test_images = ["images/img_0000.jpg", "images/img_0001.jpg", "images/img_0002.jpg"]
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if len(existing_images) < 2:
            print("❌ Need at least 2 test images")
            return False
        
        print(f"🖼️ Using {len(existing_images)} images: {existing_images}")
        
        # Load model (use small size for faster testing)
        print("📦 Loading Fast3R model...")
        model, lit_module = load_model("jedyang97/Fast3R_ViT_Large_512", device=device)
        print("✅ Model loaded successfully")
        
        # Load and process images
        print("🖼️ Loading and processing images...")
        imgs = load_images(existing_images, size=224, verbose=False)  # Small size for testing
        print(f"✅ Loaded {len(imgs)} images")
        
        # Run inference
        print("🔮 Running Fast3R inference...")
        output_dict, profiling_info = inference(
            imgs,
            model,
            device,
            dtype=torch.float32,
            verbose=True,
            profiling=True,
        )
        print(f"✅ Inference completed in {profiling_info['total_time']:.2f}s")
        
        # Check output structure
        print(f"📊 Output structure:")
        print(f"   Keys: {list(output_dict.keys())}")
        print(f"   Preds: {len(output_dict.get('preds', []))}")
        print(f"   Views: {len(output_dict.get('views', []))}")
        
        # Check individual predictions
        for i, pred in enumerate(output_dict.get('preds', [])):
            if isinstance(pred, dict):
                print(f"   Pred {i}: {list(pred.keys())}")
                if 'pts3d' in pred and pred['pts3d'] is not None:
                    pts_shape = pred['pts3d'].shape if hasattr(pred['pts3d'], 'shape') else 'no shape'
                    print(f"     pts3d shape: {pts_shape}")
        
        # Move tensors to CPU and align points
        print("🔄 Processing predictions...")
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
            print(f"⚠️ Warning during tensor processing: {e}")
        
        # Align points
        print("🎯 Aligning points...")
        lit_module.align_local_pts3d_to_global(
            preds=output_dict['preds'],
            views=output_dict['views'],
            min_conf_thr_percentile=85
        )
        print("✅ Points aligned")
        
        # Test tree analysis
        print("🌳 Running tree analysis...")
        tree_results = process_pointcloud_for_trees(output_dict, scale_factor=1.0)
        
        print("📊 Tree Analysis Results:")
        if "error" in tree_results:
            print(f"❌ Error: {tree_results['error']}")
            return False
        elif tree_results.get("success"):
            print(f"✅ Success: Found {tree_results.get('num_trees', 0)} trees")
            print(f"   Message: {tree_results.get('message', 'N/A')}")
            if tree_results.get('measurements'):
                for i, measurement in enumerate(tree_results['measurements']):
                    height = measurement.get('estimated_height', 0)
                    diameter = measurement.get('crown_diameter', 0)
                    print(f"   Tree {i+1}: Height={height:.3f}, Diameter={diameter:.3f}")
            return True
        else:
            print(f"⚠️ No trees detected: {tree_results.get('message', 'Unknown')}")
            if 'debug_info' in tree_results:
                print(f"   Debug: {tree_results['debug_info']}")
            return True  # No trees detected is still a valid result
        
    except Exception as e:
        import traceback
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_fast3r_reconstruction()
    if success:
        print("\n🎉 Real Fast3R + Tree Analysis test completed successfully!")
    else:
        print("\n❌ Test failed - check the output above")
        sys.exit(1)
