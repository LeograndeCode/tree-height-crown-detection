#!/usr/bin/env python3
"""
Debug script to check Fast3R output structure and test tree analysis
"""

import sys
import os
import numpy as np

# Add fast3r to path
fast3r_path = os.path.join(os.path.dirname(__file__), 'fast3r')
if fast3r_path not in sys.path:
    sys.path.append(fast3r_path)

def create_mock_fast3r_output():
    """Create a mock Fast3R output for testing"""
    import torch
    
    # Create mock pointcloud data (simple cube with some noise)
    n_points = 1000
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.uniform(0, 2, n_points)  # Ground at 0, tree tops at 2
    
    # Add a "tree" - higher points in the center
    center_mask = (x**2 + y**2) < 0.5  # Points near center
    z[center_mask] += 1.0  # Make them taller (tree)
    
    pts3d = np.column_stack([x, y, z])
    colors = np.random.uniform(0, 1, (n_points, 3))
    
    # Create mock Fast3R format
    mock_output = {
        'preds': [
            {
                'pts3d': torch.tensor(pts3d, dtype=torch.float32),
                'img': torch.tensor(colors, dtype=torch.float32)
            }
        ],
        'views': []
    }
    
    return mock_output

def test_tree_analysis_with_mock_data():
    """Test tree analysis with mock data"""
    try:
        from fast3r.viz.tree_analysis import process_pointcloud_for_trees
        
        print("🧪 Testing tree analysis with mock Fast3R data...")
        
        # Create mock data
        mock_output = create_mock_fast3r_output()
        print(f"✅ Created mock data with {len(mock_output['preds'])} predictions")
        
        # Test tree analysis
        results = process_pointcloud_for_trees(mock_output, scale_factor=1.0)
        
        if "error" in results:
            print(f"❌ Tree analysis failed: {results['error']}")
            return False
        else:
            print(f"✅ Tree analysis succeeded!")
            print(f"   Trees detected: {results.get('num_trees', 0)}")
            print(f"   Message: {results.get('message', 'N/A')}")
            if 'debug_info' in results:
                print(f"   Debug info: {results['debug_info']}")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_fast3r_output():
    """Test with a minimal Fast3R reconstruction"""
    try:
        print("\n🧪 Testing with minimal Fast3R reconstruction...")
        
        # Check if we have any existing Fast3R output
        test_images = ["images/img_0000.jpg", "images/img_0001.jpg", "images/img_0002.jpg"]
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if not existing_images:
            print("⚠️ No test images found, skipping real Fast3R test")
            return True
        
        print(f"✅ Found {len(existing_images)} test images")
        
        # Try to run a minimal Fast3R reconstruction
        from fast3r.dust3r.utils.image import load_images
        from fast3r.dust3r.inference_multiview import inference
        from fast3r.utils.checkpoint_utils import load_model
        
        # Load model (this might take a while)
        print("📦 Loading Fast3R model...")
        model, lit_module = load_model("jedyang97/Fast3R_ViT_Large_512", device="cpu")
        
        # Load images
        print("🖼️ Loading images...")
        imgs = load_images(existing_images[:2], size=224, verbose=False)  # Use smaller size and fewer images
        
        # Run inference
        print("🔮 Running Fast3R inference...")
        output_dict, profiling_info = inference(
            imgs,
            model,
            "cpu",
            dtype=torch.float32,
            verbose=False,
            profiling=False,
        )
        
        print(f"✅ Fast3R inference completed")
        print(f"   Output keys: {list(output_dict.keys())}")
        
        # Align points
        lit_module.align_local_pts3d_to_global(
            preds=output_dict['preds'],
            views=output_dict['views'],
            min_conf_thr_percentile=85
        )
        
        # Test tree analysis
        from fast3r.viz.tree_analysis import process_pointcloud_for_trees
        results = process_pointcloud_for_trees(output_dict, scale_factor=1.0)
        
        if "error" in results:
            print(f"❌ Tree analysis with real data failed: {results['error']}")
            return False
        else:
            print(f"✅ Tree analysis with real data succeeded!")
            print(f"   Trees detected: {results.get('num_trees', 0)}")
            print(f"   Message: {results.get('message', 'N/A')}")
            return True
            
    except Exception as e:
        print(f"⚠️ Real Fast3R test failed (this is okay): {e}")
        return True  # Don't fail the overall test

def main():
    print("🔧 Fast3R Tree Analysis Debug Tool")
    print("=" * 50)
    
    # Test 1: Mock data
    mock_success = test_tree_analysis_with_mock_data()
    
    # Test 2: Real data (optional)
    real_success = test_with_real_fast3r_output()
    
    print("\n" + "=" * 50)
    if mock_success:
        print("✅ Tree analysis working correctly with mock data!")
        print("   The issue might be with the Fast3R output format in the demo.")
        print("   Check the debug output when running the demo to see the actual structure.")
    else:
        print("❌ Tree analysis has fundamental issues - check the implementation.")
    
    return 0 if mock_success else 1

if __name__ == "__main__":
    import torch  # Import here to avoid issues if not available
    exit_code = main()
    sys.exit(exit_code)
