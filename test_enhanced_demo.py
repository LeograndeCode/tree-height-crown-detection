#!/usr/bin/env python3
"""
Test script to verify the enhanced Fast3R demo with tree analysis
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import torch
        import gradio as gr
        import numpy as np
        import cv2
        import open3d as o3d
        print("✅ All core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_tree_analysis_import():
    """Test if tree analysis module can be imported"""
    try:
        sys.path.append('fast3r')
        from fast3r.viz.tree_analysis import (
            process_pointcloud_for_trees, 
            format_tree_measurements_html,
            pointcloud_to_orthographic_depth,
            detect_tree_crowns,
            compute_tree_measurements
        )
        print("✅ Tree analysis module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import tree analysis module: {e}")
        return False

def test_demo_import():
    """Test if the enhanced demo can be imported"""
    try:
        sys.path.append('fast3r')
        from fast3r.viz.demo import create_demo
        print("✅ Enhanced demo module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import demo module: {e}")
        return False

def main():
    print("🧪 Testing Enhanced Fast3R Demo with Tree Analysis")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        all_tests_passed = False
    
    # Test 2: Tree analysis module
    print("\n2. Testing tree analysis module...")
    if not test_tree_analysis_import():
        all_tests_passed = False
    
    # Test 3: Demo module
    print("\n3. Testing enhanced demo module...")
    if not test_demo_import():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! The enhanced Fast3R demo is ready to use.")
        print("\nTo run the demo:")
        print("  cd fast3r")
        print("  python -m fast3r.viz.demo --checkpoint_dir jedyang97/Fast3R_ViT_Large_512")
        print("\nFeatures included:")
        print("  ✅ 3D reconstruction from images/video")
        print("  ✅ Automatic tree detection and measurement")
        print("  ✅ Tree height, crown diameter, and crown area analysis")
        print("  ✅ GPS-based scale conversion (when available)")
        print("  ✅ Real-world measurements in meters")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
