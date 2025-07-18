#!/usr/bin/env python3
"""
Quick test for tree analysis function
"""

import sys
import os
sys.path.append('fast3r')

try:
    import numpy as np
    import torch
    from fast3r.viz.tree_analysis import process_pointcloud_for_trees
    
    print("Testing basic functionality...")
    
    # Minimal test data
    pts = np.array([[0,0,0], [0,0,1], [1,1,2]], dtype=np.float32)
    colors = np.ones((3, 3), dtype=np.float32)
    
    mock_data = {
        'preds': [{'pts3d': torch.tensor(pts), 'img': torch.tensor(colors)}],
        'views': []
    }
    
    print("Calling process_pointcloud_for_trees...")
    result = process_pointcloud_for_trees(mock_data)
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    elif 'success' in result:
        print(f"Success: {result['success']}")
        print(f"Message: {result.get('message', 'No message')}")
    else:
        print("Unexpected result format")
        
except Exception as e:
    import traceback
    print(f"Exception: {e}")
    traceback.print_exc()
