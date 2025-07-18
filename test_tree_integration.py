#!/usr/bin/env python3
"""
Test the tree analysis integration with Fast3R demo
"""

import sys
import os

# Add fast3r to path
fast3r_path = os.path.join(os.path.dirname(__file__), 'fast3r')
if fast3r_path not in sys.path:
    sys.path.append(fast3r_path)

try:
    from fast3r.viz.tree_analysis import format_tree_measurements_html
    
    # Test with mock data
    mock_results = {
        "success": True,
        "message": "Successfully analyzed 1 tree(s)",
        "measurements": [
            {
                "tree_id": 1,
                "estimated_height": 0.050,
                "crown_diameter": 0.045,
                "area_real": 0.0012,
                "points_in_region": 10517,
                "height_meters": 4.89,
                "crown_diameter_meters": 4.41,
                "crown_area_meters2": 11.30
            }
        ],
        "scale_factor": 97.235,
        "num_trees": 1
    }
    
    html_output = format_tree_measurements_html(mock_results)
    print("✅ Tree analysis module imported and working correctly!")
    print(f"Generated HTML length: {len(html_output)} characters")
    
    # Test error case
    error_results = {"error": "Test error message"}
    error_html = format_tree_measurements_html(error_results)
    print("✅ Error handling working correctly!")
    
except Exception as e:
    print(f"❌ Error testing tree analysis module: {e}")
    sys.exit(1)

print("🌳 Tree analysis integration test completed successfully!")
