"""
Tree Height and Crown Analysis Module for Fast3R
================================================

This module provides tree height and crown dimension analysis functionality
for integration with the Fast3R Gradio demo.
"""

import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

def pointcloud_to_orthographic_depth(pcd, resolution=512, padding=0.1):
    """
    Convert 3D pointcloud to orthographic depth map and color image.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5
    
    if len(points) == 0:
        return None, None, None
    
    # Get bounds
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    ranges = max_bounds - min_bounds
    
    # Add padding
    padding_offset = ranges * padding
    min_bounds -= padding_offset
    max_bounds += padding_offset
    ranges = max_bounds - min_bounds
    
    # Use X and Y for orthographic projection (top-down view)
    x_range = ranges[0]
    y_range = ranges[1]
    
    # Make the projection square by using the larger range
    max_range = max(x_range, y_range)
    
    # Create grid
    x_normalized = (points[:, 0] - min_bounds[0]) / max_range
    y_normalized = (points[:, 1] - min_bounds[1]) / max_range
    
    # Convert to pixel coordinates
    x_pixels = (x_normalized * (resolution - 1)).astype(int)
    y_pixels = (y_normalized * (resolution - 1)).astype(int)
    
    # Initialize depth and color maps
    depth_map = np.zeros((resolution, resolution))
    color_map = np.zeros((resolution, resolution, 3))
    point_count = np.zeros((resolution, resolution))
    
    # Fill the maps
    for i in range(len(points)):
        x_pix, y_pix = x_pixels[i], y_pixels[i]
        
        if 0 <= x_pix < resolution and 0 <= y_pix < resolution:
            depth_value = points[i, 2]
            depth_map[y_pix, x_pix] += depth_value
            color_map[y_pix, x_pix] += colors[i]
            point_count[y_pix, x_pix] += 1
    
    # Average the accumulated values
    mask = point_count > 0
    depth_map[mask] /= point_count[mask]
    color_map[mask] /= point_count[mask, np.newaxis]
    
    # Normalize depth map
    if depth_map.max() > depth_map.min():
        depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    else:
        depth_map_norm = depth_map
    
    # Check orientation
    height, width = depth_map_norm.shape
    top_half = depth_map_norm[:height//2, :]
    bottom_half = depth_map_norm[height//2:, :]
    
    top_avg = np.mean(top_half[top_half > 0]) if np.any(top_half > 0) else 0
    bottom_avg = np.mean(bottom_half[bottom_half > 0]) if np.any(bottom_half > 0) else 0
    
    # If bottom half is significantly brighter than top half, flip the image
    if bottom_avg > top_avg * 1.2 and bottom_avg > 0.1:
        depth_map_norm = np.flipud(depth_map_norm)
        color_map = np.flipud(color_map)
    
    transform_info = {
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'max_range': max_range,
        'resolution': resolution,
        'padding': padding,
        'flipped': bottom_avg > top_avg * 1.2 and bottom_avg > 0.1
    }
    
    return depth_map_norm, color_map, transform_info

def detect_tree_crowns(depth_map):
    """
    Detect tree crowns from orthographic depth map.
    """
    depth_values = depth_map[depth_map > 0]
    if len(depth_values) == 0:
        return [], None
    
    # Use percentile-based thresholding for crown detection
    crown_threshold_95 = np.percentile(depth_values, 95)
    crown_threshold_90 = np.percentile(depth_values, 90)
    crown_threshold_85 = np.percentile(depth_values, 85)
    
    # Statistical approach for tree tops
    depth_std = depth_values.std()
    statistical_threshold = depth_values.mean() + 1.5 * depth_std
    
    # Choose threshold that focuses on tree crowns
    tree_threshold = max(crown_threshold_90, statistical_threshold)
    tree_threshold = min(tree_threshold, crown_threshold_95)
    
    # Apply threshold
    tree_mask = depth_map > tree_threshold
    
    if tree_mask.sum() < 100:
        tree_threshold = crown_threshold_85
        tree_mask = depth_map > tree_threshold
        
        if tree_mask.sum() < 50:
            tree_threshold = np.percentile(depth_values, 80)
            tree_mask = depth_map > tree_threshold
    
    # Morphological operations for crown structure
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    tree_mask_cleaned = cv2.morphologyEx(tree_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
    tree_mask_cleaned = cv2.morphologyEx(tree_mask_cleaned, cv2.MORPH_CLOSE, kernel_medium)
    tree_mask_cleaned = cv2.morphologyEx(tree_mask_cleaned, cv2.MORPH_CLOSE, kernel_large)
    
    # Find connected components
    num_labels, labels, stats_cv, centroids = cv2.connectedComponentsWithStats(
        tree_mask_cleaned.astype(np.uint8), connectivity=8)
    
    # Filter regions based on crown characteristics
    tree_regions = []
    min_area = 50
    
    candidate_regions = []
    for i in range(1, num_labels):
        area = stats_cv[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            region_mask = (labels == i)
            y_coords, x_coords = np.where(region_mask)
            if len(x_coords) > 0:
                width = x_coords.max() - x_coords.min() + 1
                height = y_coords.max() - y_coords.min() + 1
                aspect_ratio = max(width, height) / min(width, height)
                
                if aspect_ratio < 4.0:
                    candidate_regions.append({
                        'label': i,
                        'area': area,
                        'bbox': stats_cv[i],
                        'centroid': centroids[i],
                        'mask': region_mask,
                        'aspect_ratio': aspect_ratio,
                        'width': width,
                        'height': height
                    })
    
    # Select best crown regions
    if len(candidate_regions) >= 1:
        if len(candidate_regions) == 1:
            tree_regions = candidate_regions
        else:
            candidate_regions.sort(key=lambda x: x['area'], reverse=True)
            
            if len(candidate_regions) <= 3:
                # Merge regions
                merged_mask = np.zeros_like(tree_mask_cleaned, dtype=bool)
                total_area = 0
                
                for region in candidate_regions:
                    merged_mask |= region['mask']
                    total_area += region['area']
                
                y_coords, x_coords = np.where(merged_mask)
                if len(x_coords) > 0:
                    bbox_merged = [x_coords.min(), y_coords.min(), 
                                  x_coords.max() - x_coords.min(), 
                                  y_coords.max() - y_coords.min()]
                    centroid_merged = [np.mean(x_coords), np.mean(y_coords)]
                    
                    tree_regions = [{
                        'label': 1,
                        'area': total_area,
                        'bbox': bbox_merged,
                        'centroid': centroid_merged,
                        'mask': merged_mask
                    }]
            else:
                tree_regions = [candidate_regions[0]]
    
    return tree_regions, tree_threshold

def compute_tree_measurements(tree_regions, depth_map, transform_info, pcd_original, ground_level_reference=0.0, scale_factor=None):
    """
    Compute height and crown dimensions for detected trees.
    """
    crown_measurements = []
    max_range = transform_info['max_range']
    resolution = transform_info['resolution']
    pixel_to_real_scale = max_range / resolution
    
    points = np.asarray(pcd_original.points)
    
    for i, region in enumerate(tree_regions):
        mask = region['mask']
        bbox = region['bbox']
        
        # Get pixel coordinates of the region
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            continue
            
        # Map pixel coordinates back to X, Y in pointcloud
        min_bounds = transform_info['min_bounds']
        max_range_proj = transform_info['max_range']
        
        # Inverse mapping (corrected to match orthographic projection)
        x_real = x_coords / (resolution - 1) * max_range_proj + min_bounds[0]
        y_real = y_coords / (resolution - 1) * max_range_proj + min_bounds[1]
        
        # Find matching points in the pointcloud
        region_points_z = []
        region_points = []
        tolerance = pixel_to_real_scale * 2.0
        
        for xr, yr in zip(x_real, y_real):
            dists = np.square(points[:,0] - xr) + np.square(points[:,1] - yr)
            idx = np.argmin(dists)
            if dists[idx] < (tolerance**2):
                region_points_z.append(points[idx,2])
                region_points.append(points[idx])
        
        if len(region_points_z) == 0:
            estimated_height = 0.0
            tree_top = ground_level_reference
        else:
            region_points = np.array(region_points)
            region_points_z = np.array(region_points_z)
            
            # Use 95th percentile as tree top to avoid outliers
            tree_top = np.percentile(region_points_z, 95)
            estimated_height = tree_top - ground_level_reference
        
        # Crown diameter calculation
        bbox_width_real = bbox[2] * pixel_to_real_scale
        bbox_height_real = bbox[3] * pixel_to_real_scale
        crown_diameter = np.sqrt(bbox_width_real * bbox_height_real)
        
        # Crown area from actual pixel count
        crown_area_real = region['area'] * (pixel_to_real_scale ** 2)
        
        measurements = {
            'tree_id': i + 1,
            'area_pixels': region['area'],
            'area_real': crown_area_real,
            'centroid_pixel': region['centroid'],
            'bbox_pixels': [bbox[2], bbox[3]],
            'bbox_real': [bbox_width_real, bbox_height_real],
            'estimated_height': estimated_height,
            'tree_top_height': tree_top,
            'crown_diameter': crown_diameter,
            'ground_reference': ground_level_reference,
            'points_in_region': len(region_points_z),
        }
        
        # Add real-world measurements if scale available
        if scale_factor:
            measurements.update({
                'height_meters': estimated_height * scale_factor,
                'crown_diameter_meters': crown_diameter * scale_factor,
                'crown_area_meters2': crown_area_real * (scale_factor ** 2)
            })
        
        crown_measurements.append(measurements)
    
    return crown_measurements

def process_pointcloud_for_trees(output_dict, scale_factor=None, output_dir="temp_tree_analysis"):
    """
    Process Fast3R output to extract tree measurements.
    
    Args:
        output_dict: Fast3R output dictionary
        scale_factor: Optional scale factor for real-world measurements
        output_dir: Directory to save temporary files
    
    Returns:
        Dict containing tree measurements and analysis results
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Debug: Print output_dict structure
        print(f"🔍 Debug: Fast3R output keys: {list(output_dict.keys())}")
        
        # Extract pointcloud from Fast3R output - handle different formats
        points_list = []
        colors_list = []
        
        # Method 1: Try 'preds' structure
        if 'preds' in output_dict and output_dict['preds']:
            print(f"🔍 Debug: Found {len(output_dict['preds'])} predictions")
            for i, pred in enumerate(output_dict['preds']):
                print(f"🔍 Debug: Pred {i} keys: {list(pred.keys()) if isinstance(pred, dict) else type(pred)}")
                
                if isinstance(pred, dict) and 'pts3d' in pred and pred['pts3d'] is not None:
                    pts3d = pred['pts3d']
                    
                    # Handle tensor conversion
                    if hasattr(pts3d, 'cpu'):
                        pts3d = pts3d.cpu().numpy()
                    elif hasattr(pts3d, 'numpy'):
                        pts3d = pts3d.numpy()
                    
                    print(f"🔍 Debug: Pred {i} pts3d shape: {pts3d.shape}")
                    
                    # Handle colors
                    if 'img' in pred and pred['img'] is not None:
                        img = pred['img']
                        if hasattr(img, 'cpu'):
                            img = img.cpu().numpy()
                        elif hasattr(img, 'numpy'):
                            img = img.numpy()
                        
                        # Reshape colors to match points
                        if len(img.shape) == 3:
                            colors = img.reshape(-1, 3)
                        elif len(img.shape) == 4 and img.shape[0] == 1:
                            colors = img[0].reshape(-1, 3)
                        else:
                            colors = np.ones((pts3d.shape[0], 3)) * 0.5
                    else:
                        colors = np.ones((pts3d.shape[0], 3)) * 0.5
                    
                    # Filter valid points
                    if len(pts3d.shape) == 2 and pts3d.shape[1] == 3:
                        valid_mask = ~np.isnan(pts3d).any(axis=1) & ~np.isinf(pts3d).any(axis=1)
                        valid_count = valid_mask.sum()
                        print(f"🔍 Debug: Pred {i} valid points: {valid_count}/{len(pts3d)}")
                        
                        if valid_count > 0:
                            points_list.append(pts3d[valid_mask])
                            colors_list.append(colors[valid_mask] if len(colors) == len(pts3d) else colors)
                    elif len(pts3d.shape) == 3:
                        # Handle batch dimension
                        for batch_idx in range(pts3d.shape[0]):
                            batch_pts = pts3d[batch_idx]
                            valid_mask = ~np.isnan(batch_pts).any(axis=1) & ~np.isinf(batch_pts).any(axis=1)
                            valid_count = valid_mask.sum()
                            print(f"🔍 Debug: Pred {i} batch {batch_idx} valid points: {valid_count}/{len(batch_pts)}")
                            
                            if valid_count > 0:
                                points_list.append(batch_pts[valid_mask])
                                if len(colors.shape) == 3 and colors.shape[0] == pts3d.shape[0]:
                                    batch_colors = colors[batch_idx]
                                    colors_list.append(batch_colors[valid_mask] if len(batch_colors) == len(batch_pts) else batch_colors)
                                else:
                                    colors_list.append(np.ones((valid_count, 3)) * 0.5)
        
        # Method 2: Try 'views' structure if preds didn't work
        if not points_list and 'views' in output_dict and output_dict['views']:
            print(f"🔍 Debug: Trying views structure with {len(output_dict['views'])} views")
            for i, view in enumerate(output_dict['views']):
                print(f"🔍 Debug: View {i} keys: {list(view.keys()) if isinstance(view, dict) else type(view)}")
                
                if isinstance(view, dict) and 'pts3d' in view and view['pts3d'] is not None:
                    pts3d = view['pts3d']
                    
                    # Handle tensor conversion
                    if hasattr(pts3d, 'cpu'):
                        pts3d = pts3d.cpu().numpy()
                    elif hasattr(pts3d, 'numpy'):
                        pts3d = pts3d.numpy()
                    
                    print(f"🔍 Debug: View {i} pts3d shape: {pts3d.shape}")
                    
                    # Handle colors
                    if 'img' in view and view['img'] is not None:
                        img = view['img']
                        if hasattr(img, 'cpu'):
                            img = img.cpu().numpy()
                        elif hasattr(img, 'numpy'):
                            img = img.numpy()
                        
                        if len(img.shape) == 3:
                            colors = img.reshape(-1, 3)
                        else:
                            colors = np.ones((pts3d.shape[0], 3)) * 0.5
                    else:
                        colors = np.ones((pts3d.shape[0], 3)) * 0.5
                    
                    # Filter valid points
                    if len(pts3d.shape) == 2 and pts3d.shape[1] == 3:
                        valid_mask = ~np.isnan(pts3d).any(axis=1) & ~np.isinf(pts3d).any(axis=1)
                        valid_count = valid_mask.sum()
                        print(f"🔍 Debug: View {i} valid points: {valid_count}/{len(pts3d)}")
                        
                        if valid_count > 0:
                            points_list.append(pts3d[valid_mask])
                            colors_list.append(colors[valid_mask] if len(colors) == len(pts3d) else colors)
        
        # Check if we found any points
        total_points = sum(len(pts) for pts in points_list) if points_list else 0
        print(f"🔍 Debug: Total valid points found: {total_points}")
        
        if not points_list or total_points == 0:
            return {"error": f"No valid points found in reconstruction. Output structure: {list(output_dict.keys())}"}
        
        # Combine all points
        all_points = np.vstack(points_list)
        all_colors = np.vstack(colors_list)
        
        print(f"🔍 Debug: Combined points shape: {all_points.shape}, colors shape: {all_colors.shape}")
        
        # Ensure colors are in [0, 1] range
        if all_colors.max() > 1.0:
            all_colors = all_colors / 255.0
        all_colors = np.clip(all_colors, 0, 1)
        
        # Create Open3D pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        print(f"🔍 Debug: Created pointcloud with {len(pcd.points)} points")
        
        # Z-coordinate correction (flip if needed)
        points = np.asarray(pcd.points)
        original_z_range = f"[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]"
        points[:, 2] = points[:, 2] * -1  # Apply Z-flip
        print(f"🔍 Debug: Z-flip applied. Original: {original_z_range}, New: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Ground level correction
        ground_level = np.percentile(points[:, 2], 5)
        points[:, 2] = points[:, 2] - ground_level
        ground_level_reference = 0.0
        
        print(f"🔍 Debug: Ground correction applied. Ground level: {ground_level:.3f}")
        print(f"🔍 Debug: Final Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Update pointcloud
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Create orthographic projection
        depth_map, color_map, transform_info = pointcloud_to_orthographic_depth(pcd, resolution=512)
        
        if depth_map is None:
            return {"error": "Failed to create orthographic projection"}
        
        print(f"🔍 Debug: Orthographic projection created. Coverage: {(depth_map > 0).sum()}/{depth_map.size} pixels")
        
        # Detect tree crowns
        tree_regions, tree_threshold = detect_tree_crowns(depth_map)
        
        print(f"🔍 Debug: Tree detection completed. Found {len(tree_regions)} regions with threshold {tree_threshold:.3f}")
        
        if not tree_regions:
            return {
                "success": False,
                "message": f"No trees detected in the scene (threshold: {tree_threshold:.3f})",
                "measurements": [],
                "depth_map": depth_map,
                "tree_threshold": tree_threshold if 'tree_threshold' in locals() else 0,
                "debug_info": {
                    "total_points": total_points,
                    "z_range": f"[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]",
                    "depth_coverage": f"{(depth_map > 0).sum()}/{depth_map.size}"
                }
            }
        
        # Compute measurements
        crown_measurements = compute_tree_measurements(
            tree_regions, depth_map, transform_info, pcd, 
            ground_level_reference, scale_factor
        )
        
        # Save results
        results = {
            "success": True,
            "message": f"Successfully analyzed {len(crown_measurements)} tree(s)",
            "measurements": crown_measurements,
            "scale_factor": scale_factor,
            "depth_map": depth_map,
            "tree_threshold": tree_threshold,
            "num_trees": len(crown_measurements),
            "debug_info": {
                "total_points": total_points,
                "z_range": f"[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]",
                "depth_coverage": f"{(depth_map > 0).sum()}/{depth_map.size}"
            }
        }
        
        # Save measurements as JSON
        measurements_file = os.path.join(output_dir, "tree_measurements.json")
        with open(measurements_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_measurements = []
            for measurement in crown_measurements:
                json_measurement = {}
                for key, value in measurement.items():
                    if isinstance(value, np.ndarray):
                        json_measurement[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.float64)):
                        json_measurement[key] = float(value)
                    else:
                        json_measurement[key] = value
                json_safe_measurements.append(json_measurement)
            
            json.dump({
                "measurements": json_safe_measurements,
                "scale_factor": scale_factor,
                "num_trees": len(crown_measurements),
                "analysis_success": True,
                "debug_info": results["debug_info"]
            }, f, indent=2)
        
        print(f"🔍 Debug: Results saved to {measurements_file}")
        return results
        
    except Exception as e:
        import traceback
        error_msg = f"Tree analysis failed: {str(e)}"
        print(f"❌ Debug: {error_msg}")
        print(f"❌ Debug: Traceback: {traceback.format_exc()}")
        return {"error": error_msg}

def format_tree_measurements_html(analysis_results):
    """
    Format tree measurements as HTML for display in Gradio.
    """
    if "error" in analysis_results:
        return f"""
        <div style="background: #ffebee; border: 1px solid #f44336; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h3 style="color: #d32f2f; margin: 0 0 10px 0;">❌ Tree Analysis Error</h3>
            <p style="margin: 0; color: #666;">{analysis_results['error']}</p>
        </div>
        """
    
    if not analysis_results.get("success", False):
        return f"""
        <div style="background: #fff3e0; border: 1px solid #ff9800; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h3 style="color: #f57c00; margin: 0 0 10px 0;">⚠️ No Trees Detected</h3>
            <p style="margin: 0; color: #666;">{analysis_results.get('message', 'No trees found in the reconstruction')}</p>
        </div>
        """
    
    measurements = analysis_results.get("measurements", [])
    scale_factor = analysis_results.get("scale_factor")
    
    if not measurements:
        return """
        <div style="background: #fff3e0; border: 1px solid #ff9800; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h3 style="color: #f57c00; margin: 0 0 10px 0;">⚠️ No Measurements Available</h3>
            <p style="margin: 0; color: #666;">Tree analysis completed but no measurements could be computed.</p>
        </div>
        """
    
    html = f"""
    <div style="background: linear-gradient(145deg, #e8f5e8, #f1f8e9); border: 1px solid #4caf50; border-radius: 12px; padding: 20px; margin: 10px 0;">
        <h3 style="color: #2e7d32; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
            <span>🌳</span> Tree Analysis Results <span>📏</span>
        </h3>
        <p style="margin: 0 0 15px 0; color: #388e3c; font-weight: bold;">
            Successfully analyzed {len(measurements)} tree(s) in the scene
        </p>
    """
    
    if scale_factor:
        html += f"""
        <p style="margin: 0 0 15px 0; color: #1976d2; font-size: 14px; background: rgba(33, 150, 243, 0.1); padding: 8px; border-radius: 6px;">
            📐 Using GPS-based scale factor: {scale_factor:.3f} meters/unit
        </p>
        """
    
    for i, measurement in enumerate(measurements):
        tree_id = measurement.get('tree_id', i + 1)
        height = measurement.get('estimated_height', 0)
        diameter = measurement.get('crown_diameter', 0)
        area = measurement.get('area_real', 0)
        points = measurement.get('points_in_region', 0)
        
        html += f"""
        <div style="background: white; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #2e7d32; margin: 0 0 10px 0;">🌲 Tree {tree_id}</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
        """
        
        if scale_factor:
            height_m = measurement.get('height_meters', height * scale_factor)
            diameter_m = measurement.get('crown_diameter_meters', diameter * scale_factor)
            area_m2 = measurement.get('crown_area_meters2', area * (scale_factor ** 2))
            
            html += f"""
                <div style="background: #f3e5f5; padding: 10px; border-radius: 6px;">
                    <strong style="color: #7b1fa2;">📏 Height:</strong><br>
                    <span style="font-size: 18px; color: #4a148c;">{height_m:.2f} m</span>
                </div>
                <div style="background: #e1f5fe; padding: 10px; border-radius: 6px;">
                    <strong style="color: #0277bd;">🌿 Crown Diameter:</strong><br>
                    <span style="font-size: 18px; color: #01579b;">{diameter_m:.2f} m</span>
                </div>
                <div style="background: #e8f5e8; padding: 10px; border-radius: 6px;">
                    <strong style="color: #2e7d32;">🍃 Crown Area:</strong><br>
                    <span style="font-size: 18px; color: #1b5e20;">{area_m2:.2f} m²</span>
                </div>
            """
        else:
            html += f"""
                <div style="background: #f3e5f5; padding: 10px; border-radius: 6px;">
                    <strong style="color: #7b1fa2;">📏 Height:</strong><br>
                    <span style="font-size: 18px; color: #4a148c;">{height:.3f} units</span>
                </div>
                <div style="background: #e1f5fe; padding: 10px; border-radius: 6px;">
                    <strong style="color: #0277bd;">🌿 Crown Diameter:</strong><br>
                    <span style="font-size: 18px; color: #01579b;">{diameter:.3f} units</span>
                </div>
                <div style="background: #e8f5e8; padding: 10px; border-radius: 6px;">
                    <strong style="color: #2e7d32;">🍃 Crown Area:</strong><br>
                    <span style="font-size: 18px; color: #1b5e20;">{area:.4f} units²</span>
                </div>
            """
        
        html += f"""
                <div style="background: #fff3e0; padding: 10px; border-radius: 6px;">
                    <strong style="color: #ef6c00;">🔍 Points Analyzed:</strong><br>
                    <span style="font-size: 16px; color: #e65100;">{points:,}</span>
                </div>
            </div>
        </div>
        """
    
    if not scale_factor:
        html += """
        <div style="background: #fff3e0; border: 1px solid #ff9800; border-radius: 6px; padding: 10px; margin: 10px 0;">
            <small style="color: #ef6c00;">⚠️ No GPS-based scale available. Measurements are in Fast3R reconstruction units.</small>
        </div>
        """
    
    html += "</div>"
    return html
