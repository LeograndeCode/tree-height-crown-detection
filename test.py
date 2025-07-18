import os
import torch
import sys
import numpy as np
import argparse
import json
import pandas as pd
import random
import cv2
from scipy import ndimage

# Add fast3r directory to sys.path
# Assuming fast3r folder is in the same directory as this script
fast3r_path = os.path.join(os.path.dirname(__file__), 'fast3r')
if os.path.exists(fast3r_path):
    sys.path.append(fast3r_path)
else:
    # Alternative: if fast3r is in a different location, specify the full path
    # sys.path.append('/path/to/fast3r')
    raise ImportError(f"Fast3R not found at {fast3r_path}. Please ensure fast3r folder is in the same directory as this script.")

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.utils.checkpoint_utils import load_model
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

def detect_sky_mask(img_rgb):
    """
    Detect sky pixels using HSV color space and morphological operations.
    This is the same function used in the Gradio visualization.
    
    Args:
        img_rgb: RGB image normalized to [-1, 1]
    Returns:
        Boolean mask (as int8) where True indicates non-sky pixels.
    """
    img = ((img_rgb + 1) * 127.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105, 50, 140])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_light_blue = np.array([95, 5, 150])
    upper_light_blue = np.array([145, 100, 255])
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    lower_white = np.array([0, 0, 235])
    upper_white = np.array([180, 10, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = mask_blue | mask_light_blue | mask_white

    height = mask.shape[0]
    upper_third = int(height * 0.4)
    upper_region = hsv[:upper_third, :, :]
    mask[:upper_third, :] |= ((upper_region[:, :, 1] < 50) & (upper_region[:, :, 2] > 150))

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = mask.astype(bool)
    labels, num_labels = ndimage.label(mask)
    if num_labels > 0:
        top_row_labels = set(labels[0, :])
        top_row_labels.discard(0)
        if top_row_labels:
            mask = np.isin(labels, list(top_row_labels))
            labels, num_labels = ndimage.label(mask)
            if num_labels > 0:
                sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))
                mask_size = mask.size
                big_enough = sizes > mask_size * 0.01
                mask = np.isin(labels, np.where(big_enough)[0] + 1)
    return (~mask).astype(np.int8)

def run_fast3r_batch(img_paths, point_size=0.0004, min_conf_thr_percentile=85, global_conf_thr=1.5,
                     image_size=512, rotate_clockwise_90=False, crop_to_landscape=False,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # Load model
    checkpoint_dir = "jedyang97/Fast3R_ViT_Large_512"
    model, lit_module = load_model(checkpoint_dir, device=device, is_lightning_checkpoint=False)

    if not img_paths:
        raise ValueError("No images provided")

    # Load and preprocess images
    imgs = load_images(
        img_paths,
        size=image_size,
        verbose=True,
        rotate_clockwise_90=rotate_clockwise_90,
        crop_to_landscape=crop_to_landscape,
    )

    # Run inference
    output_dict = inference(
        imgs,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=False,
    )

    # Process predictions and move tensors to CPU.
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
        print(f"Warning: {e}")

    # Align local to global (using same parameters as Gradio interface)
    lit_module.align_local_pts3d_to_global(
        preds=output_dict['preds'],
        views=output_dict['views'],
        min_conf_thr_percentile=min_conf_thr_percentile
    )

    return output_dict

def explain_output_dict(output_dict):
    """
    Explain the structure of output_dict from Fast3R inference.
    
    output_dict contains:
    - 'views': List of input image data with metadata
    - 'preds': List of predictions for each image view
    - 'loss': Training loss (None during inference)
    """
    
    print("=" * 60)
    print("Fast3R OUTPUT_DICT STRUCTURE EXPLANATION")
    print("=" * 60)
    
    print(f"\n📝 Keys in output_dict: {list(output_dict.keys())}")
    print(f"📸 Number of views/images: {len(output_dict['views'])}")
    
    print("\n" + "=" * 40)
    print("1. VIEWS STRUCTURE")
    print("=" * 40)
    print("Contains the input images and their metadata:")
    
    for i, view in enumerate(output_dict['views']):
        print(f"\n  View {i} keys: {list(view.keys())}")
        if 'img' in view:
            print(f"    - img: {view['img'].shape} (preprocessed input image)")
        if 'true_shape' in view:
            print(f"    - true_shape: {view['true_shape']} (original image dimensions)")
        if 'idx' in view:
            print(f"    - idx: {view['idx']} (view index)")
    
    print("\n" + "=" * 40) 
    print("2. PREDICTIONS STRUCTURE")
    print("=" * 40)
    print("Contains the 3D reconstruction results for each view:")
    
    for i, pred in enumerate(output_dict['preds']):
        print(f"\n  Prediction {i} keys: {list(pred.keys())}")
        
        # Main pointcloud output
        if 'pts3d_in_other_view' in pred:
            shape = pred['pts3d_in_other_view'].shape
            print(f"    - pts3d_in_other_view: {shape}")
            print(f"      → 3D points in world coordinates (Height×Width×3)")
            print(f"      → This is your MAIN POINTCLOUD for view {i}")
        
        # Confidence map
        if 'conf' in pred:
            shape = pred['conf'].shape
            print(f"    - conf: {shape}")
            print(f"      → Confidence map for each 3D point")
        
        # Local predictions (if available)
        if 'pts3d_local' in pred:
            shape = pred['pts3d_local'].shape
            print(f"    - pts3d_local: {shape}")
            print(f"      → 3D points in local camera coordinates")
            
        if 'pts3d_local_aligned_to_global' in pred:
            shape = pred['pts3d_local_aligned_to_global'].shape
            print(f"    - pts3d_local_aligned_to_global: {shape}")
            print(f"      → Local points aligned to global coordinate system")
        
        if 'conf_local' in pred:
            shape = pred['conf_local'].shape
            print(f"    - conf_local: {shape}")
            print(f"      → Confidence map for local predictions")

def extract_camera_positions_and_pointcloud_gradio_style(output_dict, niter_PnP=100, min_conf_threshold_percentile=85):
    """
    Extract camera positions and final pointcloud from Fast3R output using the same approach as Gradio visualization.
    This includes sky masking, confidence sorting, and proper filtering.
    
    Returns:
    - camera_poses: List of 4x4 camera-to-world transformation matrices
    - estimated_focals: List of estimated focal lengths  
    - pointclouds: List of processed 3D point arrays for each view (sky-filtered, confidence-sorted)
    - confidence_maps: List of confidence arrays for each view
    - combined_pointcloud: Combined pointcloud from all views (sky-filtered)
    - combined_colors: Combined colors for the pointcloud
    - sky_filtered_data: Detailed per-view data matching Gradio visualization
    """
    
    print("\n" + "=" * 60)
    print("EXTRACTING POINTCLOUD USING GRADIO VISUALIZATION STYLE")
    print("=" * 60)
    
    # 1. Estimate camera poses using PnP
    print(f"\n🎯 Estimating camera poses using PnP (niter={niter_PnP})...")
    poses_c2w_batch, estimated_focals_batch = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=niter_PnP,
        focal_length_estimation_method='first_view_from_global_head'
    )
    
    # Extract from batch (assuming batch size = 1)
    camera_poses = poses_c2w_batch[0]  # List of 4x4 numpy arrays
    estimated_focals = estimated_focals_batch[0]  # List of focal lengths
    
    print(f"✅ Estimated {len(camera_poses)} camera poses")
    print(f"✅ Estimated focal lengths: {estimated_focals}")
    
    # 2. Process each view using Gradio visualization approach
    print(f"\n🌅 Processing each view with sky detection and confidence sorting...")
    
    frame_data_list = []
    all_combined_points = []
    all_combined_colors = []
    
    for i, (pred, view) in enumerate(zip(output_dict['preds'], output_dict['views'])):
        print(f"\n  Processing view {i}...")
        
        # Extract data from tensors (same as Gradio)
        img_rgb_orig = view['img'].cpu().squeeze().permute(1,2,0).numpy()
        pts3d_global = pred['pts3d_in_other_view'].cpu().squeeze().numpy().reshape(-1, 3)
        conf_global = pred['conf'].cpu().squeeze().numpy().flatten()
        img_rgb = view['img'].cpu().squeeze().permute(1,2,0).numpy()
        img_rgb_flat = img_rgb.reshape(-1, 3)
        
        # Detect sky mask (same as Gradio)
        not_sky_mask = detect_sky_mask(img_rgb_orig).flatten().astype(np.int8)
        print(f"    Sky detection: {np.sum(not_sky_mask == 0)} sky pixels, {np.sum(not_sky_mask == 1)} non-sky pixels")
        
        # Sort by confidence (highest first, same as Gradio)
        sort_idx_global = np.argsort(-conf_global)
        sorted_conf_global = conf_global[sort_idx_global]
        sorted_pts3d_global = pts3d_global[sort_idx_global]
        sorted_img_rgb_global = img_rgb_flat[sort_idx_global]
        sorted_not_sky_global = not_sky_mask[sort_idx_global]
        
        # Convert colors to [0,1] range (same as Gradio)
        colors_rgb_global = ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8) / 255.0
        
        # Apply confidence threshold
        conf_threshold = np.percentile(sorted_conf_global, min_conf_threshold_percentile)
        high_conf_mask = sorted_conf_global >= conf_threshold
        
        # Apply sky mask and confidence mask
        valid_mask = high_conf_mask & (sorted_not_sky_global == 1)
        
        # Filter points
        valid_points = sorted_pts3d_global[valid_mask]
        valid_colors = colors_rgb_global[valid_mask]
        valid_conf = sorted_conf_global[valid_mask]
        
        print(f"    Confidence threshold (P{min_conf_threshold_percentile}): {conf_threshold:.3f}")
        print(f"    High confidence points: {np.sum(high_conf_mask)}")
        print(f"    Sky-filtered + high confidence: {np.sum(valid_mask)} valid points")
        
        # Store processed data
        frame_data = {
            'view_idx': i,
            'sorted_pts3d_global': sorted_pts3d_global,
            'sorted_conf_global': sorted_conf_global,
            'colors_rgb_global': colors_rgb_global,
            'sorted_not_sky_global': sorted_not_sky_global,
            'valid_mask': valid_mask,
            'valid_points': valid_points,
            'valid_colors': valid_colors,
            'valid_conf': valid_conf,
            'conf_threshold': conf_threshold,
            'sky_ratio': 1.0 - np.mean(not_sky_mask),
            'max_conf': conf_global.max(),
            'img_shape': img_rgb_orig.shape[:2]
        }
        frame_data_list.append(frame_data)
        
        # Add to combined pointcloud
        if len(valid_points) > 0:
            all_combined_points.append(valid_points)
            all_combined_colors.append(valid_colors)
    
    # 3. Create final combined pointcloud
    print(f"\n🔗 Creating final combined pointcloud...")
    combined_pointcloud = np.vstack(all_combined_points) if all_combined_points else np.empty((0, 3))
    combined_colors = np.vstack(all_combined_colors) if all_combined_colors else np.empty((0, 3))
    
    total_valid_points = sum(len(fd['valid_points']) for fd in frame_data_list)
    print(f"✅ Final combined pointcloud: {combined_pointcloud.shape[0]} total points")
    print(f"✅ Average points per view: {total_valid_points / len(frame_data_list):.1f}")
    
    # 4. Summary statistics
    sky_ratios = [fd['sky_ratio'] for fd in frame_data_list]
    max_confs = [fd['max_conf'] for fd in frame_data_list]
    print(f"\n📊 Scene statistics:")
    print(f"    Sky ratios: min={min(sky_ratios):.3f}, max={max(sky_ratios):.3f}, avg={np.mean(sky_ratios):.3f}")
    print(f"    Max confidences: min={min(max_confs):.3f}, max={max(max_confs):.3f}, avg={np.mean(max_confs):.3f}")
    
    return {
        'camera_poses': camera_poses,  # List of 4x4 camera-to-world matrices
        'estimated_focals': estimated_focals,  # List of focal lengths
        'frame_data': frame_data_list,  # Detailed per-view data
        'combined_pointcloud': combined_pointcloud,  # (N, 3) array - sky filtered
        'combined_colors': combined_colors,  # (N, 3) array - sky filtered
        'processing_params': {
            'min_conf_threshold_percentile': min_conf_threshold_percentile,
            'niter_PnP': niter_PnP,
            'sky_filtering_enabled': True
        }
    }

def save_results_gradio_style(results, output_folder="output"):
    """Save the extracted results to files (Gradio visualization style)."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save camera poses
    camera_poses_file = os.path.join(output_folder, "camera_poses.npy")
    np.save(camera_poses_file, np.array(results['camera_poses']))
    print(f"💾 Saved camera poses to {camera_poses_file}")
    
    # Save combined pointcloud (sky-filtered)
    pointcloud_file = os.path.join(output_folder, "combined_pointcloud_sky_filtered.npy")
    np.save(pointcloud_file, results['combined_pointcloud'])
    print(f"💾 Saved sky-filtered combined pointcloud to {pointcloud_file}")
    
    # Save colors
    colors_file = os.path.join(output_folder, "combined_colors_sky_filtered.npy")
    np.save(colors_file, results['combined_colors'])
    print(f"💾 Saved sky-filtered pointcloud colors to {colors_file}")
    
    # Save detailed frame data
    frame_data_file = os.path.join(output_folder, "frame_data_gradio_style.npy")
    np.save(frame_data_file, results['frame_data'], allow_pickle=True)
    print(f"💾 Saved detailed frame data to {frame_data_file}")
    
    # Save processing parameters
    params_file = os.path.join(output_folder, "processing_params.json")
    with open(params_file, 'w') as f:
        json.dump(results['processing_params'], f, indent=2)
    print(f"💾 Saved processing parameters to {params_file}")
    
    # Save as PLY file for visualization
    try:
        import trimesh
        point_cloud = trimesh.PointCloud(
            vertices=results['combined_pointcloud'],
            colors=(results['combined_colors'] * 255).astype(np.uint8)  # Convert to 0-255 range
        )
        ply_file = os.path.join(output_folder, "reconstruction_sky_filtered.ply")
        point_cloud.export(ply_file)
        print(f"💾 Saved sky-filtered PLY file to {ply_file}")
        
        # Also save per-view PLY files
        for i, fd in enumerate(results['frame_data']):
            if len(fd['valid_points']) > 0:
                view_cloud = trimesh.PointCloud(
                    vertices=fd['valid_points'],
                    colors=(fd['valid_colors'] * 255).astype(np.uint8)
                )
                view_ply_file = os.path.join(output_folder, f"view_{i:02d}_sky_filtered.ply")
                view_cloud.export(view_ply_file)
                print(f"💾 Saved view {i} PLY file to {view_ply_file}")
                
    except ImportError:
        print("⚠️ trimesh not available, skipping PLY export")
        print("   Install with: pip install trimesh")
    
    # Print summary
    print(f"\n📊 Sky-filtered reconstruction summary:")
    print(f"   Total points: {len(results['combined_pointcloud'])}")
    print(f"   Views processed: {len(results['frame_data'])}")
    avg_sky_ratio = np.mean([fd['sky_ratio'] for fd in results['frame_data']])
    print(f"   Average sky ratio: {avg_sky_ratio:.1%}")
    print(f"   Confidence threshold used: P{results['processing_params']['min_conf_threshold_percentile']}")
    
    return output_folder

def load_coordinates_from_csv(csv_file):
    """
    Load real-world coordinates from a CSV file.
    
    Expected CSV format:
    image_name,timestamp,latitude,longitude,altitude,relative_alt,x_m,y_m,z_m
    img_0000.jpg,2025-07-16 21:11:40,38.6344281,-90.227515,154.69,2.576,-0.5396,1.9249,0.6250
    img_0001.jpg,2025-07-16 21:11:42,38.634425199999995,-90.22748589999999,155.059,2.945,-0.8758,0.7347,0.2530
    
    Args:
        csv_file: Path to CSV file containing coordinates
        
    Returns:
        dict: Dictionary mapping image names to real-world coordinates
    """
    try:
        df = pd.read_csv(csv_file)
        coord_data = {}
        
        for _, row in df.iterrows():
            coord_data[row['image_name']] = {
                'x': row['x_m'],
                'y': row['y_m'], 
                'z': row['z_m'],
                'lat': row['latitude'],
                'lon': row['longitude'],
                'alt': row['altitude']
            }
        
        return coord_data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def select_random_images(input_folder, csv_file, num_images=5):
    """
    Select specific images (0004, 0005, 0006, 0007) that exist in both the folder and CSV file.
    
    Args:
        input_folder: Path to folder containing images
        csv_file: Path to CSV file with coordinates
        num_images: Number of images to select (default: 5, but will use specific images)
        
    Returns:
        tuple: (selected_img_paths, coord_data_subset)
    """
    # Load coordinate data
    coord_data = load_coordinates_from_csv(csv_file)
    if coord_data is None:
        return None, None
    
    # Define specific images to select
    target_images = ["img_0004.jpg", "img_0005.jpg", "img_0006.jpg", "img_0007.jpg"]
    
    # Get all available images in folder
    all_img_files = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Filter target images that exist in both folder and CSV
    selected_images = []
    for target_img in target_images:
        if target_img in all_img_files and target_img in coord_data:
            selected_images.append(target_img)
        else:
            print(f"Warning: {target_img} not found in folder or CSV data")
    
    if len(selected_images) == 0:
        print(f"❌ None of the target images {target_images} found in both folder and CSV")
        return None, None
    
    selected_img_paths = [os.path.join(input_folder, img) for img in selected_images]
    
    # Create subset of coordinate data
    coord_data_subset = {img: coord_data[img] for img in selected_images}
    
    print(f"Selected {len(selected_images)} specific images:")
    for img in selected_images:
        coord = coord_data[img]
        print(f"  {img}: x={coord['x']:.3f}, y={coord['y']:.3f}, z={coord['z']:.3f}")
    
    return selected_img_paths, coord_data_subset

def compute_real_world_scale(camera_poses, img_paths, coord_data, min_pairs=2):
    """
    Compute the real-world scale of the reconstruction by comparing
    camera pose distances with real-world distances from CSV data.
    
    Args:
        camera_poses: List of 4x4 camera-to-world transformation matrices
        img_paths: List of image file paths (same order as camera_poses)
        coord_data: Dictionary mapping image names to real-world coordinates
        min_pairs: Minimum number of camera pairs to use for scale estimation
        
    Returns:
        dict: Dictionary containing scale factor and statistics
    """
    print("\n" + "=" * 60)
    print("COMPUTING REAL-WORLD SCALE FROM CSV COORDINATES")
    print("=" * 60)
    
    if coord_data is None:
        print("❌ No coordinate data available")
        return None
    
    # Match images with coordinates
    matched_data = []
    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        if img_name in coord_data:
            coord = coord_data[img_name]
            camera_pos = camera_poses[i][:3, 3]  # Extract camera position
            real_pos = np.array([coord['x'], coord['y'], coord['z']])
            matched_data.append({
                'img_name': img_name,
                'camera_pos': camera_pos,
                'real_pos': real_pos,
                'coord': coord
            })
        else:
            print(f"⚠️ No coordinate data found for {img_name}")
    
    if len(matched_data) < 2:
        print(f"❌ Need at least 2 images with coordinate data, found {len(matched_data)}")
        return None
    
    print(f"✅ Matched {len(matched_data)} images with real-world coordinates")
    
    # Compute pairwise distances
    scale_ratios = []
    pair_info = []
    
    for i in range(len(matched_data)):
        for j in range(i + 1, len(matched_data)):
            data_i = matched_data[i]
            data_j = matched_data[j]
            
            # Distance in Fast3R coordinate system (world units)
            fast3r_dist = np.linalg.norm(data_i['camera_pos'] - data_j['camera_pos'])
            
            # Distance in real world (meters)
            real_dist = np.linalg.norm(data_i['real_pos'] - data_j['real_pos'])
            
            if fast3r_dist > 1e-6 and real_dist > 1e-6:  # Avoid division by zero
                scale_ratio = real_dist / fast3r_dist
                scale_ratios.append(scale_ratio)
                pair_info.append({
                    'img1': data_i['img_name'],
                    'img2': data_j['img_name'],
                    'fast3r_dist': fast3r_dist,
                    'real_dist': real_dist,
                    'scale_ratio': scale_ratio
                })
                
                print(f"  {data_i['img_name']} ↔ {data_j['img_name']}:")
                print(f"    Fast3R distance: {fast3r_dist:.6f} units")
                print(f"    Real distance: {real_dist:.3f} meters")
                print(f"    Scale ratio: {scale_ratio:.3f} meters/unit")
    
    if len(scale_ratios) < min_pairs:
        print(f"❌ Need at least {min_pairs} valid pairs, found {len(scale_ratios)}")
        return None
    
    # Compute statistics
    scale_ratios = np.array(scale_ratios)
    mean_scale = np.mean(scale_ratios)
    std_scale = np.std(scale_ratios)
    median_scale = np.median(scale_ratios)
    
    print(f"\n📊 SCALE ESTIMATION RESULTS:")
    print(f"  Number of camera pairs: {len(scale_ratios)}")
    print(f"  Mean scale: {mean_scale:.3f} ± {std_scale:.3f} meters/unit")
    print(f"  Median scale: {median_scale:.3f} meters/unit")
    print(f"  Min scale: {np.min(scale_ratios):.3f} meters/unit")
    print(f"  Max scale: {np.max(scale_ratios):.3f} meters/unit")
    
    # Quality assessment
    cv = std_scale / mean_scale * 100  # Coefficient of variation
    print(f"  Coefficient of variation: {cv:.1f}%")
    
    if cv < 10:
        quality = "Excellent"
    elif cv < 20:
        quality = "Good"
    elif cv < 50:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"  Scale estimation quality: {quality}")
    
    return {
        'scale_factor': mean_scale,
        'scale_std': std_scale,
        'scale_median': median_scale,
        'scale_min': np.min(scale_ratios),
        'scale_max': np.max(scale_ratios),
        'coefficient_variation': cv,
        'quality': quality,
        'num_pairs': len(scale_ratios),
        'pair_details': pair_info,
        'matched_images': matched_data
    }

def apply_real_world_scale(pointcloud, scale_factor):
    """
    Apply real-world scale to the point cloud.
    
    Args:
        pointcloud: (N, 3) numpy array of 3D points
        scale_factor: Scale factor in meters/unit
        
    Returns:
        numpy.array: Scaled point cloud in real-world coordinates (meters)
    """
    return pointcloud * scale_factor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fast3R Point Cloud Reconstruction with Real-World Scaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_folder",
        help="Path to folder containing input images"
    )
    
    # Optional parameters
    parser.add_argument(
        "--point_size", 
        type=float, 
        default=0.0004,
        help="Point size for visualization"
    )
    parser.add_argument(
        "--min_conf_thr_percentile", 
        type=int, 
        default=85,
        help="Minimum confidence threshold percentile (0-100) - Using 85 to match Gradio visualization"
    )
    parser.add_argument(
        "--global_conf_thr", 
        type=float, 
        default=1.5,
        help="Global confidence threshold"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=512,
        choices=[224, 512],
        help="Image resolution for processing"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default="output",
        help="Output folder for results"
    )
    parser.add_argument(
        "--csv_file", 
        type=str,
        help="Path to CSV file containing real-world coordinates for scale estimation"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=5,
        help="Number of random images to select for processing (default: 5)"
    )
    parser.add_argument(
        "--rotate_clockwise_90", 
        action="store_true",
        help="Rotate images 90 degrees clockwise"
    )
    parser.add_argument(
        "--crop_to_landscape", 
        action="store_true",
        help="Crop images to landscape orientation"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Running Fast3R inference on {device}...")
    print(f"📁 Input folder: {args.input_folder}")
    print(f"🎛️ Parameters: point_size={args.point_size}, min_conf_thr_percentile={args.min_conf_thr_percentile}, global_conf_thr={args.global_conf_thr}")
    
    # Select specific images if CSV file is provided
    if args.csv_file:
        print(f"\n� Selecting specific images (0004, 0005, 0006, 0007) with CSV coordinates...")
        img_paths, coord_data = select_random_images(args.input_folder, args.csv_file, args.num_images)
        if img_paths is None:
            print("❌ Failed to select images. Exiting.")
            sys.exit(1)
    else:
        # Use all images if no CSV file provided
        print(f"\n📸 Using all available images (no CSV file provided)...")
        img_paths = [os.path.join(args.input_folder, f) for f in sorted(os.listdir(args.input_folder))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        coord_data = None
    
    # Run Fast3R
    result = run_fast3r_batch(
        img_paths,
        point_size=args.point_size,
        min_conf_thr_percentile=args.min_conf_thr_percentile,
        global_conf_thr=args.global_conf_thr,
        image_size=args.image_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
        crop_to_landscape=args.crop_to_landscape,
        device=device
    )
    
    # Explain the output structure
    explain_output_dict(result)
    
    # Extract using Gradio visualization style (with sky filtering)
    extracted = extract_camera_positions_and_pointcloud_gradio_style(result, min_conf_threshold_percentile=args.min_conf_thr_percentile)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - WHAT YOU NEED")
    print("=" * 60)
    
    print(f"\n📷 CAMERA POSITIONS (ordered by input images):")
    for i, pose in enumerate(extracted['camera_poses']):
        img_name = os.path.basename(img_paths[i]) if i < len(img_paths) else f"image_{i}"
        print(f"  Camera {i} ({img_name}):")
        print(f"    Position: {pose[:3, 3]} (world coordinates)")
        print(f"    Focal length: {extracted['estimated_focals'][i]:.1f}")
    
    print(f"\n🗃️ POINTCLOUD (Fast3R coordinates):")
    print(f"  Total points: {extracted['combined_pointcloud'].shape[0]}")
    print(f"  Coordinate range:")
    if extracted['combined_pointcloud'].shape[0] > 0:
        mins = extracted['combined_pointcloud'].min(axis=0)
        maxs = extracted['combined_pointcloud'].max(axis=0)
        print(f"    X: [{mins[0]:.3f}, {maxs[0]:.3f}]")
        print(f"    Y: [{mins[1]:.3f}, {maxs[1]:.3f}]")
        print(f"    Z: [{mins[2]:.3f}, {maxs[2]:.3f}]")
    
    # GPS-based scale estimation (if CSV file provided)
    scale_result = None
    if args.csv_file and coord_data:
        scale_result = compute_real_world_scale(
            extracted['camera_poses'], 
            img_paths, 
            coord_data
        )
        
        if scale_result:
            # Apply scale to pointcloud
            scaled_pointcloud = apply_real_world_scale(
                extracted['combined_pointcloud'], 
                scale_result['scale_factor']
            )
            
            print(f"\n🌍 REAL-WORLD SCALED POINTCLOUD:")
            print(f"  Scale factor: {scale_result['scale_factor']:.2f} meters/unit")
            print(f"  Total points: {scaled_pointcloud.shape[0]}")
            if scaled_pointcloud.shape[0] > 0:
                mins = scaled_pointcloud.min(axis=0)
                maxs = scaled_pointcloud.max(axis=0)
                print(f"  Real-world coordinate range (meters):")
                print(f"    X: [{mins[0]:.2f}, {maxs[0]:.2f}]")
                print(f"    Y: [{mins[1]:.2f}, {maxs[1]:.2f}]")
                print(f"    Z: [{mins[2]:.2f}, {maxs[2]:.2f}]")
            
            # Add scaled pointcloud to results
            extracted['scaled_pointcloud'] = scaled_pointcloud
            extracted['scale_info'] = scale_result
    
    # Save results using Gradio-style function
    save_results_gradio_style(extracted, args.output_folder)
    
    # Save additional GPS/scale information if available
    if scale_result:
        scale_file = os.path.join(args.output_folder, "scale_estimation.json")
        with open(scale_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            scale_data = scale_result.copy()
            for pair in scale_data['pair_details']:
                if 'fast3r_dist' in pair:
                    pair['fast3r_dist'] = float(pair['fast3r_dist'])
                if 'real_dist' in pair:
                    pair['real_dist'] = float(pair['real_dist'])
                if 'scale_ratio' in pair:
                    pair['scale_ratio'] = float(pair['scale_ratio'])
            
            for match in scale_data['matched_images']:
                match['camera_pos'] = match['camera_pos'].tolist()
                match['real_pos'] = match['real_pos'].tolist()
            
            json.dump(scale_data, f, indent=2)
        print(f"💾 Saved scale estimation results to {scale_file}")
        
        # Save scaled pointcloud
        if 'scaled_pointcloud' in extracted:
            scaled_pc_file = os.path.join(args.output_folder, "scaled_pointcloud.npy")
            np.save(scaled_pc_file, extracted['scaled_pointcloud'])
            print(f"💾 Saved scaled pointcloud to {scaled_pc_file}")
    
    print(f"\n✅ Done! Check the '{args.output_folder}' folder for saved files.")
    
    # Summary of changes from basic test.py to Gradio-style processing
    print(f"\n" + "=" * 60)
    print("🎨 GRADIO VISUALIZATION STYLE PROCESSING APPLIED")
    print("=" * 60)
    print("Key improvements made to match Gradio interface:")
    print("✅ Sky detection and filtering - removes sky pixels automatically")
    print("✅ Higher confidence threshold (85th percentile vs 10th percentile)")
    print("✅ Confidence-based sorting (highest confidence first)")
    print("✅ Proper color normalization [0,1] range")
    print("✅ Per-view processing matching visualization pipeline")
    print("✅ Same alignment parameters as Gradio (min_conf_thr_percentile=85)")
    print("")
    print("Your pointcloud should now have:")
    print("• Correct orientation (not upside down)")  
    print("• Sky pixels removed")
    print("• Higher quality points (better confidence filtering)")
    print("• Colors that match the Gradio visualization")
    print("")
    print("Files saved:")
    print("• reconstruction_sky_filtered.ply - Main PLY file for viewing")
    print("• combined_pointcloud_sky_filtered.npy - Numpy array of 3D points")
    print("• combined_colors_sky_filtered.npy - Numpy array of colors")
    print("• view_XX_sky_filtered.ply - Individual view PLY files")
    print("• frame_data_gradio_style.npy - Detailed processing data")
    print("• processing_params.json - Parameters used")
    
    if args.csv_file and not scale_result:
        print(f"\n⚠️ CSV-based scaling failed. Check your CSV file format and image names.")
        print(f"Expected CSV file format:")
        print(f"image_name,timestamp,latitude,longitude,altitude,relative_alt,x_m,y_m,z_m")
        print(f"img_0000.jpg,2025-07-16 21:11:40,38.6344281,-90.227515,154.69,2.576,-0.5396,1.9249,0.6250")