import os
import torch
import sys
import numpy as np
import argparse
import json
import pandas as pd
import random
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

def run_fast3r_batch(img_paths, point_size=0.0004, min_conf_thr_percentile=10, global_conf_thr=1.5,
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

    # Align local to global
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

def extract_camera_positions_and_pointcloud(output_dict, niter_PnP=100, min_conf_threshold_percentile=10):
    """
    Extract camera positions and final pointcloud from Fast3R output.
    
    Returns:
    - camera_poses: List of 4x4 camera-to-world transformation matrices
    - estimated_focals: List of estimated focal lengths  
    - pointclouds: List of 3D point arrays for each view
    - confidence_maps: List of confidence arrays for each view
    - combined_pointcloud: Combined pointcloud from all views
    """
    
    print("\n" + "=" * 60)
    print("EXTRACTING CAMERA POSITIONS AND POINTCLOUD")
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
    
    # 2. Extract pointclouds for each view
    print(f"\n🗃️ Extracting pointclouds for each view...")
    pointclouds = []
    confidence_maps = []
    
    for i, pred in enumerate(output_dict['preds']):
        # Get 3D points (in world coordinates)
        pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  # Shape: (H, W, 3)
        conf = pred['conf'].cpu().numpy().squeeze()  # Shape: (H, W)
        
        pointclouds.append(pts3d)
        confidence_maps.append(conf)
        
        print(f"  View {i}: {pts3d.shape} points, confidence range [{conf.min():.3f}, {conf.max():.3f}]")
    
    # 3. Create combined pointcloud with confidence filtering
    print(f"\n🔗 Creating combined pointcloud (conf > {min_conf_threshold_percentile}th percentile)...")
    combined_points = []
    combined_colors = []
    
    for i, (pts3d, conf) in enumerate(zip(pointclouds, confidence_maps)):
        # Get corresponding RGB colors from input image
        img_rgb = output_dict['views'][i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)
        # Convert from [-1, 1] to [0, 255]
        img_rgb = ((img_rgb + 1) * 127.5).astype(np.uint8).clip(0, 255)
        
        # Apply confidence threshold
        conf_threshold = np.percentile(conf, min_conf_threshold_percentile)
        mask = conf > conf_threshold
        
        # Flatten and filter
        valid_points = pts3d[mask]  # Shape: (N_valid, 3)
        valid_colors = img_rgb[mask]  # Shape: (N_valid, 3)
        
        combined_points.append(valid_points)
        combined_colors.append(valid_colors)
        
        print(f"  View {i}: {valid_points.shape[0]} valid points (threshold: {conf_threshold:.3f})")
    
    # Combine all points
    combined_pointcloud = np.vstack(combined_points) if combined_points else np.empty((0, 3))
    combined_pointcloud_colors = np.vstack(combined_colors) if combined_colors else np.empty((0, 3))
    
    print(f"✅ Combined pointcloud: {combined_pointcloud.shape[0]} total points")
    
    return {
        'camera_poses': camera_poses,  # List of 4x4 camera-to-world matrices
        'estimated_focals': estimated_focals,  # List of focal lengths
        'pointclouds_per_view': pointclouds,  # List of (H,W,3) arrays
        'confidence_per_view': confidence_maps,  # List of (H,W) arrays  
        'combined_pointcloud': combined_pointcloud,  # (N, 3) array
        'combined_colors': combined_pointcloud_colors,  # (N, 3) array
    }

def save_results(results, output_folder="output"):
    """Save the extracted results to files."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Save camera poses
    camera_poses_file = os.path.join(output_folder, "camera_poses.npy")
    np.save(camera_poses_file, np.array(results['camera_poses']))
    print(f"💾 Saved camera poses to {camera_poses_file}")
    
    # Save combined pointcloud
    pointcloud_file = os.path.join(output_folder, "combined_pointcloud.npy")
    np.save(pointcloud_file, results['combined_pointcloud'])
    print(f"💾 Saved combined pointcloud to {pointcloud_file}")
    
    # Save colors
    colors_file = os.path.join(output_folder, "combined_colors.npy")
    np.save(colors_file, results['combined_colors'])
    print(f"💾 Saved pointcloud colors to {colors_file}")
    
    # Save as PLY file for visualization
    try:
        import trimesh
        point_cloud = trimesh.PointCloud(
            vertices=results['combined_pointcloud'],
            colors=results['combined_colors']
        )
        ply_file = os.path.join(output_folder, "reconstruction.ply")
        point_cloud.export(ply_file)
        print(f"💾 Saved PLY file to {ply_file}")
    except ImportError:
        print("⚠️ trimesh not available, skipping PLY export")

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

def select_random_images(input_folder, csv_file, num_images=4):
    """
    Select random images that exist in both the folder and CSV file.
    
    Args:
        input_folder: Path to folder containing images
        csv_file: Path to CSV file with coordinates
        num_images: Number of images to select (default: 4)
        
    Returns:
        tuple: (selected_img_paths, coord_data_subset)
    """
    # Load coordinate data
    coord_data = load_coordinates_from_csv(csv_file)
    if coord_data is None:
        return None, None
    
    # Get all available images in folder
    all_img_files = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Filter images that have coordinates
    available_images = [img for img in all_img_files if img in coord_data]
    
    if len(available_images) < num_images:
        print(f"Warning: Only {len(available_images)} images available, requested {num_images}")
        num_images = len(available_images)
    
    # Randomly select images
    selected_images = random.sample(available_images, num_images)
    selected_img_paths = [os.path.join(input_folder, img) for img in selected_images]
    
    # Create subset of coordinate data
    coord_data_subset = {img: coord_data[img] for img in selected_images}
    
    print(f"Selected {len(selected_images)} random images:")
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
        default=10,
        help="Minimum confidence threshold percentile (0-100)"
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
        default=4,
        help="Number of random images to select for processing (default: 4)"
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
    
    # Run Fast3R
    result = run_fast3r_batch(
        args.input_folder,
        point_size=args.point_size,
        min_conf_thr_percentile=args.min_conf_thr_percentile,
        global_conf_thr=args.global_conf_thr,
        image_size=args.image_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
        crop_to_landscape=args.crop_to_landscape,
        device=device
    )
    
    # Get the image paths for reference
    img_paths = [os.path.join(args.input_folder, f) for f in sorted(os.listdir(args.input_folder))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Explain the output structure
    explain_output_dict(result)
    
    # Extract what you need
    extracted = extract_camera_positions_and_pointcloud(result, min_conf_threshold_percentile=args.min_conf_thr_percentile)
    
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
    
    # GPS-based scale estimation (if GPS file provided)
    scale_result = None
    if args.gps_file:
        scale_result = compute_real_world_scale(
            extracted['camera_poses'], 
            img_paths, 
            args.gps_file
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
    
    # Save results
    save_results(extracted, args.output_folder)
    
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
    
    if args.gps_file and not scale_result:
        print(f"\n⚠️ GPS-based scaling failed. Check your GPS file format and image names.")
        print(f"Expected GPS file format (JSON):")
        print(f'{{')
        print(f'  "image1.jpg": {{"lat": 40.7128, "lon": -74.0060, "alt": 10.0}},')
        print(f'  "image2.jpg": {{"lat": 40.7589, "lon": -73.9851, "alt": 15.0}}')
        print(f'}}')