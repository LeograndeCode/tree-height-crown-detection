# Fast3R Point Cloud Reconstruction with Real-World Scaling

This enhanced script provides Fast3R point cloud reconstruction with optional real-world scale estimation using GPS coordinates.

## Features

- **Command-line interface** with customizable parameters
- **GPS-based scale estimation** for real-world measurements
- **Camera pose extraction** with precise image-to-pose correspondence
- **Combined point cloud generation** with confidence filtering
- **Multiple output formats** (NumPy arrays, PLY files, JSON metadata)

## Usage

### Basic Usage

```bash
# Basic reconstruction with default parameters
python test.py images/

# Custom parameters
python test.py images/ --point_size 0.001 --min_conf_thr_percentile 20 --global_conf_thr 2.0

# Different image resolution for faster processing
python test.py images/ --image_size 224

# Save to custom output folder
python test.py images/ --output_folder my_results/
```

### With GPS-based Real-World Scaling

```bash
# Reconstruction with real-world scale estimation
python test.py images/ --gps_file gps_coordinates.json

# Combined with custom parameters
python test.py images/ --gps_file gps_coordinates.json --min_conf_thr_percentile 15 --output_folder scaled_results/
```

## Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_folder` | str | - | **Required**: Path to folder containing input images |
| `--point_size` | float | 0.0004 | Point size for visualization |
| `--min_conf_thr_percentile` | int | 10 | Confidence threshold percentile (0-100) |
| `--global_conf_thr` | float | 1.5 | Global confidence threshold |
| `--image_size` | int | 512 | Image resolution (224 or 512) |
| `--output_folder` | str | "output" | Output folder for results |
| `--gps_file` | str | None | Path to JSON file with GPS coordinates |
| `--rotate_clockwise_90` | flag | False | Rotate images 90° clockwise |
| `--crop_to_landscape` | flag | False | Crop images to landscape orientation |
| `--device` | str | "auto" | Device to use ("auto", "cuda", "cpu") |

## GPS Coordinates File Format

Create a JSON file with GPS coordinates for each image:

```json
{
  "image1.jpg": {
    "lat": 40.7128,
    "lon": -74.0060,
    "alt": 10.0
  },
  "image2.jpg": {
    "lat": 40.7130,
    "lon": -74.0058,
    "alt": 12.0
  }
}
```

### GPS Coordinate Requirements

- **lat**: Latitude in decimal degrees
- **lon**: Longitude in decimal degrees  
- **alt**: Altitude in meters (optional, defaults to 0)
- **Image names**: Must exactly match the filenames in your input folder

## Extracting GPS from Images (Optional)

If your images contain GPS metadata, use the provided extraction script:

```bash
# Extract GPS from all images in a folder
python extract_gps_from_images.py images/ gps_coordinates.json

# Requirements for GPS extraction
pip install pillow pillow-heif exifread
```

## Output Files

The script generates several output files:

### Standard Outputs
- `camera_poses.npy`: 4×4 camera-to-world transformation matrices
- `combined_pointcloud.npy`: Combined 3D point cloud (Fast3R coordinates)
- `combined_colors.npy`: RGB colors for each point
- `reconstruction.ply`: Point cloud in PLY format for visualization

### GPS-Enhanced Outputs (when GPS file provided)
- `scaled_pointcloud.npy`: Point cloud scaled to real-world coordinates (meters)
- `scale_estimation.json`: Detailed scale estimation results and statistics

## Understanding the Output

### Camera Poses
Each camera pose is a 4×4 transformation matrix where:
- `pose[:3, 3]` = Camera position in world coordinates
- `pose[:3, :3]` = Camera rotation matrix
- `pose[i]` corresponds to the i-th image (alphabetically sorted)

### Point Cloud Coordinates
- **Fast3R coordinates**: Arbitrary units from the reconstruction
- **Real-world coordinates**: Scaled to meters using GPS data (if provided)

### Scale Estimation Quality
The script provides quality assessment:
- **Excellent** (CV < 10%): Very reliable scale
- **Good** (CV < 20%): Reliable scale  
- **Fair** (CV < 50%): Usable but check carefully
- **Poor** (CV > 50%): Scale may be unreliable

## Examples

### Example 1: Basic Reconstruction
```bash
python test.py ./my_images/ --min_conf_thr_percentile 20
```

### Example 2: High-Quality Reconstruction
```bash
python test.py ./my_images/ \
    --image_size 512 \
    --min_conf_thr_percentile 5 \
    --point_size 0.0002 \
    --output_folder high_quality_results/
```

### Example 3: Real-World Scaled Reconstruction
```bash
# First extract GPS (if needed)
python extract_gps_from_images.py ./my_images/ gps_coords.json

# Then run reconstruction with scaling
python test.py ./my_images/ \
    --gps_file gps_coords.json \
    --min_conf_thr_percentile 15 \
    --output_folder scaled_reconstruction/
```

### Example 4: Fast Processing
```bash
python test.py ./my_images/ \
    --image_size 224 \
    --min_conf_thr_percentile 30 \
    --device cuda
```

## Tips for Best Results

1. **Image Quality**: Use well-lit, sharp images with good overlap
2. **GPS Accuracy**: Ensure GPS coordinates are accurate (< 5m error recommended)
3. **Scale Estimation**: Need at least 3-4 images with GPS for reliable scaling
4. **Confidence Filtering**: Start with `min_conf_thr_percentile=10`, increase if too noisy
5. **Memory Management**: Use `image_size=224` for large image sets to reduce memory usage

## Troubleshooting

### No GPS Data Found
- Check that image filenames in GPS file exactly match actual filenames
- Verify GPS coordinates are in decimal degrees format
- Ensure at least 2 images have valid GPS data

### Poor Scale Estimation
- Check GPS coordinate accuracy
- Ensure sufficient distance between camera positions
- Verify that images are properly matched with GPS coordinates

### Memory Issues
- Reduce `image_size` to 224
- Process fewer images at once
- Use CPU if GPU memory is insufficient

## Requirements

```bash
pip install torch torchvision numpy pillow
# For GPS extraction (optional):
pip install pillow-heif exifread
```
