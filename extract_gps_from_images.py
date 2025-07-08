#!/usr/bin/env python3
"""
GPS Extraction Script for Fast3R Scale Estimation

This script helps extract GPS coordinates from image EXIF data and create
the JSON file needed for real-world scale estimation in Fast3R.

Requirements:
    pip install pillow pillow-heif exifread

Usage:
    python extract_gps_from_images.py /path/to/images/ output_gps.json
"""

import os
import json
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_gps_from_exif(image_path):
    """
    Extract GPS coordinates from image EXIF data.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict or None: GPS coordinates as {'lat': float, 'lon': float, 'alt': float}
    """
    try:
        with Image.open(image_path) as image:
            exifdata = image.getexif()
            
            if not exifdata:
                return None
                
            gps_info = None
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_info = exifdata[tag_id]
                    break
            
            if not gps_info:
                return None
            
            # Parse GPS data
            gps_data = {}
            for key in gps_info.keys():
                name = GPSTAGS.get(key, key)
                gps_data[name] = gps_info[key]
            
            # Extract latitude
            if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                lat = convert_to_degrees(gps_data['GPSLatitude'])
                if gps_data['GPSLatitudeRef'] != 'N':
                    lat = -lat
            else:
                return None
            
            # Extract longitude
            if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                lon = convert_to_degrees(gps_data['GPSLongitude'])
                if gps_data['GPSLongitudeRef'] != 'E':
                    lon = -lon
            else:
                return None
            
            # Extract altitude (optional)
            alt = 0.0
            if 'GPSAltitude' in gps_data:
                alt = float(gps_data['GPSAltitude'])
                if 'GPSAltitudeRef' in gps_data and gps_data['GPSAltitudeRef'] == 1:
                    alt = -alt  # Below sea level
            
            return {
                'lat': lat,
                'lon': lon,
                'alt': alt
            }
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def convert_to_degrees(value):
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.
    
    Args:
        value: GPS coordinate in DMS format
        
    Returns:
        float: Coordinate in decimal degrees
    """
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def extract_gps_from_folder(folder_path, output_file):
    """
    Extract GPS coordinates from all images in a folder.
    
    Args:
        folder_path: Path to folder containing images
        output_file: Path to output JSON file
    """
    print(f"🔍 Scanning images in: {folder_path}")
    
    gps_data = {}
    image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.heic', '.heif')
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"  Processing: {filename}")
            
            gps_coords = get_gps_from_exif(image_path)
            if gps_coords:
                gps_data[filename] = gps_coords
                print(f"    ✅ GPS found: {gps_coords['lat']:.6f}, {gps_coords['lon']:.6f}, alt={gps_coords['alt']:.1f}m")
            else:
                print(f"    ❌ No GPS data found")
    
    if gps_data:
        with open(output_file, 'w') as f:
            json.dump(gps_data, f, indent=2)
        print(f"\n✅ Saved GPS data for {len(gps_data)} images to: {output_file}")
    else:
        print(f"\n❌ No GPS data found in any images")
    
    return gps_data

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_gps_from_images.py <image_folder> <output_json>")
        print("Example: python extract_gps_from_images.py ./images gps_coordinates.json")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    extract_gps_from_folder(folder_path, output_file)

if __name__ == "__main__":
    main()
