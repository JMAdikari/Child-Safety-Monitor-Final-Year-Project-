"""
Converts folders of PNG frames into .mp4 video files.
Scans all subfolders, finds sequences of PNGs, and creates one video per folder.
cmmand, venv\Scripts\activate,python convert_png_to_video.py

"""

import cv2
import os
import sys
from pathlib import Path

def convert_folder_to_video(folder_path, output_path, fps=30):
    """Convert a folder of PNG images to a single .mp4 video."""
    
    # Find all PNG files and sort them
    png_files = sorted([
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if len(png_files) < 5:
        return False  # Skip folders with too few images
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(folder_path, png_files[0]))
    if first_image is None:
        return False
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames
    count = 0
    for png_file in png_files:
        frame = cv2.imread(os.path.join(folder_path, png_file))
        if frame is not None:
            # Resize if dimensions don't match (some frames might differ)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            count += 1
    
    writer.release()
    return count > 0


def main():
    # === CHANGE THIS PATH TO YOUR UP-FALL FOLDER ===
    input_dir = r"data\raw\fight_datasets\hmdb51_fights\punch"
    output_dir = r"data\raw\fall_datasets\upfall_videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("PNG TO VIDEO CONVERTER")
    print("=" * 50)
    print(f"Scanning: {input_dir}")
    print(f"Output to: {output_dir}")
    print()
    
    converted = 0
    skipped = 0
    
    # Walk through all subfolders
    for root, dirs, files in os.walk(input_dir):
        # Check if this folder contains PNG files
        png_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(png_files) >= 5:  # Only process folders with enough frames
            # Create a meaningful output filename from the folder path
            relative_path = os.path.relpath(root, input_dir)
            # Replace path separators with underscores for filename
            video_name = relative_path.replace(os.sep, '_').replace(' ', '_')
            output_path = os.path.join(output_dir, f"{video_name}.mp4")
            
            # Skip if already converted
            if os.path.exists(output_path):
                print(f"  SKIP (exists): {video_name}")
                skipped += 1
                continue
            
            print(f"  Converting: {relative_path} ({len(png_files)} frames) ...", end=" ")
            
            success = convert_folder_to_video(root, output_path, fps=30)
            
            if success:
                # Get file size
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"OK ({size_mb:.1f} MB)")
                converted += 1
            else:
                print("FAILED")
                # Remove failed file
                if os.path.exists(output_path):
                    os.remove(output_path)
    
    print()
    print("=" * 50)
    print(f"Done! Converted: {converted} | Skipped: {skipped}")
    print(f"Videos saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()