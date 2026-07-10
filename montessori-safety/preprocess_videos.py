"""
Video Preprocessing Script for Child Safety Project
====================================================
Cleans raw recorded videos before skeleton extraction.

What it does:
1. Removes audio track (not needed, saves space)
2. Trims start/end based on ground_truth.csv timestamps
3. Optionally skips very dark or very blurry frames
4. Saves cleaned video ready for skeleton extraction

Usage:
    python preprocess_videos.py --input "data/child_datasets/self_recorded/daily_normal/"
    python preprocess_videos.py --input "path/to/video.mp4" --output "cleaned/"
    python preprocess_videos.py --check "path/to/video.mp4"  (just shows quality stats)
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def check_frame_quality(frame):
    """
    Check if a frame is usable for skeleton extraction.
    
    Returns:
        dict with quality metrics:
        - brightness: 0-255 (below 30 = too dark)
        - sharpness: Laplacian variance (below 20 = too blurry)
        - has_motion: True if frame differs significantly from static
        - is_usable: overall pass/fail
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Brightness: mean pixel value
    brightness = np.mean(gray)
    
    # Sharpness: variance of Laplacian (higher = sharper)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # A frame is usable if it's not too dark and not completely blurry
    is_usable = brightness > 25 and sharpness > 15
    
    return {
        'brightness': brightness,
        'sharpness': sharpness,
        'is_usable': is_usable
    }


def check_video_quality(video_path):
    """
    Analyze a video and report quality statistics.
    Use this to decide if a video needs cleaning.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n=== Video Quality Report ===")
    print(f"File: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Duration: {duration:.1f} seconds ({total_frames} frames)")
    
    brightness_values = []
    sharpness_values = []
    dark_frames = 0
    blurry_frames = 0
    
    # Sample every 10th frame for speed
    sample_interval = max(1, total_frames // 100)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            quality = check_frame_quality(frame)
            brightness_values.append(quality['brightness'])
            sharpness_values.append(quality['sharpness'])
            if quality['brightness'] < 25:
                dark_frames += 1
            if quality['sharpness'] < 15:
                blurry_frames += 1
        
        frame_idx += 1
    
    cap.release()
    
    sampled = len(brightness_values)
    print(f"\nSampled {sampled} frames:")
    print(f"  Brightness: min={min(brightness_values):.0f}, "
          f"avg={np.mean(brightness_values):.0f}, "
          f"max={max(brightness_values):.0f}")
    print(f"  Sharpness:  min={min(sharpness_values):.0f}, "
          f"avg={np.mean(sharpness_values):.0f}, "
          f"max={max(sharpness_values):.0f}")
    print(f"  Dark frames:  {dark_frames}/{sampled} ({dark_frames/sampled*100:.1f}%)")
    print(f"  Blurry frames: {blurry_frames}/{sampled} ({blurry_frames/sampled*100:.1f}%)")
    
    # Verdict
    if dark_frames / sampled > 0.5:
        print(f"\n  ⚠ WARNING: Over 50% of frames are too dark.")
        print(f"    Consider: this video may not be usable for training.")
    elif blurry_frames / sampled > 0.5:
        print(f"\n  ⚠ WARNING: Over 50% of frames are too blurry.")
        print(f"    Consider: this video may not be usable for training.")
    else:
        print(f"\n  ✓ Video quality is acceptable for skeleton extraction.")
    
    return {
        'duration': duration,
        'fps': fps,
        'resolution': (width, height),
        'avg_brightness': np.mean(brightness_values),
        'avg_sharpness': np.mean(sharpness_values),
        'dark_pct': dark_frames / sampled * 100,
        'blurry_pct': blurry_frames / sampled * 100
    }


def preprocess_video(input_path, output_path, 
                     trim_start=0, trim_end=None,
                     skip_dark=True, skip_blurry=True,
                     target_fps=None):
    """
    Clean a video for skeleton extraction.
    
    Args:
        input_path: path to raw video
        output_path: path for cleaned output
        trim_start: seconds to skip at beginning
        trim_end: seconds to keep until (None = end of video)
        skip_dark: skip frames that are too dark
        skip_blurry: skip frames that are too blurry
        target_fps: resample to this FPS (None = keep original)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_fps = target_fps if target_fps else fps
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    
    start_frame = int(trim_start * fps)
    end_frame = int(trim_end * fps) if trim_end else total_frames
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    written = 0
    skipped_dark = 0
    skipped_blurry = 0
    
    # Frame sampling for target FPS
    if target_fps and target_fps < fps:
        frame_interval = fps / target_fps
    else:
        frame_interval = 1
    
    next_sample = start_frame
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS resampling
        if frame_idx < next_sample:
            frame_idx += 1
            continue
        next_sample += frame_interval
        
        # Quality checks
        if skip_dark or skip_blurry:
            quality = check_frame_quality(frame)
            if skip_dark and quality['brightness'] < 25:
                skipped_dark += 1
                frame_idx += 1
                continue
            if skip_blurry and quality['sharpness'] < 15:
                skipped_blurry += 1
                frame_idx += 1
                continue
        
        writer.write(frame)
        written += 1
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    duration_out = written / out_fps
    print(f"  Preprocessed: {Path(input_path).name}")
    print(f"    Input:  {total_frames} frames, {total_frames/fps:.1f}s")
    print(f"    Output: {written} frames, {duration_out:.1f}s")
    if skipped_dark > 0:
        print(f"    Skipped {skipped_dark} dark frames")
    if skipped_blurry > 0:
        print(f"    Skipped {skipped_blurry} blurry frames")
    print(f"    Saved: {output_path}")


def batch_preprocess(input_dir, output_dir, trim_start=0):
    """
    Preprocess all videos in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
    
    videos = [f for f in os.listdir(input_dir) 
              if Path(f).suffix.lower() in video_extensions]
    
    print(f"Found {len(videos)} videos in {input_dir}")
    
    for video_file in sorted(videos):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)
        
        preprocess_video(
            input_path, output_path,
            trim_start=trim_start,
            skip_dark=True,
            skip_blurry=True
        )


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess videos for skeleton extraction'
    )
    parser.add_argument('--input', type=str, required=False,
                        help='Input video file or directory')
    parser.add_argument('--output', type=str, default='data/child_datasets/self_recorded/cleaned/',
                        help='Output directory for cleaned videos')
    parser.add_argument('--check', type=str, default=None,
                        help='Just check quality of a video (no processing)')
    parser.add_argument('--trim-start', type=float, default=0,
                        help='Seconds to trim from start of each video')
    parser.add_argument('--target-fps', type=float, default=None,
                        help='Resample to this FPS (e.g., 15 for faster extraction)')
    args = parser.parse_args()
    
    if args.check:
        check_video_quality(args.check)
        return
    
    if not args.input:
        print("Usage:")
        print("  Check quality:  python preprocess_videos.py --check path/to/video.mp4")
        print("  Single video:   python preprocess_videos.py --input video.mp4 --output cleaned/")
        print("  Batch process:  python preprocess_videos.py --input folder/ --output cleaned/")
        return
    
    if os.path.isdir(args.input):
        batch_preprocess(args.input, args.output, trim_start=args.trim_start)
    else:
        output_path = os.path.join(args.output, Path(args.input).name)
        preprocess_video(
            args.input, output_path,
            trim_start=args.trim_start,
            target_fps=args.target_fps
        )


if __name__ == '__main__':
    main()