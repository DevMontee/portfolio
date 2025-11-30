"""
KITTI DATA LOADING FOR PARAMETER TUNING

Uses your detection_pipeline.py and KITTIDataLoader to load real data.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List

# Add paths to import your modules
kitti_root = r"D:\KITTI Dataset"
detection_pipeline_dir = r"D:\kitti_sensor_fusion\Detection Pipeline"
env_setup_dir = r"D:\kitti_sensor_fusion\Environment Setup & Data Pipeline"

sys.path.insert(0, detection_pipeline_dir)
sys.path.insert(0, env_setup_dir)

try:
    from detection_pipeline import DetectionPipeline
    from kitti_dataloader import KITTIDataLoader
    print("✓ Imported detection_pipeline and KITTIDataLoader")
except ImportError as e:
    print(f"ERROR importing: {e}")
    sys.exit(1)


def load_kitti_detections(sequence_id: str, 
                         kitti_root: str = r"D:\KITTI Dataset") -> Dict:
    """
    Load detections for a sequence using your detection pipeline.
    
    This uses DBSCAN clustering on LiDAR + camera detections.
    
    Args:
        sequence_id: Sequence ID (e.g., "0000")
        kitti_root: Root path to KITTI dataset
    
    Returns:
        {
            frame_id: [
                {
                    'x': float, 'y': float, 'z': float,
                    'size': [l, w, h],
                    'conf': float,
                    'type': str
                },
                ...
            ],
            ...
        }
    """
    print(f"\n  Loading detections for sequence {sequence_id}...")
    
    try:
        pipeline = DetectionPipeline(kitti_root, sequence=sequence_id, split="training")
    except Exception as e:
        print(f"    ERROR loading pipeline: {e}")
        return {}
    
    detections = {}
    
    # Determine number of frames in sequence
    # (typically 0-153 frames per sequence)
    try:
        loader = KITTIDataLoader(kitti_root, sequence=sequence_id, split="training")
        num_frames = len(loader)
    except:
        num_frames = 150  # Default fallback
    
    print(f"    Processing {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        try:
            # Get detection results from pipeline
            results = pipeline.process_frame(frame_idx)
            
            # Convert to tracking format
            frame_detections = []
            
            # Get associated detections
            for cluster_id, det_idx in results['associations'].items():
                if det_idx >= 0:
                    # Valid association
                    det = results['camera_detections'][det_idx]
                    bound = results['bounds'][cluster_id]
                    
                    frame_detections.append({
                        'x': det['location_3d'][0],
                        'y': det['location_3d'][1],
                        'z': det['location_3d'][2],
                        'l': det['dimensions'][0],  # length
                        'w': det['dimensions'][1],  # width
                        'h': det['dimensions'][2],  # height
                        'rot': det['rotation_y'],
                        'type': det['type'],
                        'conf': det.get('score', 0.9),
                    })
            
            detections[frame_idx] = frame_detections
            
            if (frame_idx + 1) % 50 == 0:
                print(f"      Processed {frame_idx + 1}/{num_frames} frames")
        
        except Exception as e:
            print(f"    ERROR processing frame {frame_idx}: {e}")
            detections[frame_idx] = []
    
    print(f"  ✓ Loaded {len(detections)} frames with detections")
    return detections


def load_kitti_ground_truth(sequence_id: str, 
                           kitti_root: str = r"D:\KITTI Dataset") -> Dict:
    """
    Load ground truth labels for a sequence.
    
    Uses KITTIDataLoader to read official KITTI labels.
    
    Args:
        sequence_id: Sequence ID (e.g., "0000")
        kitti_root: Root path to KITTI dataset
    
    Returns:
        {
            frame_id: [
                {
                    'id': int,          # Track ID
                    'type': str,        # Object type
                    'x': float,         # 3D location
                    'y': float,
                    'z': float,
                    'l': float,         # Dimensions
                    'w': float,
                    'h': float,
                    'rot': float,       # Rotation angle
                },
                ...
            ],
            ...
        }
    """
    print(f"\n  Loading ground truth for sequence {sequence_id}...")
    
    try:
        loader = KITTIDataLoader(kitti_root, sequence=sequence_id, split="training")
        num_frames = len(loader)
    except Exception as e:
        print(f"    ERROR loading data: {e}")
        return {}
    
    ground_truth = {}
    
    print(f"    Processing {num_frames} frames...")
    
    for frame_idx in range(num_frames):
        try:
            frame = loader[frame_idx]
            labels = frame['labels']
            
            frame_gt = []
            
            for label_idx, label in enumerate(labels):
                # Skip DontCare objects
                if label['type'] == 'DontCare':
                    continue
                
                # Skip if bounding box is invalid
                if label['bbox_2d'] is None or label['location_3d'] is None:
                    continue
                
                frame_gt.append({
                    'id': label_idx,  # Use label index as object ID
                    'type': label['type'],
                    'x': label['location_3d'][0],
                    'y': label['location_3d'][1],
                    'z': label['location_3d'][2],
                    'l': label['dimensions'][0],
                    'w': label['dimensions'][1],
                    'h': label['dimensions'][2],
                    'rot': label['rotation_y'],
                })
            
            ground_truth[frame_idx] = frame_gt
            
            if (frame_idx + 1) % 50 == 0:
                print(f"      Processed {frame_idx + 1}/{num_frames} frames")
        
        except Exception as e:
            print(f"    ERROR processing frame {frame_idx}: {e}")
            ground_truth[frame_idx] = []
    
    print(f"  ✓ Loaded {len(ground_truth)} frames with ground truth")
    return ground_truth


# Test the functions
if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING KITTI DATA LOADING")
    print("="*70)
    
    sequence = "0000"
    
    # Load detections
    print("\n1. Loading detections...")
    detections = load_kitti_detections(sequence)
    print(f"   Total frames: {len(detections)}")
    if detections:
        first_frame = list(detections.keys())[0]
        print(f"   Example (frame {first_frame}): {len(detections[first_frame])} detections")
        if detections[first_frame]:
            print(f"   First detection: {detections[first_frame][0]}")
    
    # Load ground truth
    print("\n2. Loading ground truth...")
    ground_truth = load_kitti_ground_truth(sequence)
    print(f"   Total frames: {len(ground_truth)}")
    if ground_truth:
        first_frame = list(ground_truth.keys())[0]
        print(f"   Example (frame {first_frame}): {len(ground_truth[first_frame])} objects")
        if ground_truth[first_frame]:
            print(f"   First object: {ground_truth[first_frame][0]}")
    
    print("\n✓ Data loading test complete!")
