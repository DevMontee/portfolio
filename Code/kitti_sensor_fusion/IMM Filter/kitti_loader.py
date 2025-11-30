"""
KITTI Data Loader
Loads 3D detections from KITTI ground truth labels
Bridges between KITTI dataset and IMM filter
"""

import os
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class KITTILoader:
    """Load 3D detections from KITTI dataset"""
    
    def __init__(self, kitti_root: str):
        """
        Initialize KITTI loader
        
        Args:
            kitti_root: Path to KITTI dataset root
                       (D:\KITTI Dataset)
        """
        self.kitti_root = Path(kitti_root)
        self.label_dir = self.kitti_root / 'data_tracking_label_2' / 'training' / 'label_02'
        self.image_dir = self.kitti_root / 'data_tracking_image_2' / 'training' / 'image_02'
        self.velodyne_dir = self.kitti_root / 'data_tracking_velodyne' / 'training' / 'velodyne'
        
        print(f"KITTI Root: {self.kitti_root}")
        print(f"Labels Dir: {self.label_dir}")
        print(f"Labels Dir Exists: {self.label_dir.exists()}")
    
    def get_available_sequences(self) -> List[str]:
        """Get list of available sequences"""
        if not self.label_dir.exists():
            print(f"ERROR: Label directory not found: {self.label_dir}")
            return []
        
        sequences = []
        for f in sorted(self.label_dir.glob('*.txt')):
            seq_name = f.stem
            sequences.append(seq_name)
        
        return sequences
    
    def load_sequence_labels(self, sequence: str) -> Dict[int, List[Dict]]:
        """
        Load all labels for a sequence, organized by frame
        
        Args:
            sequence: Sequence ID (e.g., '0000')
        
        Returns:
            Dict mapping frame_id to list of detections
        """
        label_file = self.label_dir / f'{sequence}.txt'
        
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        frames_data = {}
        
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    parts = line.strip().split()
                    
                    # Parse KITTI format
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    obj_type = parts[2]
                    truncated = int(parts[3])
                    occluded = int(parts[4])
                    alpha = float(parts[5])
                    
                    # 2D bounding box (not used by IMM)
                    bbox_left = float(parts[6])
                    bbox_top = float(parts[7])
                    bbox_right = float(parts[8])
                    bbox_bottom = float(parts[9])
                    
                    # 3D dimensions: height, width, length
                    h = float(parts[10])
                    w = float(parts[11])
                    l = float(parts[12])
                    
                    # 3D location: x, y, z
                    x = float(parts[13])
                    y = float(parts[14])
                    z = float(parts[15])
                    
                    # Rotation
                    rotation_y = float(parts[16])
                    
                    # Score (usually 1.0 for ground truth)
                    score = float(parts[17]) if len(parts) > 17 else 1.0
                    
                    # Create detection dict
                    detection = {
                        'track_id': track_id,
                        'type': obj_type,
                        'truncated': truncated,
                        'occluded': occluded,
                        'alpha': alpha,
                        'bbox_2d': (bbox_left, bbox_top, bbox_right, bbox_bottom),
                        'x': x,
                        'y': y,
                        'z': z,
                        'l': l,
                        'w': w,
                        'h': h,
                        'rotation_y': rotation_y,
                        'score': score,
                        'confidence': score  # For IMM compatibility
                    }
                    
                    # Add to frame
                    if frame_id not in frames_data:
                        frames_data[frame_id] = []
                    
                    frames_data[frame_id].append(detection)
                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue
        
        return frames_data
    
    def get_frame_detections(self, sequence: str, frame_id: int) -> List[Dict]:
        """
        Get detections for a specific frame
        
        Args:
            sequence: Sequence ID
            frame_id: Frame number
        
        Returns:
            List of detection dictionaries
        """
        frames_data = self.load_sequence_labels(sequence)
        
        if frame_id not in frames_data:
            return []
        
        return frames_data[frame_id]
    
    def load_sequence_range(self, sequence: str, 
                           start_frame: int = 0, 
                           end_frame: int = None) -> Dict[int, List[Dict]]:
        """
        Load detections for a frame range
        
        Args:
            sequence: Sequence ID
            start_frame: First frame to load
            end_frame: Last frame to load (None = all)
        
        Returns:
            Dict mapping frame_id to detections
        """
        frames_data = self.load_sequence_labels(sequence)
        
        # Filter by frame range
        filtered_data = {}
        for frame_id in sorted(frames_data.keys()):
            if frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                break
            
            filtered_data[frame_id] = frames_data[frame_id]
        
        return filtered_data
    
    def filter_by_type(self, detections: List[Dict], obj_type: str = 'Car') -> List[Dict]:
        """
        Filter detections by object type
        
        Args:
            detections: List of detections
            obj_type: Object type to keep (e.g., 'Car', 'Pedestrian', 'Cyclist')
        
        Returns:
            Filtered detections
        """
        return [det for det in detections if det['type'] == obj_type]
    
    def filter_by_occlusion(self, detections: List[Dict], max_occlusion: int = 2) -> List[Dict]:
        """
        Filter out heavily occluded objects
        
        Args:
            detections: List of detections
            max_occlusion: Maximum occlusion level (0-3)
        
        Returns:
            Filtered detections
        """
        return [det for det in detections if det['occluded'] <= max_occlusion]
    
    def filter_by_truncation(self, detections: List[Dict], max_truncation: float = 0.5) -> List[Dict]:
        """
        Filter out heavily truncated objects
        
        Args:
            detections: List of detections
            max_truncation: Maximum truncation ratio (0-1)
        
        Returns:
            Filtered detections
        """
        return [det for det in detections if det['truncated'] <= max_truncation]
    
    def print_sequence_stats(self, sequence: str):
        """Print statistics about a sequence"""
        frames_data = self.load_sequence_labels(sequence)
        
        print(f"\n{'='*70}")
        print(f"Sequence {sequence} Statistics")
        print(f"{'='*70}")
        print(f"Total frames: {len(frames_data)}")
        
        # Count objects
        total_objects = sum(len(dets) for dets in frames_data.values())
        print(f"Total objects: {total_objects}")
        print(f"Avg objects/frame: {total_objects / len(frames_data):.1f}")
        
        # Object types
        all_detections = []
        for dets in frames_data.values():
            all_detections.extend(dets)
        
        types = {}
        for det in all_detections:
            t = det['type']
            types[t] = types.get(t, 0) + 1
        
        print(f"\nObject types:")
        for obj_type, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {obj_type}: {count}")
        
        # Frame range
        frame_ids = sorted(frames_data.keys())
        print(f"\nFrame range: {frame_ids[0]} - {frame_ids[-1]}")
        
        print(f"{'='*70}\n")


def main():
    """Test KITTI loader"""
    
    kitti_root = r'D:\KITTI Dataset'
    
    print("\n" + "="*70)
    print("KITTI DATA LOADER TEST")
    print("="*70 + "\n")
    
    # Initialize loader
    loader = KITTILoader(kitti_root)
    
    # Get available sequences
    sequences = loader.get_available_sequences()
    print(f"Available sequences: {len(sequences)}")
    print(f"Sequences: {sequences[:5]}...\n")
    
    if not sequences:
        print("ERROR: No sequences found!")
        return
    
    # Test with first sequence
    sequence = sequences[0]
    print(f"Testing with sequence: {sequence}")
    
    # Print stats
    loader.print_sequence_stats(sequence)
    
    # Load detections
    frames_data = loader.load_sequence_range(sequence, start_frame=0, end_frame=10)
    
    print(f"Loaded {len(frames_data)} frames (0-10)")
    print("\nFrame-by-frame breakdown:")
    
    for frame_id in sorted(frames_data.keys()):
        detections = frames_data[frame_id]
        
        # Filter
        cars = loader.filter_by_type(detections, 'Car')
        cars = loader.filter_by_occlusion(cars, max_occlusion=2)
        
        print(f"  Frame {frame_id:3d}: {len(detections):2d} objects, "
              f"{len(cars):2d} cars (not heavily occluded)")
        
        if frame_id == 0 and cars:
            print(f"\n    Example detection (Frame 0, Object 0):")
            det = cars[0]
            print(f"      Position: ({det['x']:.2f}, {det['y']:.2f}, {det['z']:.2f})")
            print(f"      Dimensions: l={det['l']:.2f}, w={det['w']:.2f}, h={det['h']:.2f}")
            print(f"      Type: {det['type']}")
            print(f"      Truncated: {det['truncated']}, Occluded: {det['occluded']}\n")
    
    print(f"{'='*70}")
    print("✓ KITTI Loader Test Complete")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
