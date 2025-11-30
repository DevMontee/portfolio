"""
IMM Filter with KITTI Dataset Integration
Complete bridge between KITTI data and IMM tracking
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from kitti_loader import KITTILoader
from imm_filter import IMMTracker, IMMConfig


class IMMKITTITracker:
    """IMM Tracker integrated with KITTI dataset"""
    
    def __init__(self, kitti_root: str, config: IMMConfig = None):
        """
        Initialize IMM tracker with KITTI data source
        
        Args:
            kitti_root: Path to KITTI dataset
            config: IMM configuration
        """
        self.loader = KITTILoader(kitti_root)
        self.config = config or IMMConfig()
        self.tracker = IMMTracker(self.config)
        
        self.results = {
            'sequences': {},
            'total_frames': 0,
            'total_tracks': 0,
            'total_detections': 0,
        }
    
    def process_sequence(self, sequence: str, 
                        start_frame: int = 0, 
                        end_frame: int = None,
                        obj_type: str = 'Car',
                        max_occlusion: int = 2) -> Dict:
        """
        Process a full KITTI sequence with IMM
        
        Args:
            sequence: Sequence ID
            start_frame: First frame
            end_frame: Last frame (None = all)
            obj_type: Object type to track ('Car', 'Pedestrian', etc.)
            max_occlusion: Max occlusion level to include
        
        Returns:
            Tracking results
        """
        print(f"\n{'='*70}")
        print(f"Processing Sequence {sequence}")
        print(f"{'='*70}\n")
        
        # Load KITTI data ← BRIDGE CODE STARTS HERE!
        frames_data = self.loader.load_sequence_range(
            sequence, 
            start_frame=start_frame, 
            end_frame=end_frame
        )
        
        if not frames_data:
            print(f"ERROR: No data loaded for sequence {sequence}")
            return {}
        
        print(f"Loaded {len(frames_data)} frames from KITTI")
        
        # Reset tracker for new sequence
        config = IMMConfig(dt=0.1)  # KITTI is 10Hz
        self.tracker = IMMTracker(config)
        
        # Process each frame
        sequence_results = {
            'sequence': sequence,
            'frames': {},
            'total_detections': 0,
            'total_tracks': 0,
            'frame_count': len(frames_data)
        }
        
        for frame_idx, frame_id in enumerate(sorted(frames_data.keys())):
            detections = frames_data[frame_id]
            
            # Filter detections (KITTI -> IMM compatible format)
            detections = self.loader.filter_by_type(detections, obj_type)
            detections = self.loader.filter_by_occlusion(detections, max_occlusion)
            detections = self.loader.filter_by_truncation(detections, 0.5)
            
            if not detections:
                continue
            
            # PASS TO IMM ← CRITICAL BRIDGE POINT!
            tracked_objects = self.tracker.update(detections)
            
            # Store results
            sequence_results['frames'][frame_id] = {
                'detections_count': len(detections),
                'tracks_count': len(tracked_objects),
                'tracks': tracked_objects
            }
            
            sequence_results['total_detections'] += len(detections)
            sequence_results['total_tracks'] += len(tracked_objects)
            
            # Progress
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_id:3d}: {len(detections):2d} detections, "
                      f"{len(tracked_objects):2d} tracks", end='')
                
                if tracked_objects:
                    # Show sample track
                    track = tracked_objects[0]
                    print(f" | Sample: pos=({track['x']:.1f}, {track['y']:.1f}, {track['z']:.1f}), "
                          f"CV prob={track['model_probs']['constant_velocity']:.2f}")
                else:
                    print()
        
        self.results['sequences'][sequence] = sequence_results
        self.results['total_frames'] += len(frames_data)
        self.results['total_detections'] += sequence_results['total_detections']
        self.results['total_tracks'] += sequence_results['total_tracks']
        
        return sequence_results
    
    def process_multiple_sequences(self, sequences: List[str], 
                                  start_frame: int = 0,
                                  end_frame: int = 20) -> Dict:
        """
        Process multiple sequences
        
        Args:
            sequences: List of sequence IDs
            start_frame: Starting frame for each sequence
            end_frame: Ending frame for each sequence
        
        Returns:
            Combined results
        """
        for sequence in sequences:
            self.process_sequence(sequence, 
                                start_frame=start_frame, 
                                end_frame=end_frame)
        
        return self.results
    
    def print_results_summary(self):
        """Print summary of results"""
        print(f"\n{'='*70}")
        print("TRACKING RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        for seq_id, seq_data in self.results['sequences'].items():
            print(f"Sequence {seq_id}:")
            print(f"  Frames: {seq_data['frame_count']}")
            print(f"  Detections: {seq_data['total_detections']}")
            print(f"  Tracks: {seq_data['total_tracks']}")
            print(f"  Avg detections/frame: {seq_data['total_detections'] / seq_data['frame_count']:.1f}")
            
            # Show sample frame
            sample_frame = None
            for frame_id, frame_data in seq_data['frames'].items():
                if frame_data['tracks_count'] > 0:
                    sample_frame = frame_id
                    break
            
            if sample_frame is not None:
                frame_data = seq_data['frames'][sample_frame]
                print(f"\n  Sample Frame {sample_frame}:")
                print(f"    Detections: {frame_data['detections_count']}")
                print(f"    Tracks: {frame_data['tracks_count']}")
                
                if frame_data['tracks']:
                    track = frame_data['tracks'][0]
                    print(f"\n    Sample Track {track['id']}:")
                    print(f"      Position: ({track['x']:.2f}, {track['y']:.2f}, {track['z']:.2f})")
                    print(f"      Velocity: ({track['vx']:.3f}, {track['vy']:.3f}, {track['vz']:.3f})")
                    print(f"      Yaw Rate: {track['yaw_rate']:.3f} rad/s")
                    print(f"      Models: CV={track['model_probs']['constant_velocity']:.2f}, "
                          f"Turn={track['model_probs']['turning']:.2f}")
            print()
        
        print(f"{'='*70}\n")
    
    def save_results(self, output_file: str):
        """Save results to JSON"""
        # Prepare for JSON serialization
        results_copy = dict(self.results)
        
        # Convert tracks to simpler format for JSON
        for seq_id, seq_data in results_copy['sequences'].items():
            for frame_id, frame_data in seq_data['frames'].items():
                # Simplify track data for storage
                simple_tracks = []
                for track in frame_data['tracks']:
                    simple_tracks.append({
                        'id': track['id'],
                        'x': float(track['x']),
                        'y': float(track['y']),
                        'z': float(track['z']),
                        'vx': float(track['vx']),
                        'vy': float(track['vy']),
                        'vz': float(track['vz']),
                        'yaw_rate': float(track['yaw_rate']),
                        'model_probs': track['model_probs'],
                        'confidence': float(track['confidence'])
                    })
                frame_data['tracks'] = simple_tracks
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")


def main():
    """Main demonstration"""
    
    print("\n" + "="*70)
    print("IMM FILTER - KITTI INTEGRATION DEMO")
    print("="*70 + "\n")
    
    kitti_root = r'D:\KITTI Dataset'
    
    # Initialize IMM+KITTI system
    config = IMMConfig(
        dt=0.1,              # 10Hz KITTI frames
        q_pos_cv=0.1,        # From your tuning
        q_vel_cv=0.01,
        q_pos_turn=0.2,
        q_yaw_rate=0.1,
        r_camera=0.5,        # From your tuning results
        r_lidar=0.5,
        p_cv_to_cv=0.95,
        p_cv_to_turn=0.05,
        p_turn_to_turn=0.90,
        p_turn_to_cv=0.10,
    )
    
    tracker = IMMKITTITracker(kitti_root, config)
    
    # Get available sequences
    sequences = tracker.loader.get_available_sequences()
    print(f"Available sequences: {sequences[:5]}...")
    
    if not sequences:
        print("ERROR: No KITTI sequences found!")
        return
    
    # Process sequences
    print(f"\nProcessing KITTI sequences with IMM filter...")
    sequences_to_process = sequences[:3]  # Process first 3 sequences
    
    tracker.process_multiple_sequences(
        sequences_to_process,
        start_frame=0,
        end_frame=20  # First 20 frames per sequence
    )
    
    # Print summary
    tracker.print_results_summary()
    
    # Save results
    output_file = Path(kitti_root).parent / 'imm_kitti_results.json'
    tracker.save_results(str(output_file))
    
    print(f"{'='*70}")
    print("✓ IMM+KITTI Integration Complete!")
    print(f"{'='*70}\n")
    
    print("What happened:")
    print("  1. ✓ Loaded KITTI ground truth labels")
    print("  2. ✓ Extracted 3D detections [x, y, z, l, w, h]")
    print("  3. ✓ Passed to IMM filter")
    print("  4. ✓ IMM estimated velocities and selected motion models")
    print("  5. ✓ Results saved to JSON\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
