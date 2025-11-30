"""
REAL PARAMETER EVALUATION - WITH ACTUAL KITTI DATA

Replaces the mock evaluate_params() with real KITTI evaluation.
Uses your detection_pipeline and KITTIDataLoader.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Import your data loading functions
sys.path.insert(0, r"D:\kitti_sensor_fusion\Parameter Tuning")

try:
    from load_kitti_real_data import load_kitti_detections, load_kitti_ground_truth
    from kalman_tracker import KalmanTracker
    print("✓ Imported data loaders and KalmanTracker\n")
except ImportError as e:
    print(f"ERROR: {e}")
    print("Make sure load_kitti_real_data.py is in Parameter Tuning folder")
    sys.exit(1)


def compute_mot_metrics(tracked_objects: Dict, ground_truth: Dict) -> Tuple[float, float, int]:
    """
    Compute MOT metrics: MOTA, MOTP, ID Switches
    
    MOTA = 1 - (FN + FP + ID_SW) / GT_total
    MOTP = Average distance for matched pairs
    
    Args:
        tracked_objects: {frame_id: [tracked_obj1, tracked_obj2, ...]}
        ground_truth: {frame_id: [gt_obj1, gt_obj2, ...]}
    
    Returns:
        (mota, motp, id_switches)
    """
    
    total_gt = 0
    total_missed = 0
    total_fp = 0
    total_distance = 0.0
    total_matches = 0
    id_switches = 0
    
    previous_matches = {}  # {gt_id: tracked_id}
    
    for frame_id in sorted(ground_truth.keys()):
        if frame_id not in tracked_objects:
            # No tracking output for this frame
            total_missed += len(ground_truth[frame_id])
            total_gt += len(ground_truth[frame_id])
            continue
        
        gt_frame = ground_truth[frame_id]
        tracked_frame = tracked_objects[frame_id]
        
        total_gt += len(gt_frame)
        
        # Match tracked objects to ground truth (greedy nearest neighbor)
        matched_gt = set()
        matched_tracks = set()
        current_matches = {}
        
        for gt_idx, gt_obj in enumerate(gt_frame):
            if gt_idx in matched_gt:
                continue
            
            gt_pos = np.array([gt_obj['x'], gt_obj['y'], gt_obj['z']])
            best_dist = np.inf
            best_track_idx = -1
            
            for tr_idx, tr_obj in enumerate(tracked_frame):
                if tr_idx in matched_tracks:
                    continue
                
                tr_pos = np.array(tr_obj['state']['position'])
                dist = np.linalg.norm(gt_pos - tr_pos)
                
                if dist < best_dist:
                    best_dist = dist
                    best_track_idx = tr_idx
            
            # Match if within threshold (2.0 meters)
            if best_dist < 2.0:
                matched_gt.add(gt_idx)
                matched_tracks.add(best_track_idx)
                total_matches += 1
                total_distance += best_dist
                current_matches[gt_obj['id']] = tracked_frame[best_track_idx]['id']
                
                # Check for ID switch
                if gt_obj['id'] in previous_matches:
                    if previous_matches[gt_obj['id']] != tracked_frame[best_track_idx]['id']:
                        id_switches += 1
        
        # Count missed detections (FN) and false positives (FP)
        total_missed += len(gt_frame) - len(matched_gt)
        total_fp += len(tracked_frame) - len(matched_tracks)
        
        previous_matches = current_matches
    
    # Compute MOTA
    if total_gt > 0:
        mota = 1.0 - (total_missed + total_fp + id_switches) / total_gt
    else:
        mota = 0.0
    
    # Compute MOTP (precision of matched pairs)
    if total_matches > 0:
        motp = total_distance / total_matches
    else:
        motp = 0.0
    
    return max(0.0, mota), motp, id_switches  # Clamp MOTA to [0, 1]


def evaluate_params_REAL(params: Dict, 
                         sequence_ids: List[str],
                         kitti_root: str = r"D:\KITTI Dataset") -> float:
    """
    REAL evaluation using KalmanTracker on actual KITTI data.
    
    Args:
        params: Parameter dict with all tuning params
        sequence_ids: List of sequence IDs to evaluate on
        kitti_root: Path to KITTI dataset root
    
    Returns:
        MOTA score (0-1, higher is better)
    """
    
    print(f"\n  Testing: q_pos={params['q_pos']}, q_vel={params['q_vel']}, "
          f"gate={params['gate_threshold']}, init={params['init_frames']}")
    
    # Create tracker
    try:
        tracker = KalmanTracker(
            q_pos=params['q_pos'],
            q_vel=params['q_vel'],
            r_camera=params['r_camera'],
            r_lidar=params['r_lidar'],
            gate_threshold=params['gate_threshold'],
            init_frames=params['init_frames'],
            max_age=params['max_age'],
            age_threshold=params['age_threshold']
        )
    except Exception as e:
        print(f"    ERROR creating tracker: {e}")
        return 0.0
    
    all_tracked = {}
    all_ground_truth = {}
    
    # Process each sequence
    for seq_id in sequence_ids:
        print(f"    Sequence {seq_id}...", end=" ", flush=True)
        
        try:
            # Load real detections and ground truth
            seq_detections = load_kitti_detections(seq_id, kitti_root)
            seq_ground_truth = load_kitti_ground_truth(seq_id, kitti_root)
            
            if not seq_detections or not seq_ground_truth:
                print("SKIPPED (no data)")
                continue
            
            # Reset tracker for new sequence
            tracker.tracks = {}
            tracker.next_id = 0
            tracker.frame_count = 0
            
            # Run tracking on all frames
            for frame_id in sorted(seq_detections.keys()):
                frame_dets = seq_detections[frame_id]
                
                # Convert detection format for tracker
                tracker_dets = []
                for det in frame_dets:
                    tracker_dets.append({
                        'x': det['x'],
                        'y': det['y'],
                        'z': det['z'],
                        'size': [det['l'], det['w'], det['h']],
                        'conf': det['conf'],
                    })
                
                # Update tracker
                tracked = tracker.update(frame_id, tracker_dets)
                all_tracked[frame_id] = tracked
            
            # Accumulate ground truth
            all_ground_truth.update(seq_ground_truth)
            
            print("OK")
        
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Compute MOT metrics
    if not all_tracked or not all_ground_truth:
        print(f"    No valid tracking data")
        return 0.0
    
    mota, motp, id_switches = compute_mot_metrics(all_tracked, all_ground_truth)
    
    print(f"    ✓ MOTA: {mota:.4f}, MOTP: {motp:.4f}, ID Switches: {id_switches}")
    
    return mota


# Test function
if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING REAL PARAMETER EVALUATION")
    print("="*70)
    
    # Test parameters
    test_params = {
        'q_pos': 0.1,
        'q_vel': 0.01,
        'r_camera': 0.2,
        'r_lidar': 0.2,
        'gate_threshold': 6.3,
        'init_frames': 2,
        'max_age': 30,
        'age_threshold': 2,
    }
    
    # Evaluate on small set of sequences
    tuning_sequences = ['0000', '0001']
    
    print("\nRunning test evaluation...")
    mota = evaluate_params_REAL(test_params, tuning_sequences)
    print(f"\nTest MOTA: {mota:.4f}")
    print("\n✓ Test complete!")
