"""
OCCLUSION TESTING SUITE

Runs comprehensive occlusion analysis:
1. Baseline tracking (no occlusions)
2. Random occlusions (10%, 25%, 50%)
3. Temporal occlusions (continuous frames hidden)
4. Measures robustness and re-identification

Run this after your normal tuning to test occlusion handling.
"""

import sys
import json
from pathlib import Path
import numpy as np
import copy

sys.path.insert(0, r"D:\kitti_sensor_fusion\Parameter Tuning")

try:
    from load_kitti_real_data import load_kitti_detections, load_kitti_ground_truth
    from kalman_tracker import KalmanTracker
    from evaluate_params_REAL import compute_mot_metrics
    print("✓ Imported tracking modules\n")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Import occlusion modules
try:
    from occlusion_handler import OcclusionHandler
    from occlusion_generator import SyntheticOcclusionGenerator, OcclusionTestSuite
    from occlusion_evaluator import OcclusionEvaluator
    print("✓ Imported occlusion modules\n")
except ImportError as e:
    print(f"WARNING: Occlusion modules not found locally")
    print(f"Make sure occlusion_*.py files are in the same directory")

print("\n" + "="*70)
print("OCCLUSION ROBUSTNESS TESTING SUITE")
print("="*70 + "\n")

# Load best parameters
best_params_file = Path(r"D:\kitti_sensor_fusion\Parameter Tuning\tuning_results\best_parameters.json")

if not best_params_file.exists():
    print(f"ERROR: {best_params_file} not found")
    print("Run tuning first!")
    sys.exit(1)

with open(best_params_file, 'r') as f:
    best_params = json.load(f)

print(f"✓ Loaded best parameters")
print(f"  q_pos={best_params['q_pos']}, q_vel={best_params['q_vel']}\n")

# Load data
sequence = '0000'
kitti_root = r"D:\KITTI Dataset"

print(f"Loading sequence {sequence}...")
detections_all = load_kitti_detections(sequence, kitti_root)
ground_truth_all = load_kitti_ground_truth(sequence, kitti_root)

print(f"✓ Loaded {len(detections_all)} frames of detections")
print(f"✓ Loaded {len(ground_truth_all)} frames of ground truth\n")

# =========================================================================
# HELPER FUNCTION: Evaluate tracker
# =========================================================================

def evaluate_tracker(detections_dict, ground_truth_dict):
    """
    Run tracker and compute MOT metrics.
    
    Args:
        detections_dict: {frame_id: [detections]}
        ground_truth_dict: {frame_id: [gt_objects]}
        
    Returns:
        (mota, motp, id_switches)
    """
    tracker = KalmanTracker(
        q_pos=best_params['q_pos'],
        q_vel=best_params['q_vel'],
        r_camera=best_params['r_camera'],
        r_lidar=best_params['r_lidar'],
        gate_threshold=best_params['gate_threshold'],
        init_frames=best_params['init_frames'],
        max_age=best_params['max_age'],
        age_threshold=best_params['age_threshold']
    )
    
    tracked_objects = {}
    
    for frame_id in sorted(detections_dict.keys()):
        frame_dets = detections_dict[frame_id]
        
        # Convert format
        tracker_dets = []
        for det in frame_dets:
            tracker_dets.append({
                'x': det['x'],
                'y': det['y'],
                'z': det['z'],
                'size': [det['l'], det['w'], det['h']],
                'conf': det['conf'],
            })
        
        tracked = tracker.update(frame_id, tracker_dets)
        tracked_objects[frame_id] = tracked
    
    # Compute metrics
    mota, motp, id_switches = compute_mot_metrics(tracked_objects, ground_truth_dict)
    
    return mota, motp, id_switches, tracked_objects

# =========================================================================
# TEST 1: BASELINE (No occlusions)
# =========================================================================

print("="*70)
print("TEST 1: BASELINE (No occlusions)")
print("="*70 + "\n")

print("Evaluating baseline tracker...")
mota_baseline, motp_baseline, id_sw_baseline, tracks_baseline = evaluate_tracker(
    detections_all, ground_truth_all
)

print(f"\nBaseline Results:")
print(f"  MOTA:        {mota_baseline:.4f} (59.92% expected)")
print(f"  MOTP:        {motp_baseline:.4f}")
print(f"  ID Switches: {id_sw_baseline}\n")

# =========================================================================
# TEST 2: RANDOM OCCLUSIONS
# =========================================================================

print("="*70)
print("TEST 2: RANDOM OCCLUSIONS")
print("="*70 + "\n")

occlusion_rates = [0.10, 0.25, 0.50]
occlusion_results = {}

for occlusion_rate in occlusion_rates:
    print(f"Testing with {occlusion_rate:.0%} random occlusions...")
    
    # Generate occluded detections
    generator = SyntheticOcclusionGenerator(occlusion_rate=occlusion_rate)
    occluded_dets = copy.deepcopy(detections_all)
    
    total_hidden = 0
    for frame_id in occluded_dets.keys():
        occluded_list, hidden_indices = generator.generate_random_occlusions(occluded_dets[frame_id])
        occluded_dets[frame_id] = occluded_list
        total_hidden += len(hidden_indices)
    
    # Evaluate
    mota, motp, id_switches, _ = evaluate_tracker(occluded_dets, ground_truth_all)
    
    occlusion_results[occlusion_rate] = {
        'mota': mota,
        'motp': motp,
        'id_switches': id_switches,
        'detections_hidden': total_hidden
    }
    
    print(f"  MOTA:        {mota:.4f} (drop: {mota_baseline - mota:.4f})")
    print(f"  MOTP:        {motp:.4f}")
    print(f"  ID Switches: {id_switches} (increase: {id_switches - id_sw_baseline})")
    print()

# =========================================================================
# ANALYSIS & VISUALIZATION
# =========================================================================

print("="*70)
print("ROBUSTNESS ANALYSIS")
print("="*70 + "\n")

evaluator = OcclusionEvaluator()

print(f"{'Occlusion':<15} {'MOTA':<12} {'Drop':<12} {'ID Switches':<15} {'Change':<12}")
print("-"*70)
print(f"{'Baseline':<15} {mota_baseline:<12.4f} {'-':<12} {id_sw_baseline:<15} {'-':<12}")

for occlusion_rate in occlusion_rates:
    results = occlusion_results[occlusion_rate]
    mota_drop = mota_baseline - results['mota']
    id_sw_change = results['id_switches'] - id_sw_baseline
    
    print(f"{occlusion_rate:>12.0%}   {results['mota']:<12.4f} "
          f"{mota_drop:<12.4f} {results['id_switches']:<15} {id_sw_change:+d}")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70 + "\n")

# Robustness assessment
worst_occlusion = max(occlusion_results.items(), key=lambda x: x[1]['mota'])
best_occlusion = min(occlusion_results.items(), key=lambda x: x[1]['mota'])

print(f"Baseline MOTA:              {mota_baseline:.4f}")
print(f"MOTA under 50% occlusion:   {best_occlusion[1]['mota']:.4f}")
print(f"Performance drop:           {mota_baseline - best_occlusion[1]['mota']:.4f} "
      f"({(mota_baseline - best_occlusion[1]['mota']) / mota_baseline * 100:.1f}%)")

if mota_baseline - best_occlusion[1]['mota'] < 0.10:
    print("\n✓ EXCELLENT: Tracker is robust to occlusions!")
elif mota_baseline - best_occlusion[1]['mota'] < 0.20:
    print("\n✓ GOOD: Tracker handles occlusions reasonably well")
else:
    print("\n⚠️  Tracker struggles with significant occlusions")
    print("Consider implementing re-identification mechanisms")

print("\n" + "="*70)

# Save results
output_dir = Path(r"D:\kitti_sensor_fusion\Occlusion Handling")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "OCCLUSION_TEST_RESULTS.json"

results_to_save = {
    'baseline': {
        'mota': float(mota_baseline),
        'motp': float(motp_baseline),
        'id_switches': int(id_sw_baseline)
    },
    'occlusion_tests': {
        str(rate): {
            'mota': float(data['mota']),
            'motp': float(data['motp']),
            'id_switches': int(data['id_switches']),
            'detections_hidden': int(data['detections_hidden'])
        }
        for rate, data in occlusion_results.items()
    }
}

with open(output_file, 'w') as f:
    json.dump(results_to_save, f, indent=2)

print(f"\n✓ Results saved to: {output_file}\n")
