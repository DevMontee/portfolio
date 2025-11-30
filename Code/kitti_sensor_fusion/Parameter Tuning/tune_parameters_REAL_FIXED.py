"""
PARAMETER TUNING WITH REAL KITTI DATA

This tunes parameters using actual KalmanTracker performance on KITTI sequences.
NOW READS GRID SEARCH FROM CONFIG FILE!
"""

import sys
import json
from pathlib import Path
import numpy as np
from evaluate_params_REAL import evaluate_params_REAL

print("\n" + "="*70)
print("PARAMETER TUNING - REAL KITTI EVALUATION")
print("="*70 + "\n")

# Load config
try:
    config = json.load(open('tuning_config.json'))
    print(f"✓ Config loaded")
except:
    print("ERROR: tuning_config.json not found")
    sys.exit(1)

# Define parameter grid - READ FROM CONFIG FILE
param_grid = config.get('grid_search', {
    'q_pos': [0.1, 0.3, 0.5],
    'q_vel': [0.01, 0.05, 0.1],
    'r_camera': [0.2, 0.5, 1.0],
    'r_lidar': [0.2, 0.5, 1.0],
    'gate_threshold': [6.3, 7.8, 9.0],
    'init_frames': [2, 3, 5],
    'max_age': [30, 40, 50],
    'age_threshold': [2, 3, 4],
})

# Create output dir
output_dir = Path(config['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory: {output_dir}\n")

# ============================================================================
# GRID SEARCH WITH REAL KITTI EVALUATION
# ============================================================================

print("="*70)
print("RUNNING GRID SEARCH WITH REAL KITTI DATA")
print("="*70 + "\n")

tuning_sequences = config.get('tuning_sequences', ['0000'])
kitti_data_dir = config.get('kitti_data_dir', r"D:\KITTI Dataset")

print(f"Tuning sequences: {tuning_sequences}")
print(f"KITTI data dir: {kitti_data_dir}")
print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}\n")

print("Testing parameter combinations...")
print("-"*70)

best_score = -np.inf
best_params = None
tested_count = 0
results = []

# Grid search
for q_pos in param_grid['q_pos']:
    for q_vel in param_grid['q_vel']:
        for r_camera in param_grid['r_camera']:
            for r_lidar in param_grid['r_lidar']:
                for gate in param_grid['gate_threshold']:
                    for init in param_grid['init_frames']:
                        for max_age in param_grid['max_age']:
                            for age_thresh in param_grid['age_threshold']:
                                params = {
                                    'q_pos': q_pos,
                                    'q_vel': q_vel,
                                    'r_camera': r_camera,
                                    'r_lidar': r_lidar,
                                    'gate_threshold': gate,
                                    'init_frames': init,
                                    'max_age': max_age,
                                    'age_threshold': age_thresh,
                                }
                                
                                tested_count += 1
                                
                                # Evaluate on real KITTI data
                                print(f"\n  [{tested_count}] Testing params...")
                                score = evaluate_params_REAL(params, tuning_sequences, kitti_data_dir)
                                
                                results.append({
                                    'params': params,
                                    'score': score
                                })
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    print(f"  ✓ NEW BEST: {score:.4f}")

print(f"\n\n✓ Tested {tested_count} combinations")
print(f"✓ Best MOTA: {best_score:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70 + "\n")

# Save best parameters
best_params_file = output_dir / 'best_parameters.json'
with open(best_params_file, 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"✓ {best_params_file}")

# Save all results
results_file = output_dir / 'tuning_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ {results_file}")

# Save report
report_file = output_dir / 'TUNING_REPORT_REAL.txt'
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("PARAMETER TUNING RESULTS - REAL KITTI EVALUATION\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Grid Search Configuration:\n")
    f.write(f"  Tuning sequences: {tuning_sequences}\n")
    f.write(f"  KITTI data directory: {kitti_data_dir}\n")
    f.write(f"  Total combinations tested: {tested_count}\n")
    f.write(f"  Best MOTA score: {best_score:.4f}\n\n")
    
    f.write("BEST PARAMETERS:\n")
    f.write("-"*70 + "\n")
    for key, val in best_params.items():
        f.write(f"{key:20s}: {val}\n")
    f.write("\n")
    
    f.write("TOP 10 RESULTS:\n")
    f.write("-"*70 + "\n")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        f.write(f"\n{i}. MOTA: {result['score']:.4f}\n")
        for key, val in result['params'].items():
            f.write(f"   {key:18s}: {val}\n")
    f.write("\n")
    
    f.write("USAGE:\n")
    f.write("-"*70 + "\n")
    f.write("import json\n")
    f.write("from kalman_tracker import KalmanTracker\n\n")
    f.write("with open('best_parameters.json', 'r') as f:\n")
    f.write("    params = json.load(f)\n\n")
    f.write("tracker = KalmanTracker(\n")
    f.write("    q_pos=params['q_pos'],\n")
    f.write("    q_vel=params['q_vel'],\n")
    f.write("    r_camera=params['r_camera'],\n")
    f.write("    r_lidar=params['r_lidar'],\n")
    f.write("    gate_threshold=params['gate_threshold'],\n")
    f.write("    init_frames=params['init_frames'],\n")
    f.write("    max_age=params['max_age'],\n")
    f.write("    age_threshold=params['age_threshold']\n")
    f.write(")\n")

print(f"✓ {report_file}")

print("\n" + "="*70)
print("✓ SUCCESS! Real parameter tuning complete")
print("="*70)
print(f"\nResults saved to: {output_dir}")
print(f"\nBest parameters found (MOTA: {best_score:.4f}):")
for key, val in best_params.items():
    print(f"  {key:20s}: {val}")

print(f"\nNext step: Run validation on test set!\n")
