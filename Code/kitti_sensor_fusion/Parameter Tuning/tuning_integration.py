"""
Parameter Tuning Integration Guide
Connecting tuning framework to KITTI-based multi-object tracking system
"""

import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class KITTIEvaluationMetrics:
    """Compute KITTI tracking evaluation metrics for parameter validation"""
    
    @staticmethod
    def compute_mota(tp: int, fp: int, fn: int, gt_count: int) -> float:
        """
        Multiple Object Tracking Accuracy
        MOTA = 1 - (FN + FP + IDSW) / GT
        
        For simplicity here: MOTA = 1 - (FN + FP) / GT
        """
        if gt_count == 0:
            return 0.0
        return 1.0 - (fn + fp) / gt_count
    
    @staticmethod
    def compute_motp(distances: List[float], num_matches: int) -> float:
        """
        Multiple Object Tracking Precision
        MOTP = sum(distances) / num_matches
        """
        if num_matches == 0:
            return float('inf')
        return np.mean(distances)
    
    @staticmethod
    def compute_precision(tp: int, fp: int) -> float:
        """Detection precision"""
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def compute_recall(tp: int, fn: int) -> float:
        """Detection recall"""
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)


class TrackingEvaluationWrapper:
    """Wrapper to evaluate tracker with different parameters"""
    
    def __init__(self, kalman_tracker_class, kitti_sequences: List[str]):
        """
        Args:
            kalman_tracker_class: Your KalmanTracker class
            kitti_sequences: List of sequence names to evaluate on
        """
        self.tracker_class = kalman_tracker_class
        self.kitti_sequences = kitti_sequences
        self.results_cache = {}
    
    def evaluate_parameters(self, params, sensor_type='fusion') -> float:
        """
        Evaluate tracker with given parameters
        
        Returns: negative MOTA (for minimization)
        """
        # Create parameter hash for caching
        param_hash = hash(tuple(sorted(params.to_dict().items())))
        
        if param_hash in self.results_cache:
            return self.results_cache[param_hash]
        
        total_mota = 0.0
        sequence_results = []
        
        for sequence in self.kitti_sequences:
            try:
                # Load KITTI sequence
                detections = self._load_kitti_detections(sequence, sensor_type)
                ground_truth = self._load_kitti_ground_truth(sequence)
                
                # Run tracker with parameters
                tracker = self.tracker_class(
                    q_pos=params.q_pos,
                    q_vel=params.q_vel,
                    r_camera=params.r_camera,
                    r_lidar=params.r_lidar,
                    gate_threshold=params.gate_threshold,
                    init_frames=params.init_frames,
                    max_age=params.max_age,
                    age_threshold=params.age_threshold
                )
                
                tracks = self._run_tracker(tracker, detections)
                
                # Evaluate
                tp, fp, fn, matches_dist = self._compute_matches(
                    tracks, ground_truth
                )
                
                gt_count = len(ground_truth)
                mota = KITTIEvaluationMetrics.compute_mota(tp, fp, fn, gt_count)
                
                sequence_results.append({
                    'sequence': sequence,
                    'mota': mota,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                })
                
                total_mota += mota
                
            except Exception as e:
                logger.warning(f"Error evaluating sequence {sequence}: {e}")
                continue
        
        if not sequence_results:
            return float('inf')
        
        avg_mota = total_mota / len(sequence_results)
        metric = -avg_mota  # Negative for minimization
        
        self.results_cache[param_hash] = metric
        logger.debug(f"Params → MOTA: {avg_mota:.4f}")
        
        return metric
    
    def _load_kitti_detections(self, sequence: str, sensor_type: str) -> List[Dict]:
        """
        Load detections from sensor
        
        Returns list of detection dicts for each frame:
        {
            'frame': int,
            'dets': [
                {'x': float, 'y': float, 'z': float, 'size': [l,w,h], 'conf': float},
                ...
            ]
        }
        """
        kitti_root = r"D:\KITTI Dataset"
        det_file = os.path.join(kitti_root, "detections_test", f"{sequence}_{sensor_type}.txt")
        
        detections = []
        
        try:
            with open(det_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if len(parts) < 8:
                        continue
                    
                    frame_id = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    l = float(parts[4])
                    w = float(parts[5])
                    h = float(parts[6])
                    conf = float(parts[7])
                    
                    # Group by frame
                    if not detections or detections[-1]['frame'] != frame_id:
                        detections.append({
                            'frame': frame_id,
                            'dets': []
                        })
                    
                    detections[-1]['dets'].append({
                        'x': x,
                        'y': y,
                        'z': z,
                        'size': [l, w, h],
                        'conf': conf
                    })
        
        except FileNotFoundError:
            logger.warning(f"Detection file not found: {det_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading detections: {e}")
            return []
        
        return detections
    
    def _load_kitti_ground_truth(self, sequence: str) -> List[Dict]:
        """
        Load ground truth annotations
        
        Returns list of GT objects:
        {
            'frame': int,
            'gt_id': int,
            'bbox_3d': [x, y, z, l, w, h, rot],
            ...
        }
        """
        kitti_root = r"D:\KITTI Dataset"
        label_file = os.path.join(
            kitti_root, 
            "data_tracking_label_2", 
            "training", 
            "label_02", 
            f"{sequence}.txt"
        )
        
        ground_truth = []
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if len(parts) < 15:
                        continue
                    
                    try:
                        frame_num = int(parts[0])
                        track_id = int(parts[1])
                        obj_type = parts[2]
                        
                        # 3D dimensions (height, width, length)
                        height = float(parts[8])
                        width = float(parts[9])
                        length = float(parts[10])
                        
                        # 3D location (x, y, z)
                        x = float(parts[11])
                        y = float(parts[12])
                        z = float(parts[13])
                        
                        # Rotation
                        rotation = float(parts[14])
                        
                        ground_truth.append({
                            'frame': frame_num,
                            'gt_id': track_id,
                            'bbox_3d': [x, y, z, length, width, height, rotation],
                            'type': obj_type
                        })
                    
                    except (ValueError, IndexError):
                        continue
        
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {label_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return []
        
        return ground_truth
    
    def _run_tracker(self, tracker, detections: List[Dict]) -> List[Dict]:
        """
        Run tracker over sequence
        
        Returns list of tracked objects with IDs and predictions
        """
        tracks = []
        
        for frame_data in detections:
            frame_id = frame_data['frame']
            frame_dets = frame_data['dets']
            
            # Update tracker with detections
            tracked_objects = tracker.update(frame_id, frame_dets)
            
            for obj in tracked_objects:
                tracks.append({
                    'frame': frame_id,
                    'track_id': obj['id'],
                    'bbox_3d': obj['bbox_3d'],
                    'state': obj.get('state', None)
                })
        
        return tracks
    
    def _compute_matches(self, predicted_tracks: List[Dict],
                        ground_truth: List[Dict]) -> Tuple[int, int, int, List[float]]:
        """
        Match predicted tracks to ground truth
        
        Returns: (tp, fp, fn, match_distances)
        """
        # Group by frame
        pred_by_frame = self._group_by_frame(predicted_tracks)
        gt_by_frame = self._group_by_frame(ground_truth)
        
        tp, fp, fn = 0, 0, 0
        match_distances = []
        
        # For each frame
        for frame_id in gt_by_frame:
            if frame_id not in pred_by_frame:
                fn += len(gt_by_frame[frame_id])
                continue
            
            gt_objects = gt_by_frame[frame_id]
            pred_objects = pred_by_frame[frame_id]
            
            # Match using Hungarian algorithm or simple greedy matching
            matched_pairs = self._match_detections(pred_objects, gt_objects)
            
            tp += len(matched_pairs)
            fp += len(pred_objects) - len(matched_pairs)
            fn += len(gt_objects) - len(matched_pairs)
            
            # Collect distances for MOTP
            for pred_idx, gt_idx in matched_pairs:
                dist = self._compute_bbox_distance(
                    pred_objects[pred_idx],
                    gt_objects[gt_idx]
                )
                match_distances.append(dist)
        
        # Handle frames with predictions but no GT
        for frame_id in pred_by_frame:
            if frame_id not in gt_by_frame:
                fp += len(pred_by_frame[frame_id])
        
        return tp, fp, fn, match_distances
    
    @staticmethod
    def _group_by_frame(objects: List[Dict]) -> Dict[int, List[Dict]]:
        """Group objects by frame ID"""
        grouped = {}
        for obj in objects:
            frame = obj['frame']
            if frame not in grouped:
                grouped[frame] = []
            grouped[frame].append(obj)
        return grouped
    
    @staticmethod
    def _match_detections(predictions: List[Dict],
                         ground_truth: List[Dict]) -> List[Tuple[int, int]]:
        """
        Match predictions to ground truth using greedy nearest-neighbor
        
        Returns list of (pred_idx, gt_idx) pairs
        """
        matched = []
        used_gt = set()
        
        # For each prediction, find nearest GT
        for pred_idx, pred in enumerate(predictions):
            best_dist = float('inf')
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt:
                    continue
                
                dist = np.linalg.norm(
                    np.array(pred['bbox_3d'][:3]) - np.array(gt['bbox_3d'][:3])
                )
                
                # Match threshold (e.g., 1 meter in 3D space)
                if dist < best_dist and dist < 1.0:
                    best_dist = dist
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched.append((pred_idx, best_gt_idx))
                used_gt.add(best_gt_idx)
        
        return matched
    
    @staticmethod
    def _compute_bbox_distance(pred_obj: Dict, gt_obj: Dict) -> float:
        """Compute 3D IoU or center distance between boxes"""
        # Simple center distance
        pred_center = np.array(pred_obj['bbox_3d'][:3])
        gt_center = np.array(gt_obj['bbox_3d'][:3])
        return float(np.linalg.norm(pred_center - gt_center))


class TuningExperimentRunner:
    """Run complete tuning experiment with logging"""
    
    def __init__(self, tracker_class, kitti_sequences: List[str],
                 validation_split: float = 0.8):
        """
        Args:
            tracker_class: Your KalmanTracker class
            kitti_sequences: All available sequence names
            validation_split: Fraction for tuning (0.8 = use 80% for tuning)
        """
        self.tracker_class = tracker_class
        
        # Split sequences into tuning and test sets
        n_tune = int(len(kitti_sequences) * validation_split)
        self.tune_sequences = kitti_sequences[:n_tune]
        self.test_sequences = kitti_sequences[n_tune:]
        
        self.evaluator = TrackingEvaluationWrapper(tracker_class, self.tune_sequences)
        self.test_evaluator = TrackingEvaluationWrapper(tracker_class, self.test_sequences)
        
        logger.info(f"Tuning on {len(self.tune_sequences)} sequences")
        logger.info(f"Testing on {len(self.test_sequences)} sequences")
    
    def run_tuning(self, use_bayesian: bool = True):
        """Run parameter tuning"""
        from parameter_tuning import ParameterTuningPipeline
        
        pipeline = ParameterTuningPipeline(
            metric_fn=self.evaluator.evaluate_parameters,
            validation_data=self.tune_sequences,
            output_dir='./kitti_tuning_results'
        )
        
        best_params = pipeline.run_full_tuning(use_bayesian=use_bayesian)
        return best_params
    
    def evaluate_on_test_set(self, best_params) -> Dict:
        """Evaluate best parameters on held-out test set"""
        logger.info("\nEvaluating on test set...")
        
        test_metric = self.test_evaluator.evaluate_parameters(best_params)
        test_mota = -test_metric
        
        # Also evaluate single-sensor baselines on test set
        baseline_evals = {
            'camera': self.test_evaluator.evaluate_parameters(
                self._create_camera_only_params(best_params)
            ),
            'lidar': self.test_evaluator.evaluate_parameters(
                self._create_lidar_only_params(best_params)
            ),
        }
        
        results = {
            'test_mota_fusion': test_mota,
            'test_mota_camera': -baseline_evals['camera'],
            'test_mota_lidar': -baseline_evals['lidar'],
            'fusion_gain_vs_camera': test_mota - (-baseline_evals['camera']),
            'fusion_gain_vs_lidar': test_mota - (-baseline_evals['lidar']),
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST SET RESULTS")
        logger.info("=" * 60)
        for metric_name, value in results.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        return results
    
    @staticmethod
    def _create_camera_only_params(base_params):
        """Create camera-only variant"""
        from parameter_tuning import KalmanParameters
        params = KalmanParameters(**base_params.to_dict())
        params.r_lidar = float('inf')
        return params
    
    @staticmethod
    def _create_lidar_only_params(base_params):
        """Create LiDAR-only variant"""
        from parameter_tuning import KalmanParameters
        params = KalmanParameters(**base_params.to_dict())
        params.r_camera = float('inf')
        return params


# Integration example
def integrate_with_your_tracker(KalmanTrackerClass):
    """
    Example of how to integrate tuning into your workflow
    
    Usage:
        from your_tracker_module import KalmanTracker
        integrate_with_your_tracker(KalmanTracker)
    """
    
    # Define KITTI sequences to use
    kitti_sequences = [
        '0001', '0002', '0003', '0004', '0005',  # Tuning set
        '0006', '0007', '0008', '0009', '0010',  # Test set
    ]
    
    # Create experiment runner
    runner = TuningExperimentRunner(
        tracker_class=KalmanTrackerClass,
        kitti_sequences=kitti_sequences,
        validation_split=0.5  # Use 50% for tuning, 50% for testing
    )
    
    # Run parameter tuning
    logger.info("Starting parameter tuning...")
    best_params = runner.run_tuning(use_bayesian=True)
    
    # Evaluate on test set
    test_results = runner.evaluate_on_test_set(best_params)
    
    return best_params, test_results


# Practical integration checklist
INTEGRATION_CHECKLIST = """
PARAMETER TUNING INTEGRATION CHECKLIST
=====================================

1. Connect Data Loading:
   [ ] Implement _load_kitti_detections() 
       - Load your detection results (camera/LiDAR/fused)
   [ ] Implement _load_kitti_ground_truth()
       - Load KITTI annotation files
   
2. Connect Tracker:
   [ ] Ensure KalmanTracker.__init__() accepts:
       - q_pos, q_vel (process noise)
       - r_camera, r_lidar (measurement noise)
       - gate_threshold (Mahalanobis distance)
       - init_frames, max_age, age_threshold (lifecycle)
   [ ] Tracker.update() returns list of tracked objects with:
       - 'id': int (track ID)
       - 'bbox_3d': [x,y,z,l,w,h,rot]
       - 'state': KalmanFilter state (optional)

3. Set Up Evaluation:
   [ ] Define which KITTI sequences to use
   [ ] Set validation split (typically 0.7-0.8)
   [ ] Configure logging output directory

4. Run Tuning:
   [ ] Execute integrate_with_your_tracker()
   [ ] Monitor progress (grid search takes time)
   [ ] Check results in ./kitti_tuning_results/

5. Analyze Results:
   [ ] Review best_parameters.json
   [ ] Check all_tuning_results.csv for trends
   [ ] Compare test results vs baselines
   [ ] Verify fusion gain is positive

6. Deploy:
   [ ] Load best parameters into production tracker
   [ ] Test on fresh sequences
   [ ] Monitor performance over time
"""

if __name__ == '__main__':
    print(INTEGRATION_CHECKLIST)
