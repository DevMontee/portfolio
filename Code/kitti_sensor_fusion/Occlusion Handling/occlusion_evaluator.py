"""
OCCLUSION EVALUATOR

Measures tracker performance under occlusion:
- Track fragmentation
- ID switch count and severity
- Re-identification success rate
- Trajectory completeness
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class OcclusionEvaluator:
    """
    Evaluates tracking performance with occlusion.
    
    Metrics:
    - ID Switches: Number of times same object gets new ID
    - Track Fragmentation: Ratio of tracks to ground truth objects
    - Re-ID Rate: % of objects successfully re-identified after occlusion
    - Trajectory Completeness: % of frames where track has measurements
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def compute_id_switches(self, 
                           tracked_objects: Dict[int, List[Dict]],
                           ground_truth: Dict[int, List[Dict]]) -> Tuple[int, Dict]:
        """
        Count ID switches - when same GT object gets different track IDs.
        
        Args:
            tracked_objects: {frame_id: [{'track_id': ..., 'x': ..., 'y': ...}]}
            ground_truth: {frame_id: [{'id': ..., 'x': ..., 'y': ...}]}
            
        Returns:
            (total_id_switches, detailed_analysis)
        """
        # Map each GT object to all track IDs it was assigned
        gt_to_track_ids = defaultdict(set)
        
        for frame_id in sorted(ground_truth.keys()):
            if frame_id not in tracked_objects:
                continue
            
            gt_objects = ground_truth[frame_id]
            tracked = tracked_objects[frame_id]
            
            # Match tracked to GT (simple position-based matching)
            for gt_obj in gt_objects:
                gt_id = gt_obj.get('id', -1)
                gt_pos = np.array([gt_obj.get('x', 0), gt_obj.get('y', 0), gt_obj.get('z', 0)])
                
                # Find closest tracked object
                best_track_id = -1
                best_distance = float('inf')
                
                for tracked_obj in tracked:
                    track_id = tracked_obj.get('track_id', -1)
                    track_pos = np.array([tracked_obj.get('x', 0), tracked_obj.get('y', 0), 
                                         tracked_obj.get('z', 0)])
                    
                    distance = np.linalg.norm(gt_pos - track_pos)
                    if distance < best_distance and distance < 2.0:  # 2m threshold
                        best_distance = distance
                        best_track_id = track_id
                
                if best_track_id >= 0:
                    gt_to_track_ids[gt_id].add(best_track_id)
        
        # Count ID switches (more than 1 track ID per GT object)
        id_switches = 0
        switch_details = {}
        
        for gt_id, track_ids in gt_to_track_ids.items():
            if len(track_ids) > 1:
                id_switches += len(track_ids) - 1
                switch_details[gt_id] = list(track_ids)
        
        return id_switches, switch_details
    
    def compute_track_fragmentation(self,
                                   tracked_objects: Dict[int, List[Dict]],
                                   ground_truth: Dict[int, List[Dict]]) -> Tuple[float, Dict]:
        """
        Compute track fragmentation ratio.
        
        Fragmentation = Number of tracks / Number of ground truth objects
        
        Values:
        - 1.0 = Perfect (one track per object)
        - > 1.0 = Fragmented (object split into multiple tracks)
        
        Args:
            tracked_objects: Tracked objects
            ground_truth: Ground truth objects
            
        Returns:
            (fragmentation_ratio, analysis)
        """
        # Count unique track IDs and GT IDs
        all_track_ids = set()
        all_gt_ids = set()
        
        for frame_id in sorted(tracked_objects.keys()):
            for obj in tracked_objects[frame_id]:
                track_id = obj.get('track_id', -1)
                if track_id >= 0:
                    all_track_ids.add(track_id)
        
        for frame_id in sorted(ground_truth.keys()):
            for obj in ground_truth[frame_id]:
                gt_id = obj.get('id', -1)
                if gt_id >= 0:
                    all_gt_ids.add(gt_id)
        
        num_tracks = len(all_track_ids)
        num_gt_objects = len(all_gt_ids)
        
        if num_gt_objects == 0:
            fragmentation_ratio = 1.0
        else:
            fragmentation_ratio = num_tracks / num_gt_objects
        
        analysis = {
            'num_tracks': num_tracks,
            'num_gt_objects': num_gt_objects,
            'fragmentation_ratio': fragmentation_ratio,
            'extra_tracks': max(0, num_tracks - num_gt_objects)
        }
        
        return fragmentation_ratio, analysis
    
    def compute_trajectory_completeness(self,
                                       tracked_objects: Dict[int, List[Dict]],
                                       ground_truth: Dict[int, List[Dict]]) -> Tuple[float, Dict]:
        """
        Compute trajectory completeness - % of frames where object is tracked.
        
        Measures how well track is maintained (high = good re-ID).
        
        Args:
            tracked_objects: Tracked objects
            ground_truth: Ground truth objects
            
        Returns:
            (mean_completeness, analysis_by_object)
        """
        # For each GT object, measure % of frames where tracked
        completeness_by_object = {}
        
        for frame_id in sorted(ground_truth.keys()):
            gt_objects = ground_truth[frame_id]
            tracked = tracked_objects.get(frame_id, [])
            
            for gt_obj in gt_objects:
                gt_id = gt_obj.get('id', -1)
                if gt_id < 0:
                    continue
                
                if gt_id not in completeness_by_object:
                    completeness_by_object[gt_id] = {
                        'total_frames': 0,
                        'tracked_frames': 0,
                        'frames_with_measurement': 0
                    }
                
                completeness_by_object[gt_id]['total_frames'] += 1
                
                # Check if this GT object is tracked in this frame
                gt_pos = np.array([gt_obj.get('x', 0), gt_obj.get('y', 0), gt_obj.get('z', 0)])
                
                for tracked_obj in tracked:
                    track_pos = np.array([tracked_obj.get('x', 0), tracked_obj.get('y', 0), 
                                         tracked_obj.get('z', 0)])
                    
                    distance = np.linalg.norm(gt_pos - track_pos)
                    if distance < 2.0:  # 2m threshold
                        completeness_by_object[gt_id]['tracked_frames'] += 1
                        if 'has_detection' in tracked_obj:
                            completeness_by_object[gt_id]['frames_with_measurement'] += 1
                        break
        
        # Calculate completeness percentages
        completeness_values = []
        for gt_id, data in completeness_by_object.items():
            if data['total_frames'] > 0:
                completeness = data['tracked_frames'] / data['total_frames']
                completeness_values.append(completeness)
        
        mean_completeness = np.mean(completeness_values) if completeness_values else 0.0
        
        return mean_completeness, completeness_by_object
    
    def compute_occlusion_robustness(self,
                                    results_baseline: Dict,
                                    results_occluded: Dict) -> Dict:
        """
        Measure how much performance degrades with occlusions.
        
        Args:
            results_baseline: Metrics without occlusions {mota, motp, id_switches}
            results_occluded: Metrics with occlusions {mota, motp, id_switches}
            
        Returns:
            Robustness analysis
        """
        if not results_baseline or not results_occluded:
            return {}
        
        mota_drop = results_baseline['mota'] - results_occluded['mota']
        mota_drop_pct = (mota_drop / results_baseline['mota'] * 100) if results_baseline['mota'] > 0 else 0
        
        id_switch_increase = results_occluded['id_switches'] - results_baseline['id_switches']
        
        robustness = {
            'baseline_mota': results_baseline['mota'],
            'occluded_mota': results_occluded['mota'],
            'mota_drop': mota_drop,
            'mota_drop_pct': mota_drop_pct,
            'baseline_id_switches': results_baseline['id_switches'],
            'occluded_id_switches': results_occluded['id_switches'],
            'id_switch_increase': id_switch_increase,
            'robustness_score': 1.0 - (mota_drop_pct / 100.0)  # 1.0 = no drop, 0.0 = complete failure
        }
        
        return robustness
    
    def print_analysis(self, 
                      id_switches: int,
                      id_switch_details: Dict,
                      fragmentation: float,
                      fragmentation_analysis: Dict,
                      completeness: float,
                      completeness_by_object: Dict) -> None:
        """Print detailed analysis."""
        
        print("\n" + "="*70)
        print("OCCLUSION EVALUATION RESULTS")
        print("="*70)
        
        # ID Switches
        print(f"\nID SWITCHES:")
        print(f"  Total ID switches: {id_switches}")
        if id_switch_details:
            print(f"  Objects with switches: {len(id_switch_details)}")
            for gt_id, track_ids in list(id_switch_details.items())[:3]:
                print(f"    GT {gt_id}: assigned to tracks {track_ids}")
        
        # Fragmentation
        print(f"\nTRACK FRAGMENTATION:")
        print(f"  Fragmentation ratio: {fragmentation:.3f}")
        print(f"  (1.0 = optimal, >1.0 = fragmented)")
        print(f"  Total tracks: {fragmentation_analysis['num_tracks']}")
        print(f"  Total GT objects: {fragmentation_analysis['num_gt_objects']}")
        print(f"  Extra tracks: {fragmentation_analysis['extra_tracks']}")
        
        # Completeness
        print(f"\nTRAJECTORY COMPLETENESS:")
        print(f"  Mean completeness: {completeness:.2%}")
        print(f"  (% of frames where object is tracked)")
        
        print("="*70 + "\n")
