"""
SYNTHETIC OCCLUSION GENERATOR

Creates realistic occlusion scenarios by hiding real detections.
Tests tracker robustness to missing detections.
"""

import numpy as np
from typing import List, Dict, Tuple
import copy


class SyntheticOcclusionGenerator:
    """
    Generates synthetic occlusions by randomly hiding detections.
    
    Strategies:
    - Random occlusions: Randomly hide detections
    - Spatial occlusions: Hide detections in specific regions
    - Temporal occlusions: Hide detections in continuous time windows
    """
    
    def __init__(self, occlusion_rate: float = 0.20, random_seed: int = 42):
        """
        Initialize occlusion generator.
        
        Args:
            occlusion_rate: Fraction of detections to hide (0-1)
            random_seed: Seed for reproducibility
        """
        self.occlusion_rate = occlusion_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.occlusion_log = {
            'total_detections': 0,
            'hidden_detections': 0,
            'occlusion_rate_actual': 0.0,
            'occluded_indices': []
        }
    
    def generate_random_occlusions(self, 
                                  detections: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """
        Generate random occlusions by hiding random detections.
        
        Args:
            detections: List of detections
            
        Returns:
            (occluded_detections, hidden_indices)
        """
        if not detections:
            return detections, []
        
        # Calculate number to hide
        num_to_hide = int(len(detections) * self.occlusion_rate)
        
        # Random indices to hide
        hidden_indices = np.random.choice(
            len(detections), 
            size=num_to_hide, 
            replace=False
        ).tolist()
        
        # Remove detections at hidden indices
        occluded_detections = [
            det for i, det in enumerate(detections)
            if i not in hidden_indices
        ]
        
        # Log
        self.occlusion_log['total_detections'] = len(detections)
        self.occlusion_log['hidden_detections'] = num_to_hide
        self.occlusion_log['occlusion_rate_actual'] = num_to_hide / len(detections)
        self.occlusion_log['occluded_indices'] = hidden_indices
        
        return occluded_detections, hidden_indices
    
    def generate_spatial_occlusions(self, 
                                    detections: List[Dict],
                                    occlusion_region: Tuple[float, float, float, float]) -> Tuple[List[Dict], List[int]]:
        """
        Generate spatial occlusions - hide detections in a specific region.
        
        Args:
            detections: List of detections
            occlusion_region: (x_min, x_max, y_min, y_max) in camera frame
            
        Returns:
            (occluded_detections, hidden_indices)
        """
        x_min, x_max, y_min, y_max = occlusion_region
        
        hidden_indices = []
        for i, det in enumerate(detections):
            x = det.get('x', 0)
            y = det.get('y', 0)
            
            # Check if detection is in occlusion region
            if x_min <= x <= x_max and y_min <= y <= y_max:
                hidden_indices.append(i)
        
        # Remove hidden detections
        occluded_detections = [
            det for i, det in enumerate(detections)
            if i not in hidden_indices
        ]
        
        # Log
        self.occlusion_log['total_detections'] = len(detections)
        self.occlusion_log['hidden_detections'] = len(hidden_indices)
        if len(detections) > 0:
            self.occlusion_log['occlusion_rate_actual'] = len(hidden_indices) / len(detections)
        self.occlusion_log['occluded_indices'] = hidden_indices
        
        return occluded_detections, hidden_indices
    
    def generate_temporal_occlusions(self,
                                    detections_sequence: Dict[int, List[Dict]],
                                    occlusion_duration: int = 5) -> Dict[int, List[Dict]]:
        """
        Generate temporal occlusions - hide detections for continuous frames.
        
        Simulates objects being occluded for N consecutive frames.
        
        Args:
            detections_sequence: {frame_id: [detections]}
            occlusion_duration: Number of consecutive frames to occlude
            
        Returns:
            {frame_id: [occluded_detections]}
        """
        occluded_sequence = copy.deepcopy(detections_sequence)
        
        frame_ids = sorted(detections_sequence.keys())
        if not frame_ids:
            return occluded_sequence
        
        # Randomly select occlusion windows
        num_windows = max(1, int(len(frame_ids) * self.occlusion_rate / occlusion_duration))
        
        for _ in range(num_windows):
            # Random start frame
            start_frame_idx = np.random.randint(0, len(frame_ids) - occlusion_duration)
            start_frame = frame_ids[start_frame_idx]
            
            # Hide detections for N consecutive frames
            for i in range(occlusion_duration):
                frame_idx = start_frame_idx + i
                if frame_idx < len(frame_ids):
                    frame_id = frame_ids[frame_idx]
                    # Hide random detection in this frame
                    if occluded_sequence[frame_id]:
                        random_idx = np.random.randint(0, len(occluded_sequence[frame_id]))
                        occluded_sequence[frame_id].pop(random_idx)
        
        return occluded_sequence
    
    def get_statistics(self) -> Dict:
        """Get occlusion generation statistics."""
        return self.occlusion_log.copy()
    
    def print_statistics(self) -> None:
        """Print occlusion statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("SYNTHETIC OCCLUSION GENERATION")
        print("="*70)
        print(f"\nTotal detections:      {stats['total_detections']}")
        print(f"Hidden detections:     {stats['hidden_detections']}")
        print(f"Occlusion rate:        {stats['occlusion_rate_actual']:.2%}")
        print(f"Target occlusion rate: {self.occlusion_rate:.2%}")
        print("="*70 + "\n")


class OcclusionTestSuite:
    """
    Runs multiple occlusion scenarios to evaluate tracker robustness.
    """
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {}
    
    def run_occlusion_tests(self, 
                           detections: Dict[int, List[Dict]],
                           tracker_evaluate_fn,
                           occlusion_rates: List[float] = [0.0, 0.10, 0.25, 0.50]) -> Dict:
        """
        Run tracker with different occlusion rates.
        
        Args:
            detections: {frame_id: [detections]}
            tracker_evaluate_fn: Function that evaluates tracker and returns MOTA
            occlusion_rates: List of occlusion rates to test
            
        Returns:
            {occlusion_rate: {'mota': score, 'motp': score, 'id_switches': count}}
        """
        results = {}
        
        for occlusion_rate in occlusion_rates:
            print(f"\nTesting with {occlusion_rate:.0%} occlusions...")
            
            # Generate occluded detections
            generator = SyntheticOcclusionGenerator(occlusion_rate)
            occluded_dets = copy.deepcopy(detections)
            
            for frame_id in occluded_dets.keys():
                occluded_list, _ = generator.generate_random_occlusions(occluded_dets[frame_id])
                occluded_dets[frame_id] = occluded_list
            
            # Evaluate tracker
            mota, motp, id_switches = tracker_evaluate_fn(occluded_dets)
            
            results[occlusion_rate] = {
                'mota': mota,
                'motp': motp,
                'id_switches': id_switches,
                'occlusion_rate': occlusion_rate
            }
            
            print(f"  MOTA: {mota:.4f}, MOTP: {motp:.4f}, ID Switches: {id_switches}")
        
        self.test_results = results
        return results
    
    def print_results(self) -> None:
        """Print test results."""
        if not self.test_results:
            print("No test results available")
            return
        
        print("\n" + "="*70)
        print("OCCLUSION ROBUSTNESS TEST RESULTS")
        print("="*70)
        print(f"\n{'Occlusion':<15} {'MOTA':<15} {'MOTP':<15} {'ID Switches':<15}")
        print("-"*70)
        
        for rate, results in sorted(self.test_results.items()):
            print(f"{rate:>12.0%}    {results['mota']:>12.4f}    "
                  f"{results['motp']:>12.4f}    {results['id_switches']:>12d}")
        
        print("="*70 + "\n")
