"""
OCCLUSION HANDLER

Handles track predictions during occlusions and re-identification when objects reappear.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist


class OcclusionHandler:
    """
    Handles track continuity during occlusions.
    
    Features:
    - Predicts track position during occlusion (up to max_occlusion_frames)
    - Re-identifies tracks when object reappears
    - Tracks occlusion duration and re-ID success
    """
    
    def __init__(self, max_occlusion_frames: int = 10, 
                 position_threshold: float = 2.0,
                 velocity_weight: float = 0.5):
        """
        Initialize occlusion handler.
        
        Args:
            max_occlusion_frames: Maximum frames to predict before deleting track
            position_threshold: Max distance to match re-appearing object (meters)
            velocity_weight: Weight for velocity in prediction (0-1)
        """
        self.max_occlusion_frames = max_occlusion_frames
        self.position_threshold = position_threshold
        self.velocity_weight = velocity_weight
        
        # Tracking statistics
        self.occlusion_stats = {
            'total_occlusions': 0,
            'successful_reidentifications': 0,
            'failed_reidentifications': 0,
            'max_occlusion_duration': 0,
            'occlusion_events': []  # List of {track_id, start_frame, end_frame, duration, reidentified}
        }
    
    def predict_position(self, track: Dict, frames_occluded: int) -> np.ndarray:
        """
        Predict track position during occlusion using Kalman filter extrapolation.
        
        Args:
            track: Track dictionary with 'position' and 'velocity'
            frames_occluded: Number of frames object has been occluded
            
        Returns:
            Predicted position (x, y, z)
        """
        if 'position' not in track or 'velocity' not in track:
            return track.get('position', np.array([0, 0, 0]))
        
        position = np.array(track['position'])
        velocity = np.array(track['velocity'])
        
        # Exponential decay of velocity confidence over time
        velocity_decay = np.exp(-frames_occluded * 0.1)
        
        # Predict position: current + velocity * time * decay
        predicted = position + velocity * frames_occluded * velocity_decay
        
        return predicted
    
    def find_reidentification_match(self, 
                                   predicted_track: Dict,
                                   current_detections: List[Dict],
                                   unmatched_detections: List[int]) -> Tuple[int, float]:
        """
        Find best matching detection for re-identification.
        
        Args:
            predicted_track: Track with predicted position
            current_detections: All detections in current frame
            unmatched_detections: Indices of unmatched detections
            
        Returns:
            (detection_idx, confidence) - Best match or (-1, 0)
        """
        if not unmatched_detections or 'position' not in predicted_track:
            return -1, 0.0
        
        predicted_pos = np.array(predicted_track['position'])
        
        best_idx = -1
        best_distance = float('inf')
        
        # Try to match with unmatched detections
        for det_idx in unmatched_detections:
            if det_idx >= len(current_detections):
                continue
            
            detection = current_detections[det_idx]
            det_pos = np.array([detection['x'], detection['y'], detection['z']])
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(predicted_pos - det_pos)
            
            if distance < best_distance:
                best_distance = distance
                best_idx = det_idx
        
        # Convert distance to confidence (lower distance = higher confidence)
        if best_distance < self.position_threshold:
            confidence = 1.0 - (best_distance / self.position_threshold)
            return best_idx, confidence
        
        return -1, 0.0
    
    def update_track_state(self, track: Dict, frame_idx: int, 
                           is_occluded: bool = False) -> None:
        """
        Update track's occlusion state.
        
        Args:
            track: Track to update
            frame_idx: Current frame index
            is_occluded: Whether track is occluded in this frame
        """
        if 'occlusion_info' not in track:
            track['occlusion_info'] = {
                'occluded': False,
                'occlusion_start_frame': None,
                'frames_occluded': 0,
                'reidentification_count': 0
            }
        
        occ_info = track['occlusion_info']
        
        if is_occluded:
            if not occ_info['occluded']:
                # Start new occlusion
                occ_info['occluded'] = True
                occ_info['occlusion_start_frame'] = frame_idx
                occ_info['frames_occluded'] = 0
                self.occlusion_stats['total_occlusions'] += 1
            else:
                # Continue occlusion
                occ_info['frames_occluded'] += 1
                self.occlusion_stats['max_occlusion_duration'] = max(
                    self.occlusion_stats['max_occlusion_duration'],
                    occ_info['frames_occluded']
                )
        else:
            if occ_info['occluded']:
                # Occlusion ended - object was re-identified!
                duration = occ_info['frames_occluded']
                self.occlusion_stats['successful_reidentifications'] += 1
                occ_info['reidentification_count'] += 1
                
                # Record event
                self.occlusion_stats['occlusion_events'].append({
                    'track_id': track.get('track_id', -1),
                    'start_frame': occ_info['occlusion_start_frame'],
                    'end_frame': frame_idx,
                    'duration': duration,
                    'reidentified': True
                })
            
            occ_info['occluded'] = False
            occ_info['occlusion_start_frame'] = None
            occ_info['frames_occluded'] = 0
    
    def should_delete_track(self, track: Dict) -> bool:
        """
        Determine if track should be deleted due to prolonged occlusion.
        
        Args:
            track: Track to evaluate
            
        Returns:
            True if track should be deleted
        """
        occ_info = track.get('occlusion_info', {})
        
        if occ_info.get('occluded', False):
            if occ_info.get('frames_occluded', 0) > self.max_occlusion_frames:
                return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get occlusion handling statistics."""
        stats = self.occlusion_stats.copy()
        
        if stats['total_occlusions'] > 0:
            stats['reidentification_rate'] = (
                stats['successful_reidentifications'] / stats['total_occlusions']
            )
        else:
            stats['reidentification_rate'] = 0.0
        
        return stats
    
    def print_statistics(self) -> None:
        """Print occlusion handling statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("OCCLUSION HANDLING STATISTICS")
        print("="*70)
        print(f"\nTotal occlusion events:       {stats['total_occlusions']}")
        print(f"Successful re-identifications: {stats['successful_reidentifications']}")
        print(f"Failed re-identifications:     {stats['failed_reidentifications']}")
        print(f"Re-identification rate:       {stats['reidentification_rate']:.2%}")
        print(f"Max occlusion duration:       {stats['max_occlusion_duration']} frames")
        
        if stats['occlusion_events']:
            print(f"\nOcclusion Events:")
            print("-"*70)
            for event in stats['occlusion_events'][:5]:  # Show first 5
                print(f"  Track {event['track_id']:3d}: "
                      f"Frames {event['start_frame']}-{event['end_frame']} "
                      f"(duration: {event['duration']} frames) "
                      f"→ {'Re-ID' if event['reidentified'] else 'Lost'}")
        
        print("="*70 + "\n")
