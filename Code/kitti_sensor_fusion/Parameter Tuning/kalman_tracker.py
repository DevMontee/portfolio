"""
MULTI-OBJECT TRACKER WRAPPER

Wraps the ExtendedKalmanFilter (single object) into a full multi-object tracker.
Uses your existing EKF and adds:
- Data association
- Track management
- Multi-object tracking
- Tuning parameter interface
"""

import numpy as np
from typing import List, Dict
import sys
import importlib.util

# Handle import with spaces in directory name
spec = importlib.util.spec_from_file_location(
    "kalman_filter",
    r"D:\kitti_sensor_fusion\Kalman Filter Implementation\kalman_filter.py"
)
kalman_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kalman_module)
ExtendedKalmanFilter = kalman_module.ExtendedKalmanFilter


class KalmanTracker:
    """
    Multi-object tracker using Extended Kalman Filters.
    
    Uses your ExtendedKalmanFilter for each object and adds:
    - Data association (Hungarian-style matching)
    - Track creation/deletion
    - Multi-object tracking
    """
    
    def __init__(self, q_pos, q_vel, r_camera, r_lidar,
                 gate_threshold, init_frames, max_age, age_threshold):
        """
        Initialize tracker with tuning parameters.
        
        Args:
            q_pos: Process noise for position (0.1-1.0)
            q_vel: Process noise for velocity (0.01-0.1)
            r_camera: Measurement noise for camera (0.2-1.0)
            r_lidar: Measurement noise for LiDAR (0.2-1.0)
            gate_threshold: Data association gate (6.3-11.3)
            init_frames: Frames to confirm track (2-5)
            max_age: Max frames without detection (30-50)
            age_threshold: Frames before activation (2-5)
        """
        # Store tuning parameters
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_camera = r_camera
        self.r_lidar = r_lidar
        self.gate_threshold = gate_threshold
        self.init_frames = init_frames
        self.max_age = max_age
        self.age_threshold = age_threshold
        
        # Build process noise matrix
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        
        # Build measurement noise matrix (average camera and LiDAR)
        r_avg = (r_camera + r_lidar) / 2.0
        self.R = np.diag([r_avg, r_avg, r_avg])
        
        # Tracking state
        self.tracks = {}  # {track_id: track_data}
        self.next_id = 0
        self.frame_count = 0
        self.dt = 0.1  # Assume 100ms between frames
    
    def update(self, frame_id: int, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            frame_id: Current frame ID
            detections: List of detections:
                [
                    {
                        'x': float, 'y': float, 'z': float,
                        'size': [l, w, h],
                        'conf': float
                    },
                    ...
                ]
        
        Returns:
            List of tracked objects:
            [
                {
                    'id': int,
                    'bbox_3d': [x, y, z, l, w, h, rot],
                    'state': {...}
                },
                ...
            ]
        """
        self.frame_count = frame_id
        
        # PREDICT: Advance all existing tracks
        for track_id, track in self.tracks.items():
            track['ekf'].predict()
            track['frames_since_detection'] += 1
        
        # DATA ASSOCIATION: Match detections to tracks
        matched_pairs, unmatched_dets, unmatched_tracks = \
            self._associate_detections(detections)
        
        # UPDATE: Update matched tracks with detections
        for track_id, det_idx in matched_pairs:
            det = detections[det_idx]
            
            # Convert detection to measurement vector
            z = np.array([det['x'], det['y'], det['z']])
            
            # Update EKF with measurement
            self.tracks[track_id]['ekf'].update(z)
            self.tracks[track_id]['frames_since_detection'] = 0
            self.tracks[track_id]['age'] += 1
            self.tracks[track_id]['confidence'] = det['conf']
        
        # CREATE: New tracks from unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            
            # Create new track with initial state
            ekf = ExtendedKalmanFilter(
                dt=self.dt,
                process_noise=self.Q,
                measurement_noise=self.R
            )
            
            # Initialize with detection as initial position, zero velocity
            init_state = np.array([
                det['x'], det['y'], det['z'],
                0.0, 0.0, 0.0  # Zero initial velocity
            ])
            ekf.set_state(init_state)
            
            # Create track
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'ekf': ekf,
                'age': 1,
                'frames_since_detection': 0,
                'confidence': det['conf'],
                'created_frame': frame_id,
            }
        
        # CLEANUP: Remove old/dead tracks
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track['frames_since_detection'] > self.max_age:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        # OUTPUT: Format tracked objects
        tracked_objects = []
        
        for track_id, track in self.tracks.items():
            # Only output confirmed tracks
            if track['age'] >= self.age_threshold:
                state = track['ekf'].get_state()
                pos = state[:3]
                vel = state[3:6]
                
                # Create 3D bbox (assuming fixed size for simplicity)
                bbox_3d = np.concatenate([
                    pos,
                    [3.8, 1.6, 1.5],  # Default car dimensions (l, w, h)
                    [0.0]  # rotation
                ])
                
                tracked_objects.append({
                    'id': track_id,
                    'bbox_3d': bbox_3d.tolist(),
                    'state': {
                        'position': pos.tolist(),
                        'velocity': vel.tolist(),
                        'age': track['age'],
                        'confidence': track['confidence'],
                    }
                })
        
        return tracked_objects
    
    def _associate_detections(self, detections: List[Dict]):
        """
        Associate detections to existing tracks.
        
        Returns:
            (matched_pairs, unmatched_dets, unmatched_tracks)
            - matched_pairs: List of (track_id, det_idx) tuples
            - unmatched_dets: List of detection indices
            - unmatched_tracks: List of track IDs
        """
        if not detections or not self.tracks:
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(self.tracks.keys())
            return [], unmatched_dets, unmatched_tracks
        
        # Build cost matrix: distance from each track to each detection
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        
        cost_matrix = np.full((n_tracks, n_dets), np.inf)
        
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_pos = track['ekf'].get_position()
            
            for j, det in enumerate(detections):
                det_pos = np.array([det['x'], det['y'], det['z']])
                distance = np.linalg.norm(track_pos - det_pos)
                cost_matrix[i, j] = distance
        
        # Simple greedy matching (Hungarian would be better)
        matched_pairs = []
        matched_dets = set()
        matched_tracks = set()
        
        # Greedy: match smallest distances first
        for _ in range(min(n_tracks, n_dets)):
            # Find minimum cost
            i_min, j_min = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            
            if cost_matrix[i_min, j_min] < self.gate_threshold:
                track_id = track_ids[i_min]
                matched_pairs.append((track_id, j_min))
                matched_dets.add(j_min)
                matched_tracks.add(i_min)
                
                # Mark row and column as used
                cost_matrix[i_min, :] = np.inf
                cost_matrix[:, j_min] = np.inf
            else:
                break
        
        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        unmatched_tracks = [track_ids[i] for i in range(n_tracks) if i not in matched_tracks]
        
        return matched_pairs, unmatched_dets, unmatched_tracks


# For testing
if __name__ == '__main__':
    # Quick test
    tracker = KalmanTracker(
        q_pos=0.1, q_vel=0.01, r_camera=0.2, r_lidar=0.2,
        gate_threshold=6.3, init_frames=2, max_age=30, age_threshold=2
    )
    
    # Simulate detections
    detections = [
        {'x': 0.0, 'y': 0.0, 'z': 5.0, 'size': [3.8, 1.6, 1.5], 'conf': 0.9},
        {'x': 1.0, 'y': 1.0, 'z': 6.0, 'size': [3.8, 1.6, 1.5], 'conf': 0.8},
    ]
    
    tracked = tracker.update(0, detections)
    print(f"Tracked {len(tracked)} objects")
    for obj in tracked:
        print(f"  Track {obj['id']}: {obj['state']}")
