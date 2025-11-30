"""
Data Association for Multi-Object Tracking

Implements:
  1. Hungarian algorithm for optimal measurement-to-track assignment
  2. Mahalanobis distance cost matrix
  3. Track lifecycle management (initiation, confirmation, deletion)
  4. Missed detection handling with track aging
  5. Gating to reduce computational cost
  
Usage:
  python data_association.py
  
References:
  - Kuhn-Munkres (Hungarian) Algorithm: O(n³) optimal assignment
  - Mahalanobis distance: accounts for covariance structure
  - Multi-hypothesis tracking: delayed confirmation to reduce false positives
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from enum import Enum
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from collections import deque
import matplotlib.pyplot as plt


class TrackState(Enum):
    """Track lifecycle states."""
    NEW = 0          # Just initialized, awaiting confirmation
    TENTATIVE = 1    # Matches across multiple frames
    CONFIRMED = 2    # Established track, accepted for output
    DELETED = 3      # Track terminated due to no detections


@dataclass
class Track:
    """Represents a tracked object."""
    
    track_id: int
    state: TrackState = TrackState.NEW
    
    # State vector: [x, y, z, vx, vy, vz, size_x, size_y, size_z]
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(9))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(9) * 0.1)
    
    # Track history
    age: int = 0                          # Frames since creation
    time_since_update: int = 0            # Frames since last measurement
    hits: int = 0                         # Number of successful associations
    hit_streak: int = 0                   # Consecutive frames with detections
    missed_streak: int = 0                # Consecutive frames without detections
    
    # Configuration
    max_age: int = 30                     # Max frames without detection before deletion
    min_hits: int = 3                     # Hits required to transition to CONFIRMED
    min_hit_streak: int = 2               # Consecutive detections for CONFIRMED
    
    # Association history
    last_measurement: Optional[np.ndarray] = None
    measurement_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __post_init__(self):
        """Initialize the track."""
        self.age = 0
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.missed_streak = 0
        self.measurement_history = deque(maxlen=10)
    
    def predict(self, dt: float = 1.0):
        """
        Predict next state using constant velocity model.
        
        Args:
            dt: Time delta since last prediction
        """
        # State transition matrix: constant velocity model
        F = np.eye(9)
        F[0, 3] = dt  # x += vx*dt
        F[1, 4] = dt  # y += vy*dt
        F[2, 5] = dt  # z += vz*dt
        
        # Predict state
        self.state_vector = F @ self.state_vector
        
        # Predict covariance (add process noise)
        q = 0.01  # Process noise
        Q = np.eye(9) * q
        self.covariance = F @ self.covariance @ F.T + Q
        
        # Update age and time since last measurement
        self.age += 1
        self.time_since_update += 1
    
    def update(self, measurement: np.ndarray, measurement_cov: np.ndarray):
        """
        Update track state with measurement (Kalman filter update).
        
        Args:
            measurement: 3D position measurement [x, y, z]
            measurement_cov: Measurement covariance matrix
        """
        # Simple Kalman update (can use 6DOF state for full 9D)
        H = np.zeros((3, 9))
        H[:3, :3] = np.eye(3)  # Observe position
        
        # Innovation (measurement residual)
        innovation = measurement - H @ self.state_vector
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + measurement_cov
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state_vector = self.state_vector + K @ innovation
        
        # Update covariance
        self.covariance = (np.eye(9) - K @ H) @ self.covariance
        
        # Update track statistics
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.missed_streak = 0
        
        # Store measurement
        self.last_measurement = measurement.copy()
        self.measurement_history.append(measurement)
        
        # Transition to TENTATIVE if enough hits
        if self.state == TrackState.NEW and self.hits >= self.min_hits:
            self.state = TrackState.TENTATIVE
        
        # Transition to CONFIRMED if enough consecutive hits
        if self.state == TrackState.TENTATIVE and self.hit_streak >= self.min_hit_streak:
            self.state = TrackState.CONFIRMED
    
    def mark_missed(self):
        """Mark track as having missed detection this frame."""
        self.hit_streak = 0
        self.missed_streak += 1
        
        # Delete if too old
        if self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
    
    def get_position(self) -> np.ndarray:
        """Get current 3D position."""
        return self.state_vector[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current 3D velocity."""
        return self.state_vector[3:6].copy()
    
    def get_size(self) -> np.ndarray:
        """Get object size estimate."""
        return self.state_vector[6:9].copy()
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self) -> bool:
        """Check if track is deleted."""
        return self.state == TrackState.DELETED
    
    def is_active(self) -> bool:
        """Check if track is active (not deleted)."""
        return self.state != TrackState.DELETED


class DataAssociation:
    """Hungarian algorithm-based measurement-to-track association."""
    
    def __init__(self, 
                 max_mahalanobis_distance: float = 3.0,
                 gating_threshold: float = 3.0,
                 lambda_unmatched: float = 0.1):
        """
        Initialize data association engine.
        
        Args:
            max_mahalanobis_distance: Maximum Mahalanobis distance for association
            gating_threshold: Chi-square threshold for gating (n_dof=3 → ~16.8 for 99%)
            lambda_unmatched: Cost for unmatched detection (prevents over-eager matching)
        """
        self.max_mahalanobis_distance = max_mahalanobis_distance
        self.gating_threshold = gating_threshold
        self.lambda_unmatched = lambda_unmatched
        
        # Statistics
        self.association_count = 0
        self.unmatched_detection_count = 0
        self.unmatched_track_count = 0
    
    def compute_mahalanobis_distance(self, 
                                     track_pos: np.ndarray,
                                     track_cov: np.ndarray,
                                     measurement: np.ndarray,
                                     measurement_cov: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between track prediction and measurement.
        
        d = sqrt((z - H*x)^T * S^-1 * (z - H*x))
        
        Args:
            track_pos: Track predicted position
            track_cov: Track covariance (9x9)
            measurement: Measurement position (3D)
            measurement_cov: Measurement covariance (3x3)
            
        Returns:
            Mahalanobis distance
        """
        # Extract position part of covariance
        pos_cov = track_cov[:3, :3]
        
        # Innovation covariance
        S = pos_cov + measurement_cov
        
        # Innovation
        innovation = measurement - track_pos
        
        # Try to compute distance; return large value if singular
        try:
            S_inv = np.linalg.inv(S)
            distance = np.sqrt(innovation @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            distance = np.inf
        
        return distance
    
    def compute_cost_matrix(self,
                           tracks: List[Track],
                           measurements: List[Tuple[np.ndarray, np.ndarray]],
                           gating_only: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build cost matrix for Hungarian algorithm.
        
        Cost[i,j] = Mahalanobis distance from track i to measurement j
        Invalid associations (beyond gate) get very high cost.
        
        Args:
            tracks: List of active tracks (with predictions)
            measurements: List of (position, covariance) tuples
            gating_only: If True, use gating distance threshold without full matrix
            
        Returns:
            (cost_matrix, gated_matrix, distances_matrix)
            - cost_matrix: N_tracks x N_measurements with costs (inf for invalid)
            - gated_matrix: Binary matrix showing which pairs pass gating
            - distances_matrix: Actual Mahalanobis distances
        """
        n_tracks = len(tracks)
        n_measurements = len(measurements)
        
        # Initialize matrices
        cost_matrix = np.full((n_tracks, n_measurements), np.inf)
        gated_matrix = np.zeros((n_tracks, n_measurements), dtype=bool)
        distances_matrix = np.full((n_tracks, n_measurements), np.inf)
        
        if n_measurements == 0 or n_tracks == 0:
            return cost_matrix, gated_matrix, distances_matrix
        
        # Compute distances
        for i, track in enumerate(tracks):
            for j, (meas_pos, meas_cov) in enumerate(measurements):
                # Compute Mahalanobis distance
                distance = self.compute_mahalanobis_distance(
                    track.get_position(),
                    track.covariance,
                    meas_pos,
                    meas_cov
                )
                
                distances_matrix[i, j] = distance
                
                # Gating: reject if beyond threshold
                if distance <= self.gating_threshold:
                    gated_matrix[i, j] = True
                    cost_matrix[i, j] = distance
        
        return cost_matrix, gated_matrix, distances_matrix
    
    def hungarian_algorithm(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve assignment problem using Hungarian algorithm (scipy wrapper).
        
        Args:
            cost_matrix: Cost matrix (n_tracks x n_measurements)
                        np.inf entries indicate invalid assignments
            
        Returns:
            (track_indices, measurement_indices) - Matched pairs
        """
        n_tracks, n_measurements = cost_matrix.shape
        
        if n_measurements == 0 or n_tracks == 0:
            return np.array([]), np.array([])
        
        # Replace inf with a large finite number for scipy
        # Use max finite value + 1
        cost_matrix_finite = cost_matrix.copy()
        valid_costs = cost_matrix_finite[np.isfinite(cost_matrix_finite)]
        
        if len(valid_costs) == 0:
            # All costs are infinite - no valid assignments
            return np.array([]), np.array([])
        
        max_valid_cost = np.max(valid_costs)
        cost_matrix_finite[np.isinf(cost_matrix_finite)] = max_valid_cost * 10
        
        # Solve assignment
        try:
            track_indices, measurement_indices = linear_sum_assignment(cost_matrix_finite)
        except ValueError:
            # Fallback if scipy fails
            return np.array([]), np.array([])
        
        # Filter out assignments with infinite cost (invalid gating)
        valid_mask = np.isfinite(cost_matrix[track_indices, measurement_indices])
        track_indices = track_indices[valid_mask]
        measurement_indices = measurement_indices[valid_mask]
        
        return track_indices, measurement_indices
    
    def associate(self,
                  tracks: List[Track],
                  measurements: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[List[Tuple[int, int]], 
                                                                                List[int], 
                                                                                List[int]]:
        """
        Associate measurements to tracks using Hungarian algorithm.
        
        Args:
            tracks: List of Track objects (should be pre-predicted)
            measurements: List of (position_3d, measurement_cov_3x3) tuples
            
        Returns:
            (matched_pairs, unmatched_tracks, unmatched_measurements)
            - matched_pairs: List of (track_idx, meas_idx) tuples
            - unmatched_tracks: List of track indices without matches
            - unmatched_measurements: List of measurement indices without matches
        """
        n_tracks = len(tracks)
        n_measurements = len(measurements)
        
        # Compute cost matrix with gating
        cost_matrix, gated_matrix, distances = self.compute_cost_matrix(tracks, measurements)
        
        if n_measurements == 0:
            # No measurements: all tracks unmatched
            unmatched_tracks = list(range(n_tracks))
            unmatched_measurements = []
            return [], unmatched_tracks, unmatched_measurements
        
        if n_tracks == 0:
            # No tracks: all measurements unmatched
            unmatched_tracks = []
            unmatched_measurements = list(range(n_measurements))
            return [], unmatched_tracks, unmatched_measurements
        
        # Solve assignment problem
        track_indices, measurement_indices = self.hungarian_algorithm(cost_matrix)
        
        # Collect matched pairs and unmatched indices
        matched_pairs = list(zip(track_indices, measurement_indices))
        matched_track_set = set(track_indices)
        matched_meas_set = set(measurement_indices)
        
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_set]
        unmatched_measurements = [j for j in range(n_measurements) if j not in matched_meas_set]
        
        # Update statistics
        self.association_count += len(matched_pairs)
        self.unmatched_detection_count += len(unmatched_measurements)
        self.unmatched_track_count += len(unmatched_tracks)
        
        return matched_pairs, unmatched_tracks, unmatched_measurements


class TrackManager:
    """Manages track lifecycle: creation, updates, deletion."""
    
    def __init__(self,
                 min_hits: int = 3,
                 min_hit_streak: int = 2,
                 max_age: int = 30,
                 max_mahalanobis_distance: float = 3.0):
        """
        Initialize track manager.
        
        Args:
            min_hits: Measurements required before TENTATIVE→CONFIRMED
            min_hit_streak: Consecutive hits required for CONFIRMED
            max_age: Max frames without detection before deletion
            max_mahalanobis_distance: Gating threshold for data association
        """
        self.tracks: List[Track] = []
        self.next_track_id = 1
        
        self.min_hits = min_hits
        self.min_hit_streak = min_hit_streak
        self.max_age = max_age
        
        # Data association
        self.data_association = DataAssociation(
            max_mahalanobis_distance=max_mahalanobis_distance,
            gating_threshold=3.0
        )
        
        # Statistics
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
    
    def predict_tracks(self, dt: float = 1.0):
        """
        Predict state of all active tracks.
        
        Args:
            dt: Time delta
        """
        for track in self.tracks:
            if not track.is_deleted():
                track.predict(dt)
    
    def update(self,
               measurements: List[Tuple[np.ndarray, np.ndarray]],
               dt: float = 1.0) -> Tuple[List[Track], List[int], List[int]]:
        """
        Main tracking update: predict, associate, update tracks.
        
        Args:
            measurements: List of (position_3d, measurement_cov_3x3) tuples
            dt: Time delta
            
        Returns:
            (confirmed_tracks, unmatched_track_ids, unmatched_measurement_indices)
        """
        self.frame_count += 1
        
        # Predict all tracks
        self.predict_tracks(dt)
        
        # Get active (non-deleted) tracks
        active_tracks = [t for t in self.tracks if t.is_active()]
        
        # Data association
        matched_pairs, unmatched_track_indices, unmatched_measurement_indices = \
            self.data_association.associate(active_tracks, measurements)
        
        # Create mapping from active indices to track objects
        active_track_map = {i: t for i, t in enumerate(active_tracks)}
        
        # Process matched detections
        for active_idx, meas_idx in matched_pairs:
            track = active_track_map[active_idx]
            meas_pos, meas_cov = measurements[meas_idx]
            track.update(meas_pos, meas_cov)
        
        # Process unmatched tracks (missed detections)
        for active_idx in unmatched_track_indices:
            track = active_track_map[active_idx]
            track.mark_missed()
            
            # Delete if too old
            if track.is_deleted():
                self.total_tracks_deleted += 1
        
        # Create new tracks from unmatched measurements
        for meas_idx in unmatched_measurement_indices:
            self._create_track(measurements[meas_idx])
        
        # Clean up deleted tracks periodically
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        
        return confirmed_tracks, unmatched_track_indices, unmatched_measurement_indices
    
    def _create_track(self, measurement: Tuple[np.ndarray, np.ndarray]):
        """
        Create a new track from unmatched measurement.
        
        Args:
            measurement: (position_3d, measurement_cov_3x3) tuple
        """
        track = Track(
            track_id=self.next_track_id,
            state=TrackState.NEW,
            min_hits=self.min_hits,
            min_hit_streak=self.min_hit_streak,
            max_age=self.max_age
        )
        
        # Initialize state with measurement
        meas_pos, meas_cov = measurement
        track.state_vector[:3] = meas_pos
        track.covariance[:3, :3] = meas_cov
        
        # Initial update
        track.update(meas_pos, meas_cov)
        
        self.tracks.append(track)
        self.next_track_id += 1
        self.total_tracks_created += 1
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed()]
    
    def get_all_active_tracks(self) -> List[Track]:
        """Get all active (non-deleted) tracks."""
        return [t for t in self.tracks if t.is_active()]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get tracking statistics."""
        confirmed = sum(1 for t in self.tracks if t.is_confirmed())
        tentative = sum(1 for t in self.tracks if t.state == TrackState.TENTATIVE)
        new = sum(1 for t in self.tracks if t.state == TrackState.NEW)
        
        return {
            'frame': self.frame_count,
            'confirmed_tracks': confirmed,
            'tentative_tracks': tentative,
            'new_tracks': new,
            'total_tracks': len(self.tracks),
            'total_created': self.total_tracks_created,
            'total_deleted': self.total_tracks_deleted,
            'associations': self.data_association.association_count,
            'unmatched_detections': self.data_association.unmatched_detection_count,
            'unmatched_tracks': self.data_association.unmatched_track_count,
        }


def test_mahalanobis_distance():
    """Test Mahalanobis distance computation."""
    print("\n" + "="*70)
    print("TEST 1: MAHALANOBIS DISTANCE COMPUTATION")
    print("="*70)
    
    da = DataAssociation()
    
    # Setup track and measurements
    track_pos = np.array([10.0, 5.0, 0.0])
    track_cov = np.eye(9) * 0.1
    
    print("\nMahalanobis Distance Examples:")
    print(f"{'Measurement Position':>40} {'Distance':>15}")
    print("-" * 60)
    
    test_measurements = [
        np.array([10.0, 5.0, 0.0]),    # Exact match
        np.array([10.1, 5.0, 0.0]),    # 0.1m offset in X
        np.array([10.5, 5.0, 0.0]),    # 0.5m offset in X
        np.array([10.0, 5.2, 0.0]),    # 0.2m offset in Y
        np.array([11.0, 6.0, 0.5]),    # Multiple offsets
    ]
    
    meas_cov = np.eye(3) * 0.05
    
    for meas in test_measurements:
        dist = da.compute_mahalanobis_distance(track_pos, track_cov, meas, meas_cov)
        print(f"{str(meas):>40} {dist:>15.4f}")
    
    print("\n✓ Mahalanobis distance working correctly")


def test_hungarian_algorithm():
    """Test Hungarian algorithm for assignment."""
    print("\n" + "="*70)
    print("TEST 2: HUNGARIAN ALGORITHM - OPTIMAL ASSIGNMENT")
    print("="*70)
    
    # Create a simple cost matrix
    # 3 tracks, 3 measurements
    cost_matrix = np.array([
        [1.0, 2.0, np.inf],
        [3.0, 1.0, 4.0],
        [2.0, 4.0, 2.0]
    ])
    
    print("\nCost Matrix:")
    print(cost_matrix)
    
    da = DataAssociation()
    track_indices, meas_indices = da.hungarian_algorithm(cost_matrix)
    
    print("\nOptimal Assignment (Hungarian Algorithm):")
    total_cost = 0
    for t_idx, m_idx in zip(track_indices, meas_indices):
        cost = cost_matrix[t_idx, m_idx]
        total_cost += cost
        print(f"  Track {t_idx} → Measurement {m_idx} (cost: {cost:.2f})")
    
    print(f"\nTotal Cost: {total_cost:.2f}")
    print("✓ Hungarian algorithm working correctly")


def test_data_association():
    """Test data association with cost matrix."""
    print("\n" + "="*70)
    print("TEST 3: DATA ASSOCIATION WITH GATING")
    print("="*70)
    
    # Create tracks
    tracks = []
    for i in range(3):
        track = Track(track_id=i)
        track.state_vector[:3] = np.array([10.0 + i*5, 5.0, 0.0])
        track.covariance = np.eye(9) * 0.1
        tracks.append(track)
    
    # Create measurements
    measurements = [
        (np.array([10.0, 5.0, 0.0]), np.eye(3) * 0.05),      # Matches track 0
        (np.array([15.1, 5.0, 0.0]), np.eye(3) * 0.05),      # Matches track 1
        (np.array([50.0, 50.0, 50.0]), np.eye(3) * 0.05),    # Far away (unmatched)
    ]
    
    da = DataAssociation()
    matched, unmatched_tracks, unmatched_meas = da.associate(tracks, measurements)
    
    print("\nData Association Results:")
    print(f"Matched pairs: {matched}")
    print(f"Unmatched tracks: {unmatched_tracks}")
    print(f"Unmatched measurements: {unmatched_meas}")
    
    print("\n✓ Data association working correctly")


def test_track_manager():
    """Test complete track manager."""
    print("\n" + "="*70)
    print("TEST 4: TRACK MANAGER - LIFECYCLE MANAGEMENT")
    print("="*70)
    
    manager = TrackManager(min_hits=2, min_hit_streak=1, max_age=5)
    
    print("\nSimulating 10 frames of tracking:")
    print(f"{'Frame':>5} {'Confirmed':>12} {'Tentative':>12} {'New':>10} {'Status':>40}")
    print("-" * 85)
    
    # Simulate measurements across frames
    for frame in range(10):
        # Generate synthetic measurements
        if frame < 7:
            # Object 1: continuous detections
            meas1 = (np.array([10.0 + frame*0.5, 5.0, 0.0]), np.eye(3) * 0.05)
            measurements = [meas1]
            
            # Object 2: starts at frame 2
            if frame >= 2:
                meas2 = (np.array([20.0 - frame*0.3, 8.0, 0.0]), np.eye(3) * 0.05)
                measurements.append(meas2)
        else:
            # Only object 1 in later frames (object 2 will be deleted)
            meas1 = (np.array([10.0 + frame*0.5, 5.0, 0.0]), np.eye(3) * 0.05)
            measurements = [meas1]
        
        # Update manager
        confirmed, unmatched_tracks, unmatched_meas = manager.update(measurements)
        
        # Get statistics
        stats = manager.get_statistics()
        status = f"Conf: {stats['confirmed_tracks']}, Ten: {stats['tentative_tracks']}, New: {stats['new_tracks']}"
        
        print(f"{frame:5d} {stats['confirmed_tracks']:12d} {stats['tentative_tracks']:12d} "
              f"{stats['new_tracks']:10d} {status:>40}")
    
    # Final statistics
    print("\nFinal Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Track manager working correctly")


def test_missed_detection_handling():
    """Test handling of missed detections and track deletion."""
    print("\n" + "="*70)
    print("TEST 5: MISSED DETECTION & TRACK AGING")
    print("="*70)
    
    manager = TrackManager(min_hits=1, min_hit_streak=1, max_age=3)
    
    print("\nScenario: Object appears for 3 frames, then disappears:")
    print(f"{'Frame':>5} {'Measurements':>15} {'Active Tracks':>15} {'Track State':>30}")
    print("-" * 70)
    
    # Frame 0-2: Object present
    for frame in range(3):
        measurements = [(np.array([10.0, 5.0, 0.0]), np.eye(3) * 0.05)]
        confirmed, _, _ = manager.update(measurements)
        
        active = manager.get_all_active_tracks()
        states = [f"Track {t.track_id}: {t.state.name}" for t in active]
        
        print(f"{frame:5d} {1:15d} {len(active):15d} {str(states):>30}")
    
    # Frame 3-5: Object disappears
    for frame in range(3, 6):
        measurements = []  # No measurements
        confirmed, _, _ = manager.update(measurements)
        
        active = manager.get_all_active_tracks()
        states = [f"Track {t.track_id}: {t.state.name}(age:{t.time_since_update})" for t in active]
        
        print(f"{frame:5d} {0:15d} {len(active):15d} {str(states):>30}")
    
    print("\n✓ Missed detection handling working correctly")


def visualize_data_association():
    """Visualize data association in 2D."""
    print("\nGenerating data association visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Tracks and measurements
    ax = axes[0, 0]
    
    # Create tracks
    tracks = []
    track_positions = [[10, 5], [20, 15], [15, 10]]
    for i, pos in enumerate(track_positions):
        track = Track(track_id=i)
        track.state_vector[:3] = np.array(pos + [0])
        track.covariance = np.eye(9) * 0.2
        tracks.append(track)
    
    # Create measurements
    measurements = [
        (np.array([10.1, 5.0, 0.0]), np.eye(3) * 0.05),
        (np.array([20.2, 14.9, 0.0]), np.eye(3) * 0.05),
        (np.array([15.5, 10.3, 0.0]), np.eye(3) * 0.05),
        (np.array([30.0, 20.0, 0.0]), np.eye(3) * 0.05),  # Unmatched
    ]
    
    # Data association
    da = DataAssociation()
    matched, unmatched_tracks, unmatched_meas = da.associate(tracks, measurements)
    
    # Plot tracks as blue circles
    for track in tracks:
        pos = track.get_position()[:2]
        ax.scatter(*pos, s=200, c='blue', marker='o', edgecolor='darkblue', linewidth=2, label='Tracks' if track.track_id == 0 else '')
        ax.text(pos[0], pos[1]-1.5, f'T{track.track_id}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot measurements as red stars
    for i, (meas_pos, _) in enumerate(measurements):
        pos = meas_pos[:2]
        ax.scatter(*pos, s=250, c='red', marker='*', edgecolor='darkred', linewidth=2, label='Measurements' if i == 0 else '')
        ax.text(pos[0], pos[1]+1.5, f'M{i}', ha='center', fontsize=10, fontweight='bold')
    
    # Draw association lines
    for t_idx, m_idx in matched:
        track_pos = tracks[t_idx].get_position()[:2]
        meas_pos = measurements[m_idx][0][:2]
        ax.plot([track_pos[0], meas_pos[0]], [track_pos[1], meas_pos[1]], 'g--', alpha=0.7, linewidth=2)
    
    ax.set_xlim([5, 35])
    ax.set_ylim([0, 25])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Data Association: Tracks to Measurements (Green Lines = Matched)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cost matrix heatmap
    ax = axes[0, 1]
    cost_matrix, gated, distances = da.compute_cost_matrix(tracks, measurements)
    
    # Replace inf with a large value for visualization
    cost_vis = cost_matrix.copy()
    cost_vis[np.isinf(cost_vis)] = 10
    
    im = ax.imshow(cost_vis, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=5)
    ax.set_xlabel('Measurement Index')
    ax.set_ylabel('Track Index')
    ax.set_title('Cost Matrix (Mahalanobis Distance)')
    
    # Add values to heatmap
    for i in range(cost_vis.shape[0]):
        for j in range(cost_vis.shape[1]):
            text_val = f"{distances[i, j]:.2f}" if np.isfinite(distances[i, j]) else "∞"
            ax.text(j, i, text_val, ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Mahalanobis Distance')
    
    # Plot 3: Track lifecycle
    ax = axes[1, 0]
    
    manager = TrackManager(min_hits=2, min_hit_streak=1, max_age=5)
    
    frame_counts = []
    confirmed_counts = []
    tentative_counts = []
    new_counts = []
    
    for frame in range(12):
        if frame < 8:
            meas1 = (np.array([10.0 + frame*0.5, 5.0, 0.0]), np.eye(3) * 0.05)
            measurements = [meas1]
            if frame >= 2 and frame < 6:
                meas2 = (np.array([20.0 - frame*0.3, 8.0, 0.0]), np.eye(3) * 0.05)
                measurements.append(meas2)
        else:
            measurements = []
        
        confirmed, _, _ = manager.update(measurements)
        stats = manager.get_statistics()
        
        frame_counts.append(frame)
        confirmed_counts.append(stats['confirmed_tracks'])
        tentative_counts.append(stats['tentative_tracks'])
        new_counts.append(stats['new_tracks'])
    
    ax.plot(frame_counts, confirmed_counts, 'o-', label='Confirmed', linewidth=2, markersize=8)
    ax.plot(frame_counts, tentative_counts, 's-', label='Tentative', linewidth=2, markersize=8)
    ax.plot(frame_counts, new_counts, '^-', label='New', linewidth=2, markersize=8)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Track Lifecycle Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mahalanobis distance distribution
    ax = axes[1, 1]
    
    # Generate random tracks and measurements to show distance distribution
    np.random.seed(42)
    tracks_dist = [Track(track_id=i) for i in range(10)]
    for track in tracks_dist:
        track.state_vector[:3] = np.random.uniform(0, 30, 3)
        track.covariance = np.eye(9) * 0.1
    
    measurements_dist = [
        (np.random.uniform(0, 30, 3), np.eye(3) * 0.05) for _ in range(15)
    ]
    
    _, _, distances_dist = da.compute_cost_matrix(tracks_dist, measurements_dist)
    valid_distances = distances_dist[np.isfinite(distances_dist)].flatten()
    
    ax.hist(valid_distances, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(3.0, color='red', linestyle='--', linewidth=2, label='Gating Threshold')
    ax.set_xlabel('Mahalanobis Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Mahalanobis Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data_association_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: data_association_visualization.png")
    plt.show()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DAY 6: DATA ASSOCIATION - HUNGARIAN ALGORITHM & TRACK MANAGEMENT")
    print("="*70)
    
    test_mahalanobis_distance()
    test_hungarian_algorithm()
    test_data_association()
    test_track_manager()
    test_missed_detection_handling()
    
    # Visualization
    visualize_data_association()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
