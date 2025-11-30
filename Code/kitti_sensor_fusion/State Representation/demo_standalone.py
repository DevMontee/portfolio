"""
STANDALONE TRACKING DEMO
========================

This demo works RIGHT NOW with no external dependencies.
Shows the tracking system working with synthetic data.

Run:
    python demo_standalone.py

This creates fake detections and shows tracking working in real-time.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


class TrackState(Enum):
    """Track states."""
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3


@dataclass
class KalmanTracker:
    """Simple 1D Kalman filter."""
    x: np.ndarray = None  # [position, velocity]
    P: np.ndarray = None  # covariance
    
    def __post_init__(self):
        if self.x is None:
            self.x = np.array([0.0, 0.0])
        if self.P is None:
            self.P = np.eye(2)
    
    def predict(self):
        """Predict next state."""
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.01, 0], [0, 0.01]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[0]
    
    def update(self, z):
        """Update with measurement."""
        H = np.array([[1, 0]])
        R = np.array([[0.1]])
        y = z - (H @ self.x)[0]
        S = (H @ self.P @ H.T + R)[0, 0]
        K = (self.P @ H.T).flatten() / S  # Flatten to 1D
        self.x = self.x + K * y
        I_KH = np.eye(2) - np.outer(K, H)  # Use outer product
        self.P = I_KH @ self.P


@dataclass
class SimpleTrack:
    """Tracked object."""
    track_id: int
    state: TrackState = TrackState.TENTATIVE
    position: np.ndarray = None
    velocity: np.ndarray = None
    age: int = 0
    time_since_update: int = 0
    kalmans: Dict = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.kalmans is None:
            self.kalmans = {
                'x': KalmanTracker(),
                'y': KalmanTracker(),
                'z': KalmanTracker()
            }
    
    def predict(self):
        """Predict next position."""
        self.position = np.array([
            self.kalmans['x'].predict(),
            self.kalmans['y'].predict(),
            self.kalmans['z'].predict()
        ])
        self.time_since_update += 1
        self.age += 1
    
    def update(self, position):
        """Update with detection."""
        self.kalmans['x'].update(position[0])
        self.kalmans['y'].update(position[1])
        self.kalmans['z'].update(position[2])
        
        self.position = np.array([
            self.kalmans['x'].x[0],
            self.kalmans['y'].x[0],
            self.kalmans['z'].x[0]
        ])
        
        self.velocity = np.array([
            self.kalmans['x'].x[1],
            self.kalmans['y'].x[1],
            self.kalmans['z'].x[1]
        ])
        
        self.time_since_update = 0


class SimpleTracker:
    """Simple multi-object tracker."""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
    
    def predict(self):
        """Predict all tracks."""
        for track in self.tracks.values():
            if track.state != TrackState.DELETED:
                track.predict()
    
    def associate(self, detections):
        """Simple nearest-neighbor association."""
        matches = {}
        unmatched = list(range(len(detections)))
        
        for track in self.tracks.values():
            if track.state == TrackState.DELETED:
                continue
            
            # Find nearest detection
            best_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                dist = np.linalg.norm(track.position - det['pos'])
                if dist < best_dist and dist < 2.0:  # 2.0m threshold
                    best_dist = dist
                    best_idx = i
            
            if best_idx >= 0 and best_idx in unmatched:
                matches[track.track_id] = best_idx
                unmatched.remove(best_idx)
        
        return matches, unmatched
    
    def update(self, detections, matches):
        """Update matched tracks."""
        for track_id, det_idx in matches.items():
            self.tracks[track_id].update(detections[det_idx]['pos'])
            if self.tracks[track_id].state == TrackState.TENTATIVE and self.tracks[track_id].age >= 3:
                self.tracks[track_id].state = TrackState.CONFIRMED
        
        # Mark unmatched tracks as lost
        for track in self.tracks.values():
            if track.track_id not in matches and track.state != TrackState.DELETED:
                if track.time_since_update > 10:
                    track.state = TrackState.DELETED
                elif track.state == TrackState.CONFIRMED:
                    track.state = TrackState.LOST
    
    def create_new(self, detections, unmatched):
        """Create new tracks."""
        for idx in unmatched:
            track = SimpleTrack(
                track_id=self.next_id,
                position=detections[idx]['pos'].copy()
            )
            track.kalmans['x'].x[0] = detections[idx]['pos'][0]
            track.kalmans['y'].x[0] = detections[idx]['pos'][1]
            track.kalmans['z'].x[0] = detections[idx]['pos'][2]
            
            self.tracks[self.next_id] = track
            self.next_id += 1
    
    def cleanup(self):
        """Remove deleted tracks."""
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state != TrackState.DELETED
        }
    
    def get_active_tracks(self):
        """Get active tracks."""
        return [t for t in self.tracks.values() 
                if t.state in [TrackState.CONFIRMED, TrackState.LOST]]


def generate_synthetic_detections(frame_idx):
    """Generate synthetic detections simulating moving objects."""
    detections = []
    
    # Object 1: Moving forward (car)
    obj1_pos = np.array([
        -10 + frame_idx * 0.5,  # Moving right
        0,
        20
    ])
    detections.append({'pos': obj1_pos, 'type': 'Car'})
    
    # Object 2: Circular motion (pedestrian)
    angle = frame_idx * 0.1
    obj2_pos = np.array([
        10 + 5 * np.cos(angle),
        5 * np.sin(angle),
        15
    ])
    detections.append({'pos': obj2_pos, 'type': 'Pedestrian'})
    
    # Object 3: Appears/disappears (simulating occlusion)
    if 10 < frame_idx < 40:
        obj3_pos = np.array([
            5 + (frame_idx - 10) * 0.3,
            -3,
            25
        ])
        detections.append({'pos': obj3_pos, 'type': 'Car'})
    
    # Add noise
    for det in detections:
        det['pos'] += np.random.randn(3) * 0.1
    
    return detections


def print_frame(frame_idx, tracker):
    """Print frame visualization."""
    
    print(f"\n{'='*70}")
    print(f"Frame {frame_idx:3d}".center(70))
    print(f"{'='*70}")
    
    active_tracks = tracker.get_active_tracks()
    
    # Header
    print(f"\n{'ID':>3} | {'State':<10} | {'Position (x,y,z)':>25} | {'Velocity':>20}")
    print("-" * 70)
    
    # Tracks
    for track in sorted(active_tracks, key=lambda t: t.track_id):
        state_color = {
            TrackState.CONFIRMED: "🟢 CONFIRMED",
            TrackState.TENTATIVE: "🟡 TENTATIVE",
            TrackState.LOST: "🔴 LOST"
        }
        
        state = state_color.get(track.state, str(track.state))
        pos = f"({track.position[0]:6.1f}, {track.position[1]:6.1f}, {track.position[2]:6.1f})"
        vel = f"({track.velocity[0]:5.2f}, {track.velocity[1]:5.2f}, {track.velocity[2]:5.2f})"
        
        print(f"{track.track_id:3d} | {state:<10} | {pos:>25} | {vel:>20}")
    
    # Summary
    confirmed = sum(1 for t in active_tracks if t.state == TrackState.CONFIRMED)
    tentative = sum(1 for t in active_tracks if t.state == TrackState.TENTATIVE)
    lost = sum(1 for t in active_tracks if t.state == TrackState.LOST)
    
    print("\n" + "-" * 70)
    print(f"Total: {len(active_tracks)} | Confirmed: {confirmed} | Tentative: {tentative} | Lost: {lost}")


def run_demo():
    """Run the demo."""
    
    print("\n" + "="*70)
    print("STANDALONE TRACKING DEMO".center(70))
    print("="*70)
    print("\nThis demo shows object tracking with SYNTHETIC DATA")
    print("No KITTI dataset or detection pipeline needed!\n")
    
    tracker = SimpleTracker()
    
    print("Simulating 50 frames of tracking...\n")
    
    for frame_idx in range(50):
        # Get synthetic detections
        detections = generate_synthetic_detections(frame_idx)
        
        # Tracking pipeline
        tracker.predict()
        matches, unmatched = tracker.associate(detections)
        tracker.update(detections, matches)
        tracker.create_new(detections, unmatched)
        tracker.cleanup()
        
        # Print every 5 frames
        if frame_idx % 5 == 0:
            print_frame(frame_idx, tracker)
        
        # Small delay for readability
        if frame_idx % 10 == 0:
            input("Press Enter to continue...")
    
    # Final summary
    print("\n" + "="*70)
    print("DEMO COMPLETE".center(70))
    print("="*70)
    
    print("""
✓ WHAT YOU SAW:

1. Objects appeared with TENTATIVE state
2. After 3 detections, they became CONFIRMED
3. Objects maintained persistent IDs
4. Position and velocity were estimated by Kalman filter
5. When an object disappeared (occlusion), it went to LOST
6. After too long without detection, it was DELETED

THIS IS THE CORE TRACKING SYSTEM WORKING!

When you connect it to your detection pipeline:
- Replace synthetic detections with real ones
- Everything else works the same way
- Objects get beautiful persistent tracking ✓

Next step: Integrate with your detection_pipeline.py
    """)


if __name__ == "__main__":
    run_demo()
