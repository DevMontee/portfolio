"""
REAL DATA TRACKING
==================

Connect the tracking system to your detection pipeline + KITTI dataset.

Run:
    python track_real_data.py
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import sys

# Add paths for importing from other days
sys.path.insert(0, r"D:\kitti_sensor_fusion\Environment Setup & Data Pipeline")  # For kitti_dataloader.py
sys.path.insert(0, r"D:\kitti_sensor_fusion\Detection Pipeline")  # For detection_pipeline.py


class TrackState(Enum):
    """Track states."""
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3


@dataclass
class KalmanTracker:
    """Simple 1D Kalman filter."""
    x: np.ndarray = None
    P: np.ndarray = None
    
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
        K = (self.P @ H.T).flatten() / S
        self.x = self.x + K * y
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ self.P


@dataclass
class SimpleTrack:
    """Tracked object."""
    track_id: int
    state: TrackState = TrackState.TENTATIVE
    position: np.ndarray = None
    velocity: np.ndarray = None
    object_type: str = "Unknown"
    confidence: float = 0.0
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
    
    def update(self, position, obj_type="Unknown", confidence=0.9):
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
        
        self.object_type = obj_type
        self.confidence = confidence
        self.time_since_update = 0


class SimpleTracker:
    """Simple multi-object tracker."""
    
    def __init__(self, max_age=30, min_hits=3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_hits = min_hits
    
    def predict(self):
        """Predict all tracks."""
        for track in self.tracks.values():
            if track.state != TrackState.DELETED:
                track.predict()
    
    def associate(self, detections, distance_threshold=5.0):
        """Simple nearest-neighbor association."""
        matches = {}
        unmatched = list(range(len(detections)))
        
        for track in self.tracks.values():
            if track.state == TrackState.DELETED:
                continue
            
            best_dist = float('inf')
            best_idx = -1
            
            for i, det in enumerate(detections):
                dist = np.linalg.norm(track.position - det['location_3d'])
                if dist < best_dist and dist < distance_threshold:
                    best_dist = dist
                    best_idx = i
            
            if best_idx >= 0 and best_idx in unmatched:
                matches[track.track_id] = best_idx
                unmatched.remove(best_idx)
        
        return matches, unmatched
    
    def update(self, detections, matches):
        """Update matched tracks."""
        for track_id, det_idx in matches.items():
            det = detections[det_idx]
            self.tracks[track_id].update(
                det['location_3d'],
                det.get('type', 'Unknown'),
                det.get('confidence', 0.9)
            )
            if self.tracks[track_id].state == TrackState.TENTATIVE and self.tracks[track_id].age >= self.min_hits:
                self.tracks[track_id].state = TrackState.CONFIRMED
        
        # Mark unmatched tracks as lost
        for track in self.tracks.values():
            if track.track_id not in matches and track.state != TrackState.DELETED:
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
                elif track.state == TrackState.CONFIRMED:
                    track.state = TrackState.LOST
    
    def create_new(self, detections, unmatched):
        """Create new tracks."""
        for idx in unmatched:
            det = detections[idx]
            track = SimpleTrack(
                track_id=self.next_id,
                position=det['location_3d'].copy(),
                object_type=det.get('type', 'Unknown'),
                confidence=det.get('confidence', 0.9)
            )
            track.kalmans['x'].x[0] = det['location_3d'][0]
            track.kalmans['y'].x[0] = det['location_3d'][1]
            track.kalmans['z'].x[0] = det['location_3d'][2]
            
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


def run_real_data_tracking():
    """Run tracking on real KITTI data."""
    
    print("\n" + "="*70)
    print("REAL DATA TRACKING - CONNECTING DETECTION PIPELINE".center(70))
    print("="*70 + "\n")
    
    # Check if detection pipeline exists
    try:
        from detection_pipeline import DetectionPipeline
        from kitti_dataloader import KITTIDataLoader
        print("✓ Detection pipeline modules found\n")
    except ImportError as e:
        print(f"❌ Error: Could not import detection pipeline")
        print(f"   {e}")
        print("\nMake sure these files exist in the same directory:")
        print("  - detection_pipeline.py")
        print("  - kitti_dataloader.py")
        return
    
    # Configuration
    KITTI_ROOT = r"D:\KITTI Dataset"
    SEQUENCE = "0000"
    START_FRAME = 0
    NUM_FRAMES = 50  # Process first 50 frames
    
    # Check KITTI dataset exists
    if not Path(KITTI_ROOT).exists():
        print(f"❌ KITTI dataset not found at: {KITTI_ROOT}")
        print(f"\nUpdate KITTI_ROOT in this script to your dataset location")
        return
    
    print(f"Loading KITTI dataset from: {KITTI_ROOT}")
    print(f"Sequence: {SEQUENCE}\n")
    
    # Initialize
    try:
        loader = KITTIDataLoader(KITTI_ROOT, sequence=SEQUENCE, split="training")
        detection_pipeline = DetectionPipeline(KITTI_ROOT, sequence=SEQUENCE, split="training")
        tracker = SimpleTracker(max_age=30, min_hits=3)
        
        print(f"✓ Loaded {len(loader)} frames")
        print(f"✓ Tracker initialized\n")
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Processing {NUM_FRAMES} frames with tracking...\n")
    print(f"{'Frame':>6} | {'Detections':>10} | {'Tracks':>6} | {'Confirmed':>9} | {'Description'}")
    print("-" * 80)
    
    tracking_results = []
    
    for frame_idx in range(START_FRAME, min(START_FRAME + NUM_FRAMES, len(loader))):
        try:
            # Get frame
            frame = loader[frame_idx]
            
            # Get detections from your detection pipeline
            detection_results = detection_pipeline.process_frame(frame_idx)
            
            # Extract detections in the format our tracker expects
            detections = []
            if 'labels' in frame:
                for label in frame['labels']:
                    # Labels are dictionaries with keys: type, location_3d, etc.
                    obj_type = label.get('type', 'Unknown')
                    location = label.get('location_3d', np.array([-10, -1, -1]))
                    
                    # Filter out 'DontCare' objects and invalid locations
                    if obj_type == 'DontCare' or (location[0] == -10 and location[1] == -1):
                        continue
                    
                    # Convert to detection format
                    detection = {
                        'location_3d': np.array(location, dtype=np.float32),
                        'type': obj_type,
                        'confidence': 0.9
                    }
                    detections.append(detection)
            
            # Tracking pipeline
            tracker.predict()
            matches, unmatched = tracker.associate(detections, distance_threshold=5.0)
            tracker.update(detections, matches)
            tracker.create_new(detections, unmatched)
            tracker.cleanup()
            
            # Get active tracks
            active_tracks = tracker.get_active_tracks()
            confirmed_tracks = [t for t in active_tracks if t.state == TrackState.CONFIRMED]
            
            # Description
            desc = "Initial setup" if frame_idx < 3 else ""
            if frame_idx == 10:
                desc = "Objects confirmed"
            elif frame_idx == 30:
                desc = "Stable tracking"
            
            print(f"{frame_idx:6d} | {len(detections):10d} | {len(active_tracks):6d} | "
                  f"{len(confirmed_tracks):9d} | {desc}")
            
            tracking_results.append({
                'frame_idx': frame_idx,
                'detections': detections,
                'tracks': active_tracks,
                'num_detections': len(detections),
                'num_tracks': len(active_tracks),
                'num_confirmed': len(confirmed_tracks)
            })
        
        except Exception as e:
            print(f"{frame_idx:6d} | Error: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("TRACKING SUMMARY".center(70))
    print("="*70 + "\n")
    
    if tracking_results:
        num_detections = [r['num_detections'] for r in tracking_results]
        num_tracks = [r['num_tracks'] for r in tracking_results]
        num_confirmed = [r['num_confirmed'] for r in tracking_results]
        
        print(f"Frames processed: {len(tracking_results)}")
        print(f"\nDetections:")
        print(f"  Total: {sum(num_detections)}")
        print(f"  Avg per frame: {np.mean(num_detections):.1f}")
        print(f"  Max per frame: {max(num_detections)}")
        
        print(f"\nTracks:")
        print(f"  Max concurrent: {max(num_tracks)}")
        print(f"  Avg per frame: {np.mean(num_tracks):.1f}")
        
        print(f"\nConfirmed tracks:")
        print(f"  Total confirmed: {max(num_confirmed)}")
        print(f"  Avg per frame: {np.mean(num_confirmed):.1f}")
        
        # Show some tracks
        last_result = tracking_results[-1]
        if last_result['tracks']:
            print(f"\n\nActive tracks at frame {last_result['frame_idx']}:")
            print(f"\n{'ID':>3} | {'Type':<12} | {'State':<10} | {'Position (x,y,z)':>25} | {'Velocity':>20}")
            print("-" * 85)
            
            for track in sorted(last_result['tracks'], key=lambda t: t.track_id):
                state_sym = {
                    TrackState.CONFIRMED: "🟢",
                    TrackState.TENTATIVE: "🟡",
                    TrackState.LOST: "🔴"
                }.get(track.state, "⚪")
                
                pos = f"({track.position[0]:6.1f}, {track.position[1]:6.1f}, {track.position[2]:6.1f})"
                vel = f"({track.velocity[0]:5.2f}, {track.velocity[1]:5.2f}, {track.velocity[2]:5.2f})"
                
                print(f"{track.track_id:3d} | {track.object_type:<12} | {state_sym} {track.state.name:<8} | {pos:>25} | {vel:>20}")
    
    print("\n" + "="*70)
    print("✓ REAL DATA TRACKING COMPLETE".center(70))
    print("="*70)


if __name__ == "__main__":
    run_real_data_tracking()
