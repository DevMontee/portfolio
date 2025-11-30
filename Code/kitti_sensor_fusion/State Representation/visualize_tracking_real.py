"""
TRACKING VISUALIZATION
======================

Visualize tracked objects on KITTI images with:
- Bounding boxes
- Track IDs
- Object types
- Positions and velocities

Run:
    python visualize_tracking_real.py
"""

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import sys

# Add paths
sys.path.insert(0, r"D:\kitti_sensor_fusion\Environment Setup & Data Pipeline")
sys.path.insert(0, r"D:\kitti_sensor_fusion\Detection Pipeline")

try:
    from kitti_dataloader import KITTIDataLoader
    from detection_pipeline import DetectionPipeline
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)


class TrackState(Enum):
    """Track states."""
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3


@dataclass
class KalmanTracker:
    """1D Kalman filter."""
    x: np.ndarray = None
    P: np.ndarray = None
    
    def __post_init__(self):
        if self.x is None:
            self.x = np.array([0.0, 0.0])
        if self.P is None:
            self.P = np.eye(2)
    
    def predict(self):
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.01, 0], [0, 0.01]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[0]
    
    def update(self, z):
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
    bbox_2d: np.ndarray = None
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
        if self.bbox_2d is None:
            self.bbox_2d = np.zeros(4)
        if self.kalmans is None:
            self.kalmans = {
                'x': KalmanTracker(),
                'y': KalmanTracker(),
                'z': KalmanTracker()
            }
    
    def predict(self):
        self.position = np.array([
            self.kalmans['x'].predict(),
            self.kalmans['y'].predict(),
            self.kalmans['z'].predict()
        ])
        self.time_since_update += 1
        self.age += 1
    
    def update(self, position, bbox_2d=None, obj_type="Unknown", confidence=0.9):
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
        
        if bbox_2d is not None:
            self.bbox_2d = bbox_2d
        
        self.object_type = obj_type
        self.confidence = confidence
        self.time_since_update = 0


class SimpleTracker:
    """Multi-object tracker."""
    
    def __init__(self, max_age=30, min_hits=3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_hits = min_hits
    
    def predict(self):
        for track in self.tracks.values():
            if track.state != TrackState.DELETED:
                track.predict()
    
    def associate(self, detections, distance_threshold=5.0):
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
        for track_id, det_idx in matches.items():
            det = detections[det_idx]
            self.tracks[track_id].update(
                det['location_3d'],
                det.get('bbox_2d'),
                det.get('type', 'Unknown'),
                det.get('confidence', 0.9)
            )
            if self.tracks[track_id].state == TrackState.TENTATIVE and self.tracks[track_id].age >= self.min_hits:
                self.tracks[track_id].state = TrackState.CONFIRMED
        
        for track in self.tracks.values():
            if track.track_id not in matches and track.state != TrackState.DELETED:
                if track.time_since_update > self.max_age:
                    track.state = TrackState.DELETED
                elif track.state == TrackState.CONFIRMED:
                    track.state = TrackState.LOST
    
    def create_new(self, detections, unmatched):
        for idx in unmatched:
            det = detections[idx]
            track = SimpleTrack(
                track_id=self.next_id,
                position=det['location_3d'].copy(),
                bbox_2d=det.get('bbox_2d'),
                object_type=det.get('type', 'Unknown'),
                confidence=det.get('confidence', 0.9)
            )
            track.kalmans['x'].x[0] = det['location_3d'][0]
            track.kalmans['y'].x[0] = det['location_3d'][1]
            track.kalmans['z'].x[0] = det['location_3d'][2]
            
            self.tracks[self.next_id] = track
            self.next_id += 1
    
    def cleanup(self):
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.state != TrackState.DELETED}
    
    def get_active_tracks(self):
        return [t for t in self.tracks.values() if t.state in [TrackState.CONFIRMED, TrackState.LOST]]


def draw_tracking_results(image, tracks):
    """Draw tracking results on image."""
    
    image = image.copy()
    h, w = image.shape[:2]
    
    # Colors for different states
    state_colors = {
        TrackState.CONFIRMED: (0, 255, 0),      # Green
        TrackState.TENTATIVE: (0, 165, 255),    # Orange
        TrackState.LOST: (0, 0, 255)            # Red
    }
    
    for track in sorted(tracks, key=lambda t: t.track_id):
        if track.bbox_2d is None or np.any(np.isnan(track.bbox_2d)):
            continue
        
        x1, y1, x2, y2 = track.bbox_2d.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        color = state_colors.get(track.state, (255, 255, 255))
        thickness = 3 if track.state == TrackState.CONFIRMED else 2
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw track ID and type
        label = f"ID:{track.track_id} {track.object_type}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(
            image,
            (x1, y1 - text_size[1] - 8),
            (x1 + text_size[0] + 8, y1),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            image, label, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        
        # Position info below box
        pos_text = f"({track.position[0]:.1f}, {track.position[1]:.1f}, {track.position[2]:.1f})"
        cv2.putText(
            image, pos_text, (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    
    return image


def draw_stats(image, frame_idx, num_detections, num_tracks, num_confirmed):
    """Draw statistics panel."""
    
    image = image.copy()
    
    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    
    # Text
    y_pos = 25
    texts = [
        f"Frame: {frame_idx}",
        f"Detections: {num_detections}",
        f"Total Tracks: {num_tracks}",
        f"Confirmed: {num_confirmed}"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(
            image, text, (10, y_pos + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    
    return image


def run_visualization():
    """Run visualization."""
    
    print("\n" + "="*80)
    print("TRACKING VISUALIZATION".center(80))
    print("="*80 + "\n")
    
    KITTI_ROOT = r"D:\KITTI Dataset"
    SEQUENCE = "0000"
    
    if not Path(KITTI_ROOT).exists():
        print(f"❌ KITTI dataset not found at: {KITTI_ROOT}")
        return
    
    print(f"Loading KITTI dataset...")
    
    try:
        loader = KITTIDataLoader(KITTI_ROOT, sequence=SEQUENCE, split="training")
        detection_pipeline = DetectionPipeline(KITTI_ROOT, sequence=SEQUENCE, split="training")
        tracker = SimpleTracker(max_age=30, min_hits=3)
        
        print(f"✓ Loaded {len(loader)} frames\n")
        print(f"Controls:")
        print(f"  'q' - Quit")
        print(f"  'p' - Pause/Resume")
        print(f"  'n' - Next frame (when paused)")
        print(f"  's' - Save frame as PNG\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Create output directory
    output_dir = Path("tracking_frames")
    output_dir.mkdir(exist_ok=True)
    
    paused = False
    frame_idx = 0
    saved_count = 0
    
    while frame_idx < len(loader):
        try:
            # Get frame
            frame = loader[frame_idx]
            image = frame['image'].copy()
            
            # Get detections
            detection_results = detection_pipeline.process_frame(frame_idx)
            
            # Extract detections
            detections = []
            if 'labels' in frame:
                for label in frame['labels']:
                    obj_type = label.get('type', 'Unknown')
                    location = label.get('location_3d', np.array([-10, -1, -1]))
                    bbox_2d = label.get('bbox_2d')
                    
                    # Filter out invalid objects
                    if obj_type == 'DontCare' or (location[0] == -10 and location[1] == -1):
                        continue
                    
                    detection = {
                        'location_3d': np.array(location, dtype=np.float32),
                        'bbox_2d': np.array(bbox_2d, dtype=np.float32),
                        'type': obj_type,
                        'confidence': 0.9
                    }
                    detections.append(detection)
            
            # Tracking
            tracker.predict()
            matches, unmatched = tracker.associate(detections, distance_threshold=5.0)
            tracker.update(detections, matches)
            tracker.create_new(detections, unmatched)
            tracker.cleanup()
            
            # Get tracks
            active_tracks = tracker.get_active_tracks()
            
            # Visualize
            annotated = draw_tracking_results(image, active_tracks)
            annotated = draw_stats(
                annotated, frame_idx,
                len(detections),
                len(active_tracks),
                len([t for t in active_tracks if t.state == TrackState.CONFIRMED])
            )
            
            # Display
            cv2.imshow('Tracking Visualization', annotated)
            
            key = cv2.waitKey(50 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print(f"Quitting...")
                break
            
            elif key == ord('p'):
                paused = not paused
                status = "⏸ PAUSED" if paused else "▶ PLAYING"
                print(f"{status}")
            
            elif key == ord('s'):
                filename = output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(filename), annotated)
                saved_count += 1
                print(f"💾 Saved: {filename}")
            
            elif key == ord('n') and paused:
                frame_idx += 1
                continue
            
            if not paused:
                frame_idx += 1
        
        except Exception as e:
            print(f"❌ Error at frame {frame_idx}: {e}")
            frame_idx += 1
    
    cv2.destroyAllWindows()
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE".center(80))
    print(f"{'='*80}\n")
    print(f"Processed frames: {frame_idx}")
    print(f"Frames saved: {saved_count}")
    if saved_count > 0:
        print(f"Output directory: {output_dir.resolve()}\n")


if __name__ == "__main__":
    run_visualization()
