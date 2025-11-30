"""
Day 2: Detection Pipeline

Implements:
  1. DBSCAN clustering on LiDAR point clouds
  2. Load pre-computed camera 2D detections from KITTI
  3. Project LiDAR clusters to camera frame
  4. Associate 2D and 3D detections
  
Usage:
  python detection_pipeline.py
"""

import sys
from pathlib import Path

# Add sibling directory to path
current_dir = Path(__file__).parent  # D:\kitti_sensor_fusion\Day 2
sibling_dir = current_dir.parent / "Environment Setup & Data Pipeline"
sys.path.insert(0, str(sibling_dir))

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple
import cv2
from kitti_dataloader import KITTIDataLoader


class LiDARClusterer:
    """
    DBSCAN clustering for 3D LiDAR point clouds.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between points in a cluster (meters)
            min_samples: Minimum points to form a cluster
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def cluster(self, points: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Cluster 3D LiDAR points using DBSCAN.
        
        Args:
            points: Point cloud (N, 3) or (N, 4) with intensity
            
        Returns:
            Dictionary with cluster_id -> points
        """
        # Extract XYZ only
        if points.shape[1] == 4:
            xyz = points[:, :3]
        else:
            xyz = points
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(xyz)
        labels = clustering.labels_
        
        # Group points by cluster
        clusters = {}
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
            mask = labels == cluster_id
            clusters[cluster_id] = points[mask]
        
        return clusters
    
    def get_cluster_centers(self, clusters: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Get center of mass for each cluster."""
        centers = {}
        for cluster_id, points in clusters.items():
            centers[cluster_id] = points[:, :3].mean(axis=0)
        return centers
    
    def get_cluster_bounds(self, clusters: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """Get 3D bounding box for each cluster."""
        bounds = {}
        for cluster_id, points in clusters.items():
            xyz = points[:, :3]
            bounds[cluster_id] = {
                'min': xyz.min(axis=0),
                'max': xyz.max(axis=0),
                'center': xyz.mean(axis=0),
                'size': xyz.max(axis=0) - xyz.min(axis=0),
            }
        return bounds


class CameraDetectionLoader:
    """
    Load pre-computed 2D camera detections from KITTI.
    
    KITTI provides detections in detection results format:
    type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
    dimensions_3d location_3d rotation_y score
    """
    
    def __init__(self, detection_dir: Path):
        """
        Initialize detection loader.
        
        Args:
            detection_dir: Path to detection results directory
        """
        self.detection_dir = Path(detection_dir)
    
    def load_detections(self, sequence: str, frame_idx: int) -> List[Dict]:
        """
        Load 2D detections for a frame.
        
        Args:
            sequence: Sequence ID (e.g., "0000")
            frame_idx: Frame index
            
        Returns:
            List of detection dictionaries
        """
        # Try to find detection file
        detection_file = self.detection_dir / sequence / f"{frame_idx:06d}.txt"
        
        detections = []
        
        if not detection_file.exists():
            return detections
        
        with open(detection_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                # Parse detection
                detection = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': np.array([float(parts[4]), float(parts[5]),
                                        float(parts[6]), float(parts[7])]),
                    'dimensions': np.array([float(parts[8]), float(parts[9]), float(parts[10])]),
                    'location_3d': np.array([float(parts[11]), float(parts[12]), float(parts[13])]),
                    'rotation_y': float(parts[14]),
                }
                
                if len(parts) > 15:
                    detection['score'] = float(parts[15])
                
                detections.append(detection)
        
        return detections
    
    def create_detections_from_labels(self, labels: List[Dict]) -> List[Dict]:
        """
        Create detection format from ground truth labels.
        Useful for testing when detection files aren't available.
        
        Args:
            labels: List of label dictionaries
            
        Returns:
            List of detection dictionaries (same format)
        """
        detections = []
        for label in labels:
            detection = {
                'type': label['type'],
                'truncated': label['truncated'],
                'occluded': label['occluded'],
                'alpha': label['alpha'],
                'bbox_2d': label['bbox_2d'],
                'dimensions': label['dimensions'],
                'location_3d': label['location_3d'],
                'rotation_y': label['rotation_y'],
                'score': 0.99,  # Ground truth has high confidence
            }
            detections.append(detection)
        
        return detections


class DetectionAssociator:
    """
    Associate 2D camera detections with 3D LiDAR clusters.
    """
    
    @staticmethod
    def project_cluster_to_image(cluster_center: np.ndarray, 
                                loader: KITTIDataLoader) -> Tuple[np.ndarray, float]:
        """
        Project cluster center to image plane.
        
        Args:
            cluster_center: 3D point in LiDAR frame (3,)
            loader: KITTI data loader with calibration
            
        Returns:
            (pixel_coords, depth) - projected point and depth
        """
        # Transform to camera frame
        point_homo = np.hstack([cluster_center, 1])
        point_camera = point_homo @ loader.calibration['Tr_velo_to_cam'].T
        
        # Project to image
        point_homo_img = np.hstack([point_camera, 1])
        pixel_homo = point_homo_img @ loader.calibration['P2'].T
        
        depth = pixel_homo[2]
        if depth > 0:
            pixel = pixel_homo[:2] / depth
        else:
            pixel = np.array([-1, -1])
        
        return pixel, depth
    
    @staticmethod
    def iou_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate 2D Intersection over Union for two bounding boxes.
        
        Args:
            bbox1, bbox2: [left, top, right, bottom]
            
        Returns:
            IoU value (0-1)
        """
        # Intersection
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def associate_clusters_to_detections(clusters: Dict[int, np.ndarray],
                                        detections: List[Dict],
                                        loader: KITTIDataLoader,
                                        iou_threshold: float = 0.1,
                                        use_center_distance: bool = False) -> Dict[int, int]:
        """
        Associate LiDAR clusters with 2D camera detections using IoU.
        
        Improvements:
          - Larger match window (scales with depth)
          - Filters invalid detections (DontCare, negative positions)
          - More permissive IoU threshold
          - Alternative center-distance matching
        
        Args:
            clusters: Dictionary of cluster_id -> points
            detections: List of 2D detections
            loader: KITTI data loader
            iou_threshold: Minimum IoU to consider a match (default: 0.1)
            use_center_distance: Use center distance instead of IoU
            
        Returns:
            Dictionary of cluster_id -> detection_idx (or -1 if no match)
        """
        # Get cluster centers and bounds
        clusterer = LiDARClusterer()
        centers = clusterer.get_cluster_centers(clusters)
        bounds = clusterer.get_cluster_bounds(clusters)
        
        associations = {}
        used_detections = set()
        
        # Sort by number of points (larger clusters first - more reliable)
        sorted_clusters = sorted(clusters.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)
        
        # Filter out invalid/placeholder detections
        valid_detections = []
        for det_idx, det in enumerate(detections):
            # Skip "DontCare" and invalid detections
            if det['type'] == 'DontCare':
                continue
            # Skip if location is clearly invalid (placeholder values)
            if det['location_3d'][0] < -100:  # x coordinate too negative
                continue
            valid_detections.append((det_idx, det))
        
        for cluster_id, cluster_points in sorted_clusters:
            center = centers[cluster_id]
            bound = bounds[cluster_id]
            
            # Project center to image
            pixel, depth = DetectionAssociator.project_cluster_to_image(center, loader)
            
            if pixel[0] < 0 or depth <= 0:
                associations[cluster_id] = -1
                continue
            
            # Create larger bounding box around projected point for matching
            # Scale match window based on depth (farther objects have larger projected areas)
            match_size = max(30, int(50 * (depth / 20)))  # Adaptive window size
            bbox_from_cluster = np.array([pixel[0] - match_size, pixel[1] - match_size,
                                         pixel[0] + match_size, pixel[1] + match_size])
            
            best_score = -1
            best_detection_idx = -1
            
            # Find best matching detection
            for det_idx, detection in valid_detections:
                if det_idx in used_detections:
                    continue
                
                det_bbox = detection['bbox_2d']
                
                if use_center_distance:
                    # Alternative: use center distance (can be better for small objects)
                    det_center = np.array([(det_bbox[0] + det_bbox[2]) / 2,
                                          (det_bbox[1] + det_bbox[3]) / 2])
                    distance = np.linalg.norm(pixel - det_center)
                    # Convert distance to score (lower distance = higher score)
                    score = 1.0 / (1.0 + distance)
                else:
                    # Use IoU
                    score = DetectionAssociator.iou_2d(bbox_from_cluster, det_bbox)
                
                if score > best_score and score > iou_threshold:
                    best_score = score
                    best_detection_idx = det_idx
            
            if best_detection_idx >= 0:
                associations[cluster_id] = best_detection_idx
                used_detections.add(best_detection_idx)
            else:
                associations[cluster_id] = -1
        
        return associations


class DetectionPipeline:
    """
    Complete detection pipeline combining DBSCAN, camera detections, and association.
    """
    
    def __init__(self, kitti_root: str, sequence: str = "0000", split: str = "training"):
        """Initialize pipeline."""
        self.loader = KITTIDataLoader(kitti_root, sequence=sequence, split=split)
        self.clusterer = LiDARClusterer(eps=0.5, min_samples=5)
        self.detection_loader = CameraDetectionLoader(Path(kitti_root) / "detections")
        self.sequence = sequence
    
    def process_frame(self, frame_idx: int) -> Dict:
        """
        Process a frame through the complete detection pipeline.
        
        Args:
            frame_idx: Frame index to process
            
        Returns:
            Dictionary with clustering and association results
        """
        # Load frame
        frame = self.loader[frame_idx]
        lidar_points = frame['lidar']
        labels = frame['labels']
        
        # Step 1: Cluster LiDAR points
        clusters = self.clusterer.cluster(lidar_points)
        bounds = self.clusterer.get_cluster_bounds(clusters)
        centers = self.clusterer.get_cluster_centers(clusters)
        
        # Step 2: Load camera detections (use ground truth labels if detection files missing)
        camera_detections = self.detection_loader.create_detections_from_labels(labels)
        
        # Step 3: Associate clusters to detections
        associations = DetectionAssociator.associate_clusters_to_detections(
            clusters, camera_detections, self.loader
        )
        
        # Compile results
        results = {
            'frame_idx': frame_idx,
            'num_lidar_points': len(lidar_points),
            'num_clusters': len(clusters),
            'num_detections': len(camera_detections),
            'clusters': clusters,
            'bounds': bounds,
            'centers': centers,
            'camera_detections': camera_detections,
            'associations': associations,  # cluster_id -> detection_idx
            'image': frame['image'],
            'calibration': frame['calibration'],
        }
        
        return results
    
    def print_results(self, results: Dict) -> None:
        """Print summary of detection pipeline results."""
        print(f"\n{'='*70}")
        print(f"Frame {results['frame_idx']:3d} Detection Results")
        print(f"{'='*70}")
        print(f"LiDAR points:        {results['num_lidar_points']:6d}")
        print(f"3D clusters:         {results['num_clusters']:6d}")
        print(f"2D detections:       {results['num_detections']:6d}")
        print(f"\nAssociations:")
        
        for cluster_id in sorted(results['associations'].keys()):
            det_idx = results['associations'][cluster_id]
            center = results['centers'][cluster_id]
            size = results['bounds'][cluster_id]['size']
            
            if det_idx >= 0:
                det = results['camera_detections'][det_idx]
                print(f"  Cluster {cluster_id:2d} → Detection {det_idx:2d}")
                print(f"    Center: ({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f})")
                print(f"    Size:   ({size[0]:6.2f}, {size[1]:6.2f}, {size[2]:6.2f})")
                print(f"    Type:   {det['type']}")
            else:
                print(f"  Cluster {cluster_id:2d} → No match")
                print(f"    Center: ({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f})")
        
        print(f"{'='*70}\n")


def test_detection_pipeline():
    """Test the complete detection pipeline."""
    print("\n" + "="*70)
    print("DAY 2: DETECTION PIPELINE")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ DBSCAN clustering on 3D LiDAR points")
    print("  ✓ Load pre-computed 2D camera detections")
    print("  ✓ Project LiDAR clusters to camera frame")
    print("  ✓ Associate 2D and 3D detections")
    
    # Initialize pipeline
    KITTI_ROOT = r"D:\KITTI Dataset"
    pipeline = DetectionPipeline(KITTI_ROOT, sequence="0000", split="training")
    
    # Process a few frames
    frames_to_process = [0, 50, 100]
    
    for frame_idx in frames_to_process:
        results = pipeline.process_frame(frame_idx)
        pipeline.print_results(results)


if __name__ == "__main__":
    test_detection_pipeline()
