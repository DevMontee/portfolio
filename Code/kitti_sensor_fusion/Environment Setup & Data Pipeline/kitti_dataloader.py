import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2


class KITTIDataLoader:
    """
    Loads and synchronizes KITTI dataset sequences.
    
    KITTI structure:
    data_tracking_image_2/
    data_tracking_velodyne/
    data_tracking_calib/
    data_tracking_label_2/
    data_splits/
    
    Each sequence has numbered frames (e.g., 0000.png, 0001.png, ...)
    """
    
    def __init__(self, kitti_root: str, sequence: str = "0000", split: str = "training"):
        """
        Initialize KITTI data loader.
        
        Args:
            kitti_root: Path to KITTI tracking dataset root
            sequence: Sequence ID (e.g., "0000", "0001", ...)
            split: "training" or "testing"
        """
        self.kitti_root = Path(kitti_root)
        self.sequence = sequence
        self.split = split
        
        # Define paths - Updated for actual KITTI structure with image_02, velodyne, label_02 subfolders
        self.image_dir = self.kitti_root / f"data_tracking_image_2" / split / "image_02" / sequence
        self.lidar_dir = self.kitti_root / f"data_tracking_velodyne" / split / "velodyne" / sequence
        self.calib_dir = self.kitti_root / f"data_tracking_calib" / split
        # Labels are sequence-level files (one .txt per sequence, not per frame)
        self.label_dir = self.kitti_root / f"data_tracking_label_2" / split / "label_02"
        
        # Verify directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Load calibration matrices (constant for entire sequence)
        self.calibration = self._load_calibration()
        
        # Get number of frames
        self.num_frames = len(list(self.image_dir.glob("*.png")))
        
        if self.num_frames == 0:
            raise ValueError(f"No images found in {self.image_dir}")
    
    def _load_calibration(self) -> Dict[str, np.ndarray]:
        """
        Load calibration matrices from KITTI calib file.
        
        Returns calibration dict with keys:
            - P0, P1, P2, P3: Projection matrices for 4 cameras
            - R0_rect: Rectification rotation matrix
            - Tr_velo_to_cam: LiDAR to camera transformation
            - Tr_imu_to_velo: IMU to LiDAR transformation (rarely used)
        """
        # Calibration files are in calib/ subfolder
        calib_file = self.calib_dir / "calib" / f"{int(self.sequence):04d}.txt"
        
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        
        calibration = {}
        
        with open(calib_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Handle both formats:
                # Format 1: "key: value value value ..." (with colon)
                # Format 2: "key value value value ..." (without colon)
                
                if ':' in line:
                    # Format with colon
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                else:
                    # Format without colon - split on first space
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    key = parts[0]
                    value = parts[1]
                
                try:
                    values = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    continue
                
                # Normalize key names for consistency
                if key == 'R_rect':
                    key = 'R0_rect'
                elif key == 'Tr_velo_cam':
                    key = 'Tr_velo_to_cam'
                elif key == 'Tr_imu_velo':
                    key = 'Tr_imu_to_velo'
                
                # Reshape based on key type
                if key in ['P0', 'P1', 'P2', 'P3']:
                    calibration[key] = values.reshape(3, 4)
                elif key == 'R0_rect':
                    calibration[key] = values.reshape(3, 3)
                elif key == 'Tr_velo_to_cam':
                    calibration[key] = values.reshape(3, 4)
                elif key == 'Tr_imu_to_velo':
                    calibration[key] = values.reshape(3, 4)
        
        return calibration
    
    def load_image(self, frame_idx: int) -> np.ndarray:
        """
        Load camera image (left color camera - cam2).
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Image array (H, W, 3) in BGR format
        """
        image_path = self.image_dir / f"{frame_idx:06d}.png"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        
        return image
    
    def load_lidar(self, frame_idx: int) -> np.ndarray:
        """
        Load LiDAR point cloud in Velodyne coordinate system.
        
        KITTI LiDAR format: .bin file with N x 4 floats (x, y, z, intensity)
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Point cloud array (N, 4) - [x, y, z, intensity]
        """
        lidar_path = self.lidar_dir / f"{frame_idx:06d}.bin"
        
        if not lidar_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
        
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
        
        return points
    
    def load_labels(self, frame_idx: int) -> List[Dict]:
        """
        Load ground truth object labels for a frame.
        
        KITTI label format (for tracking):
        frame_id track_id type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom 
        dimensions_3d location_3d rotation_y score
        
        Handles two formats:
        1. Frame-level: one .txt file per frame
        2. Sequence-level: one .txt file per sequence (with frame_id in each line)
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of label dictionaries with keys:
                - frame_id, track_id, type, truncated, occluded
                - bbox_2d: [left, top, right, bottom]
                - bbox_2d_size: [width, height]
                - dimensions: [height, width, length] (3D object dimensions)
                - location_3d: [x, y, z] (3D position in camera frame)
                - rotation_y: rotation around Y axis
        """
        labels = []
        
        # Try frame-level labels first (standard format)
        frame_label_path = self.label_dir / f"{frame_idx:06d}.txt"
        if frame_label_path.exists():
            with open(frame_label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 15:
                        continue
                    
                    label = {
                        'frame_id': int(parts[0]),
                        'track_id': int(parts[1]),
                        'type': parts[2],
                        'truncated': float(parts[3]),
                        'occluded': int(parts[4]),
                        'alpha': float(parts[5]),
                        'bbox_2d': np.array([float(parts[6]), float(parts[7]), 
                                            float(parts[8]), float(parts[9])]),
                        'bbox_2d_size': np.array([float(parts[8]) - float(parts[6]),
                                                 float(parts[9]) - float(parts[7])]),
                        'dimensions': np.array([float(parts[10]), float(parts[11]), float(parts[12])]),
                        'location_3d': np.array([float(parts[13]), float(parts[14]), float(parts[15])]),
                        'rotation_y': float(parts[16]),
                    }
                    
                    if len(parts) > 17:
                        label['score'] = float(parts[17])
                    
                    labels.append(label)
            
            return labels
        
        # Try sequence-level labels (alternative format: one file per sequence)
        sequence_label_path = self.label_dir / f"{int(self.sequence):04d}.txt"
        if sequence_label_path.exists():
            with open(sequence_label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 15:
                        continue
                    
                    # First field is frame_id - filter for current frame
                    current_frame_id = int(parts[0])
                    if current_frame_id != frame_idx:
                        continue
                    
                    label = {
                        'frame_id': current_frame_id,
                        'track_id': int(parts[1]),
                        'type': parts[2],
                        'truncated': float(parts[3]),
                        'occluded': int(parts[4]),
                        'alpha': float(parts[5]),
                        'bbox_2d': np.array([float(parts[6]), float(parts[7]), 
                                            float(parts[8]), float(parts[9])]),
                        'bbox_2d_size': np.array([float(parts[8]) - float(parts[6]),
                                                 float(parts[9]) - float(parts[7])]),
                        'dimensions': np.array([float(parts[10]), float(parts[11]), float(parts[12])]),
                        'location_3d': np.array([float(parts[13]), float(parts[14]), float(parts[15])]),
                        'rotation_y': float(parts[16]),
                    }
                    
                    if len(parts) > 17:
                        label['score'] = float(parts[17])
                    
                    labels.append(label)
            
            return labels
        
        # No labels found
        return labels
    
    def project_lidar_to_camera(self, points_lidar: np.ndarray) -> np.ndarray:
        """
        Project LiDAR points from Velodyne frame to camera frame.
        
        Transformation chain: Velodyne → Camera → Image
        
        Args:
            points_lidar: LiDAR points (N, 3) or (N, 4) with intensity
            
        Returns:
            Points in camera coordinate system (N, 3)
        """
        # Extract xyz if intensity is included
        if points_lidar.shape[1] == 4:
            points_xyz = points_lidar[:, :3]
        else:
            points_xyz = points_lidar
        
        # Add homogeneous coordinate
        points_homog = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        
        # Apply Tr_velo_to_cam (3x4 matrix)
        points_camera = points_homog @ self.calibration['Tr_velo_to_cam'].T
        
        return points_camera  # (N, 3)
    
    def project_to_image(self, points_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project camera frame points to image plane using camera intrinsics.
        
        Args:
            points_camera: Points in camera frame (N, 3)
            
        Returns:
            tuple: (pixel_coords, depth)
                - pixel_coords: (N, 2) normalized image coordinates
                - depth: (N,) depth values for each point
        """
        # Get projection matrix P2 (left color camera)
        P2 = self.calibration['P2']
        
        # Add homogeneous coordinate
        points_homog = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
        
        # Project: p_image = P @ p_camera
        pixel_homog = points_homog @ P2.T  # (N, 3)
        
        # Normalize by depth (3rd coordinate)
        depth = pixel_homog[:, 2]
        valid_mask = depth > 0
        
        pixel_coords = np.zeros((points_camera.shape[0], 2), dtype=np.float32)
        pixel_coords[valid_mask] = pixel_homog[valid_mask, :2] / depth[valid_mask, np.newaxis]
        
        return pixel_coords, depth
    
    def get_frame(self, frame_idx: int) -> Dict:
        """
        Load all synchronized data for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Dictionary containing:
                - image: Camera image (H, W, 3)
                - lidar: Point cloud (N, 4)
                - labels: List of ground truth objects
                - calibration: Calibration matrices
                - metadata: Frame number and sequence info
        """
        data = {
            'image': self.load_image(frame_idx),
            'lidar': self.load_lidar(frame_idx),
            'labels': self.load_labels(frame_idx),
            'calibration': self.calibration,
            'metadata': {
                'frame_idx': frame_idx,
                'sequence': self.sequence,
                'num_frames': self.num_frames,
            }
        }
        
        return data
    
    def __len__(self) -> int:
        """Return number of frames in sequence."""
        return self.num_frames
    
    def __getitem__(self, idx: int) -> Dict:
        """Allow indexing: data = loader[0]"""
        return self.get_frame(idx)


class KITTIDataProcessor:
    """Helper class for common data processing operations."""
    
    @staticmethod
    def filter_lidar_fov(points_lidar: np.ndarray, image_shape: Tuple[int, int],
                        calibration: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Filter LiDAR points to only those visible in camera FOV.
        
        Args:
            points_lidar: Points in LiDAR frame (N, 4)
            image_shape: Camera image shape (H, W)
            calibration: Calibration dictionary
            
        Returns:
            Filtered points (M, 4) where M <= N
        """
        loader = KITTIDataLoader.__new__(KITTIDataLoader)
        loader.calibration = calibration
        
        # Project to camera
        points_camera = loader.project_lidar_to_camera(points_lidar)
        
        # Project to image
        pixel_coords, depth = loader.project_to_image(points_camera)
        
        # Filter by image bounds and positive depth
        H, W = image_shape
        valid_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) & \
                     (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H) & \
                     (depth > 0)
        
        return points_lidar[valid_mask]
    
    @staticmethod
    def get_lidar_statistics(points: np.ndarray) -> Dict:
        """Compute basic statistics of point cloud."""
        return {
            'num_points': len(points),
            'range': np.linalg.norm(points[:, :3], axis=1).max(),
            'x_range': [points[:, 0].min(), points[:, 0].max()],
            'y_range': [points[:, 1].min(), points[:, 1].max()],
            'z_range': [points[:, 2].min(), points[:, 2].max()],
        }


if __name__ == "__main__":
    # Example usage
    KITTI_ROOT = r"D:\KITTI Dataset"  # Updated path
    
    # Load data for sequence 0000
    loader = KITTIDataLoader(KITTI_ROOT, sequence="0000", split="training")
    
    print(f"Dataset loaded: {len(loader)} frames")
    print(f"Calibration matrices loaded: {list(loader.calibration.keys())}")
    
    # Load first frame
    frame = loader[0]
    print(f"\nFrame 0:")
    print(f"  Image shape: {frame['image'].shape}")
    print(f"  LiDAR points: {frame['lidar'].shape}")
    print(f"  Labels: {len(frame['labels'])} objects")
    
    # Print first object label
    if frame['labels']:
        label = frame['labels'][0]
        print(f"\n  First object:")
        print(f"    Type: {label['type']}, Track ID: {label['track_id']}")
        print(f"    2D bbox: {label['bbox_2d']}")
        print(f"    3D location: {label['location_3d']}")
        print(f"    3D dimensions (h,w,l): {label['dimensions']}")
    
    # Test projections
    lidar_points = frame['lidar'][:100]
    points_camera = loader.project_lidar_to_camera(lidar_points)
    pixel_coords, depth = loader.project_to_image(points_camera)
    
    print(f"\nProjection test (first 5 points):")
    print(f"  LiDAR (xyz):     {lidar_points[:5, :3]}")
    print(f"  Camera (xyz):    {points_camera[:5]}")
    print(f"  Image (uv):      {pixel_coords[:5]}")
    print(f"  Depth:           {depth[:5]}")
    
    # LiDAR statistics
    stats = KITTIDataProcessor.get_lidar_statistics(lidar_points)
    print(f"\nLiDAR statistics (first 100 points):")
    for key, val in stats.items():
        print(f"  {key}: {val}")
