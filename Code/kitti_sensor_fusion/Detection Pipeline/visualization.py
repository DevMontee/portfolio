"""
Visualization for Detection Pipeline - Processes ALL Frames

Visualizes:
  1. 2D camera detections on image per frame
  2. Detailed frame analysis with cluster associations
  3. Association statistics across all frames
  
Usage:
  python visualization.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from datetime import datetime

# Add sibling directory to path
current_dir = Path(__file__).parent
sibling_dir = current_dir.parent / "Environment Setup & Data Pipeline"
sys.path.insert(0, str(sibling_dir))

from kitti_dataloader import KITTIDataLoader
from detection_pipeline import DetectionPipeline


# Create output directory for visualizations
OUTPUT_DIR = Path(__file__).parent / "visualizations" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Visualization output directory: {OUTPUT_DIR}\n")


class DetectionVisualizer:
    """Visualize detection pipeline results."""
    
    @staticmethod
    def get_cluster_colors(num_clusters: int) -> dict:
        """Generate distinct colors for clusters using HSV color space."""
        colors = {}
        for cluster_id in range(num_clusters):
            hue = (cluster_id / max(num_clusters, 1)) % 1.0
            saturation = 0.8
            value = 0.9
            rgb = hsv_to_rgb([hue, saturation, value])
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors[cluster_id] = bgr
        return colors
    
    @staticmethod
    def draw_2d_bbox(image: np.ndarray, bbox: np.ndarray, 
                    color: tuple, thickness: int = 2, 
                    label: str = None) -> np.ndarray:
        """Draw 2D bounding box on image (OpenCV - BGR 0-255 format)."""
        image = image.copy()
        left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
        
        if label:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = left
            text_y = max(top - 5, 20)
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), color, -1)
            cv2.putText(image, label, (text_x + 2, text_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    @staticmethod
    def project_3d_bbox_to_2d(center: np.ndarray, calibration: dict) -> np.ndarray:
        """Project 3D bounding box center to 2D image coordinates."""
        point_homo = np.hstack([center, 1])
        point_camera = point_homo @ calibration['Tr_velo_to_cam'].T
        point_homo_img = np.hstack([point_camera, 1])
        pixel_homo = point_homo_img @ calibration['P2'].T
        
        depth = pixel_homo[2]
        if depth > 0:
            return pixel_homo[:2] / depth
        return None
    
    @staticmethod
    def visualize_frame(results: dict) -> None:
        """Create detailed visualization and save it."""
        image = results['image'].copy()
        frame_idx = results['frame_idx']
        calibration = results['calibration']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # LEFT: Camera Image with 2D Detections
        ax_left = axes[0]
        ax_left.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_left.set_title(f'Frame {frame_idx} - 2D Detections (Camera)')
        ax_left.axis('off')
        
        for det_idx, detection in enumerate(results['camera_detections']):
            if detection['type'] == 'DontCare':
                continue
            
            bbox = detection['bbox_2d']
            color_rgb = (0, 1, 0)  # Green (normalized for matplotlib)
            
            rect = patches.Rectangle((bbox[0], bbox[1]), 
                                    bbox[2] - bbox[0], bbox[3] - bbox[1],
                                    linewidth=2, edgecolor=color_rgb, facecolor='none')
            ax_left.add_patch(rect)
            
            label = f"{detection['type']} (D{det_idx})"
            if 'score' in detection:
                label += f" {detection['score']:.2f}"
            ax_left.text(bbox[0], bbox[1] - 5, label, 
                        fontsize=8, color='green', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # RIGHT: Image with Associations
        ax_right = axes[1]
        image_assoc = image.copy()
        colors = DetectionVisualizer.get_cluster_colors(len(results['clusters']))
        
        for cluster_id, det_idx in results['associations'].items():
            if det_idx < 0:
                continue
            
            center = results['centers'][cluster_id]
            pixel = DetectionVisualizer.project_3d_bbox_to_2d(center, calibration)
            
            if pixel is not None and 0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0]:
                cv2.circle(image_assoc, (int(pixel[0]), int(pixel[1])), 5, colors[cluster_id], -1)
        
        ax_right.imshow(cv2.cvtColor(image_assoc, cv2.COLOR_BGR2RGB))
        ax_right.set_title(f'Frame {frame_idx} - Cluster Associations')
        ax_right.axis('off')
        
        stats_text = (
            f"LiDAR Points: {results['num_lidar_points']}\n"
            f"3D Clusters: {results['num_clusters']}\n"
            f"2D Detections: {results['num_detections']}\n"
            f"Associations: {sum(1 for v in results['associations'].values() if v >= 0)}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        save_path = OUTPUT_DIR / f"frame_{frame_idx:06d}_detailed.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def visualize_detections_on_image(results: dict) -> None:
        """Visualize 2D detections with associations and save."""
        image = results['image'].copy()
        frame_idx = results['frame_idx']
        
        # Red: unassociated
        for det_idx, detection in enumerate(results['camera_detections']):
            if detection['type'] == 'DontCare':
                continue
            is_associated = any(v == det_idx for v in results['associations'].values())
            if not is_associated:
                bbox = detection['bbox_2d']
                image = DetectionVisualizer.draw_2d_bbox(
                    image, bbox, (0, 0, 255), label=f"{detection['type']} (D{det_idx})", thickness=2
                )
        
        # Green: associated
        for cluster_id, det_idx in results['associations'].items():
            if det_idx < 0:
                continue
            detection = results['camera_detections'][det_idx]
            bbox = detection['bbox_2d']
            image = DetectionVisualizer.draw_2d_bbox(
                image, bbox, (0, 255, 0), label=f"{detection['type']} (C{cluster_id})", thickness=2
            )
        
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_idx} - 2D Detections\n(Green=Associated, Red=Unassociated)')
        plt.axis('off')
        
        save_path = OUTPUT_DIR / f"frame_{frame_idx:06d}_detections.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def visualize_association_stats(results_list: list) -> None:
        """Visualize association statistics across all frames and save."""
        frame_ids = [r['frame_idx'] for r in results_list]
        num_clusters = [r['num_clusters'] for r in results_list]
        num_detections = [r['num_detections'] for r in results_list]
        num_associations = [sum(1 for v in r['associations'].values() if v >= 0) 
                           for r in results_list]
        association_rates = [na / max(nc, 1) * 100 for na, nc in 
                            zip(num_associations, num_clusters)]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Clusters vs Detections
        ax = axes[0, 0]
        ax.plot(frame_ids, num_clusters, 'b-', label='Clusters', linewidth=1, alpha=0.7)
        ax.plot(frame_ids, num_detections, 'r-', label='Detections', linewidth=1, alpha=0.7)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Count')
        ax.set_title('Clusters vs Detections Over Frames')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Number of Associations
        ax = axes[0, 1]
        ax.plot(frame_ids, num_associations, 'g-', linewidth=1, alpha=0.7)
        ax.fill_between(frame_ids, num_associations, alpha=0.3, color='green')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Number of Associations')
        ax.set_title('Associations Per Frame')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Association Rate
        ax = axes[1, 0]
        ax.plot(frame_ids, association_rates, 'g-', linewidth=1, alpha=0.7)
        ax.fill_between(frame_ids, association_rates, alpha=0.3, color='green')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Association Rate (%)')
        ax.set_title('Cluster Association Rate')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Detection Coverage
        ax = axes[1, 1]
        detection_coverage = [na / max(nd, 1) * 100 for na, nd in 
                             zip(num_associations, num_detections)]
        ax.plot(frame_ids, detection_coverage, 'm-', linewidth=1, alpha=0.7)
        ax.fill_between(frame_ids, detection_coverage, alpha=0.3, color='magenta')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Detection Coverage (%)')
        ax.set_title('Detection Coverage (Detections with Clusters)')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "association_statistics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics summary
        print(f"\n  Statistics Summary:")
        print(f"  - Total frames: {len(results_list)}")
        print(f"  - Avg clusters/frame: {np.mean(num_clusters):.1f}")
        print(f"  - Avg detections/frame: {np.mean(num_detections):.1f}")
        print(f"  - Avg associations/frame: {np.mean(num_associations):.1f}")
        print(f"  - Avg association rate: {np.mean(association_rates):.1f}%")


def test_visualization():
    """Test visualization with ALL frames."""
    print("\n" + "="*70)
    print("VISUALIZATION: DETECTION PIPELINE RESULTS (ALL FRAMES)")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}\n")
    
    KITTI_ROOT = r"D:\KITTI Dataset"
    pipeline = DetectionPipeline(KITTI_ROOT, sequence="0000", split="training")
    
    # KITTI sequence 0000 has 154 frames
    total_frames = 154
    frames_to_process = list(range(total_frames))
    results_list = []
    
    print(f"Processing all {total_frames} frames...")
    print("(This will take several minutes)\n")
    
    failed_frames = []
    for i, frame_idx in enumerate(frames_to_process, 1):
        if i % 10 == 0:
            print(f"  [{i:3d}/{total_frames}] Progress: {i/total_frames*100:.1f}%")
        
        try:
            results = pipeline.process_frame(frame_idx)
            results_list.append(results)
        except Exception as e:
            failed_frames.append((frame_idx, str(e)[:50]))
    
    print(f"\n✓ Processed {len(results_list)}/{total_frames} frames successfully")
    if failed_frames:
        print(f"✗ Failed to process {len(failed_frames)} frames")
    
    # === Visualization 1: Per-frame visualizations ===
    print("\n" + "-"*70)
    print(f"Generating visualizations for {len(results_list)} frames...")
    print("-"*70 + "\n")
    
    for i, results in enumerate(results_list, 1):
        frame_idx = results['frame_idx']
        if i % 10 == 0:
            print(f"  [{i:3d}/{len(results_list)}] Saving frame {frame_idx}...")
        DetectionVisualizer.visualize_detections_on_image(results)
        DetectionVisualizer.visualize_frame(results)
    
    print(f"\n✓ Saved {len(results_list) * 2} frame visualizations")
    
    # === Visualization 2: Statistics ===
    print("\n" + "-"*70)
    print("Generating association statistics...")
    print("-"*70)
    DetectionVisualizer.visualize_association_stats(results_list)
    print(f"  ✓ Saved: association_statistics.png")
    
    print("\n" + "="*70)
    print("✓ All visualizations saved successfully!")
    print(f"📁 Check output folder: {OUTPUT_DIR}")
    print(f"📊 Total files generated: {len(results_list) * 2 + 1}")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_visualization()
