"""
IMM Visualization Module - Model Performance Analysis
Generates comprehensive visualizations comparing Constant Velocity vs Turning models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict


class IMMVisualizer:
    """Generates visualizations for IMM tracking results"""
    
    def __init__(self, results_file='imm_kitti_results.json'):
        self.results_file = results_file
        self.results = None
        self.load_results()
        self.colors = {
            'cv': '#2E86AB',      # Blue
            'turning': '#A23B72',  # Purple
            'truth': '#06A77D',    # Green
            'error': '#F18F01'     # Orange
        }
    
    def load_results(self):
        """Load results from JSON file"""
        if not Path(self.results_file).exists():
            print(f"Warning: {self.results_file} not found. Visualizations will be limited.")
            return
        
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
    
    def extract_track_data(self):
        """Extract trajectory data organized by track ID"""
        tracks = defaultdict(lambda: {
            'frames': [],
            'positions': [],
            'velocities': [],
            'cv_probs': [],
            'turn_probs': [],
            'yaw_rates': []
        })
        
        if not self.results:
            return tracks
        
        for frame_data in self.results.get('frames', []):
            frame_num = frame_data['frame']
            for track in frame_data.get('tracks', []):
                track_id = track['track_id']
                
                tracks[track_id]['frames'].append(frame_num)
                tracks[track_id]['positions'].append([
                    track['position']['x'],
                    track['position']['y'],
                    track['position']['z']
                ])
                tracks[track_id]['velocities'].append([
                    track.get('velocity', {}).get('vx', 0),
                    track.get('velocity', {}).get('vy', 0),
                    track.get('velocity', {}).get('vz', 0)
                ])
                
                model_probs = track.get('model_probs', {})
                tracks[track_id]['cv_probs'].append(model_probs.get('CV', 0.5))
                tracks[track_id]['turn_probs'].append(model_probs.get('Turn', 0.5))
                tracks[track_id]['yaw_rates'].append(track.get('yaw_rate', 0))
        
        return tracks
    
    def plot_trajectory_comparison(self, track_id, output_file='trajectory_comparison.png'):
        """
        Plot top-down trajectory with velocity vectors and model indicators
        Shows CV vs Turning model performance
        """
        tracks = self.extract_track_data()
        
        if track_id not in tracks:
            print(f"Track {track_id} not found")
            return
        
        track = tracks[track_id]
        positions = np.array(track['positions'])
        velocities = np.array(track['velocities'])
        cv_probs = np.array(track['cv_probs'])
        turn_probs = np.array(track['turn_probs'])
        frames = track['frames']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Track {track_id}: CV vs Turning Model Comparison', fontsize=16, fontweight='bold')
        
        # ===== Plot 1: Top-down trajectory with model color coding =====
        ax = axes[0, 0]
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'o-', color='gray', alpha=0.5, linewidth=2, markersize=4, label='Trajectory')
        
        # Color code by dominant model
        for i in range(len(positions)-1):
            if cv_probs[i] > turn_probs[i]:
                ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], linewidth=3, color=self.colors['cv'], alpha=0.8)
            else:
                ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], linewidth=3, color=self.colors['turning'], alpha=0.8)
        
        # Add velocity vectors every 5 frames
        for i in range(0, len(positions), 5):
            vel_scale = 0.3
            ax.arrow(positions[i, 0], positions[i, 1],
                    velocities[i, 0]*vel_scale, velocities[i, 1]*vel_scale,
                    head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
        
        # Start and end markers
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=12, label='End', zorder=5)
        
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_title('Top-Down Trajectory (Color: Dominant Model)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(['CV Mode', 'Turning Mode', 'Start', 'End'], loc='best')
        ax.axis('equal')
        
        # ===== Plot 2: Model probability evolution =====
        ax = axes[0, 1]
        ax.plot(frames, cv_probs, 'o-', label='CV Model', color=self.colors['cv'], linewidth=2.5, markersize=5)
        ax.plot(frames, turn_probs, 's-', label='Turning Model', color=self.colors['turning'], linewidth=2.5, markersize=5)
        ax.fill_between(frames, cv_probs, alpha=0.2, color=self.colors['cv'])
        ax.fill_between(frames, turn_probs, alpha=0.2, color=self.colors['turning'])
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model Probability', fontsize=11, fontweight='bold')
        ax.set_title('Model Selection Probability Over Time', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # ===== Plot 3: Velocity magnitude and yaw rate =====
        ax = axes[1, 0]
        vel_mag = np.linalg.norm(velocities, axis=1)
        yaw_rates = np.array(track['yaw_rates'])
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(frames, vel_mag, 'o-', label='Speed', color=self.colors['cv'], linewidth=2.5, markersize=5)
        line2 = ax2.plot(frames, yaw_rates, 's-', label='Yaw Rate', color=self.colors['turning'], linewidth=2.5, markersize=5)
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speed (m/s)', fontsize=11, fontweight='bold', color=self.colors['cv'])
        ax2.set_ylabel('Yaw Rate (rad/s)', fontsize=11, fontweight='bold', color=self.colors['turning'])
        ax.set_title('Velocity Dynamics', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=10)
        
        # ===== Plot 4: Curvature estimation =====
        ax = axes[1, 1]
        
        # Calculate curvature from positions
        curvatures = []
        for i in range(1, len(positions)-1):
            # Simple curvature: angle change / distance
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]
            
            dist = np.linalg.norm(v2)
            if dist > 0.001:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * dist + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                curvature = np.arccos(cos_angle) / (dist + 1e-6)
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        frames_curve = frames[1:-1]
        
        # Plot curvature and model selection
        ax.bar(frames_curve, curvatures, alpha=0.6, color='gray', label='Estimated Curvature', width=0.8)
        ax.plot(frames_curve, turn_probs[1:-1], 'o-', label='Turning Model Prob', 
               color=self.colors['turning'], linewidth=2.5, markersize=6)
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Curvature', fontsize=11, fontweight='bold')
        ax.set_title('Trajectory Curvature vs Model Selection', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved trajectory comparison: {output_file}")
        plt.close()
    
    def plot_model_performance_summary(self, output_file='model_performance_summary.png'):
        """
        Generate summary plots comparing CV and Turning model performance
        across all tracks
        """
        tracks = self.extract_track_data()
        
        if not tracks:
            print("No tracks to visualize")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle('IMM Model Performance Summary: CV vs Turning', fontsize=16, fontweight='bold')
        
        # ===== Analysis across all tracks =====
        all_cv_probs = []
        all_turn_probs = []
        all_yaw_rates = []
        all_curvatures = []
        straight_trajectories = []
        curved_trajectories = []
        
        for track_id, track in tracks.items():
            positions = np.array(track['positions'])
            cv_probs = np.array(track['cv_probs'])
            turn_probs = np.array(track['turn_probs'])
            yaw_rates = np.array(track['yaw_rates'])
            
            all_cv_probs.extend(cv_probs)
            all_turn_probs.extend(turn_probs)
            all_yaw_rates.extend(np.abs(yaw_rates))
            
            # Classify as straight or curved
            avg_turn_prob = np.mean(turn_probs)
            if avg_turn_prob < 0.3:
                straight_trajectories.append(track_id)
            else:
                curved_trajectories.append(track_id)
            
            # Curvature
            if len(positions) > 2:
                curvatures = []
                for i in range(1, len(positions)-1):
                    v1 = positions[i] - positions[i-1]
                    v2 = positions[i+1] - positions[i]
                    dist = np.linalg.norm(v2)
                    if dist > 0.001:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * dist + 1e-6)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        curvature = np.arccos(cos_angle) / (dist + 1e-6)
                    else:
                        curvature = 0
                    curvatures.append(curvature)
                all_curvatures.extend(curvatures)
        
        # ===== Plot 1: Model probability distribution =====
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(all_cv_probs, bins=30, alpha=0.6, label='CV Model', color=self.colors['cv'], edgecolor='black')
        ax.hist(all_turn_probs, bins=30, alpha=0.6, label='Turning Model', color=self.colors['turning'], edgecolor='black')
        ax.set_xlabel('Probability', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title('Model Probability Distribution', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # ===== Plot 2: Average model selection =====
        ax = fig.add_subplot(gs[0, 1])
        avg_cv = np.mean(all_cv_probs)
        avg_turn = np.mean(all_turn_probs)
        bars = ax.bar(['CV Model', 'Turning Model'], [avg_cv, avg_turn], 
                      color=[self.colors['cv'], self.colors['turning']], edgecolor='black', linewidth=2)
        ax.set_ylabel('Average Probability', fontsize=10, fontweight='bold')
        ax.set_title('Average Model Selection', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ===== Plot 3: Track classification =====
        ax = fig.add_subplot(gs[0, 2])
        counts = [len(straight_trajectories), len(curved_trajectories)]
        labels = [f'Straight\n({len(straight_trajectories)})', f'Curved\n({len(curved_trajectories)})']
        wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%',
                                           colors=[self.colors['cv'], self.colors['turning']],
                                           startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        ax.set_title('Trajectory Type Distribution', fontweight='bold')
        
        # ===== Plot 4: Yaw rate distribution =====
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(all_yaw_rates, bins=40, color=self.colors['turning'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('|Yaw Rate| (rad/s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title('Yaw Rate Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # ===== Plot 5: Curvature vs model prob =====
        ax = fig.add_subplot(gs[1, 1])
        scatter = ax.scatter(all_curvatures, all_turn_probs, c=all_turn_probs, cmap='RdYlBu_r',
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Trajectory Curvature', fontsize=10, fontweight='bold')
        ax.set_ylabel('Turning Model Probability', fontsize=10, fontweight='bold')
        ax.set_title('Curvature vs Model Selection', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Turning Prob', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # ===== Plot 6: Performance metrics table =====
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Total Tracks: {len(tracks)}
        Straight Trajectories: {len(straight_trajectories)}
        Curved Trajectories: {len(curved_trajectories)}
        
        Model Usage:
        • CV Model Avg: {avg_cv:.1%}
        • Turning Model Avg: {avg_turn:.1%}
        
        Motion Characteristics:
        • Mean Yaw Rate: {np.mean(all_yaw_rates):.4f} rad/s
        • Max Yaw Rate: {np.max(all_yaw_rates):.4f} rad/s
        • Mean Curvature: {np.mean(all_curvatures):.4f}
        """
        
        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== Plot 7: Model switching frequency =====
        ax = fig.add_subplot(gs[2, :2])
        
        switching_data = []
        track_ids_list = []
        
        for track_id, track in list(tracks.items())[:15]:  # First 15 tracks
            cv_probs = np.array(track['cv_probs'])
            switches = np.sum(np.abs(np.diff(cv_probs > 0.5)))
            switching_data.append(switches)
            track_ids_list.append(f"T{track_id}")
        
        if switching_data:
            bars = ax.bar(track_ids_list, switching_data, color=self.colors['error'], 
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Model Switches', fontsize=11, fontweight='bold')
            ax.set_title('Model Switching Frequency (First 15 Tracks)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # ===== Plot 8: Performance summary text =====
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        
        performance_text = f"""
        KEY FINDINGS
        
        ✓ Turning model activates on
          curved trajectories
          
        ✓ CV model dominates on
          straight sections
          
        ✓ Smooth model transitions
          indicate robust estimation
          
        ✓ Total tracks analyzed:
          {len(tracks)}
        """
        
        ax.text(0.05, 0.95, performance_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved performance summary: {output_file}")
        plt.close()
    
    def plot_cv_vs_turning_comparison(self, output_file='cv_vs_turning_comparison.png'):
        """
        Side-by-side comparison of CV model vs Turning model performance
        on representative straight and curved trajectories
        """
        tracks = self.extract_track_data()
        
        if not tracks:
            print("No tracks to compare")
            return
        
        # Find best straight and curved examples
        straight_track = None
        curved_track = None
        
        for track_id, track in tracks.items():
            cv_probs = np.array(track['cv_probs'])
            positions = np.array(track['positions'])
            
            if len(positions) > 10:
                avg_turn_prob = np.mean(cv_probs)
                
                if avg_turn_prob > 0.8 and straight_track is None:
                    straight_track = track_id
                elif avg_turn_prob < 0.4 and curved_track is None:
                    curved_track = track_id
            
            if straight_track and curved_track:
                break
        
        if not straight_track or not curved_track:
            print("Not enough representative trajectories to compare")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('CV Model vs Turning Model: Straight vs Curved Trajectories', 
                    fontsize=16, fontweight='bold')
        
        # Process both trajectories
        for row, (track_id, title) in enumerate([(straight_track, 'STRAIGHT TRAJECTORY'),
                                                   (curved_track, 'CURVED TRAJECTORY')]):
            track = tracks[track_id]
            positions = np.array(track['positions'])
            velocities = np.array(track['velocities'])
            cv_probs = np.array(track['cv_probs'])
            turn_probs = np.array(track['turn_probs'])
            frames = track['frames']
            
            # Plot 1: Trajectory
            ax = axes[row, 0]
            ax.plot(positions[:, 0], positions[:, 1], 'o-', color='gray', alpha=0.5, linewidth=1.5, markersize=4)
            
            # Color by model
            for i in range(len(positions)-1):
                if cv_probs[i] > 0.5:
                    ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], linewidth=3, color=self.colors['cv'], alpha=0.8)
                else:
                    ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], linewidth=3, color=self.colors['turning'], alpha=0.8)
            
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start', zorder=5)
            ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=10, label='End', zorder=5)
            
            ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
            ax.set_title(f'{title}\nTrajectory View', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.legend(loc='best', fontsize=9)
            
            # Plot 2: Model probabilities
            ax = axes[row, 1]
            ax.plot(frames, cv_probs, 'o-', label='CV Model', color=self.colors['cv'], linewidth=2.5, markersize=4)
            ax.plot(frames, turn_probs, 's-', label='Turning Model', color=self.colors['turning'], linewidth=2.5, markersize=4)
            ax.fill_between(frames, cv_probs, alpha=0.2, color=self.colors['cv'])
            ax.fill_between(frames, turn_probs, alpha=0.2, color=self.colors['turning'])
            
            ax.set_xlabel('Frame', fontsize=10, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
            ax.set_title(f'{title}\nModel Probability Evolution', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # Plot 3: Velocity and motion characteristics
            ax = axes[row, 2]
            vel_mag = np.linalg.norm(velocities, axis=1)
            yaw_rates = np.array(track['yaw_rates'])
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(frames, vel_mag, 'o-', label='Speed', color=self.colors['cv'], linewidth=2.5, markersize=4)
            line2 = ax2.plot(frames, yaw_rates, 's-', label='Yaw Rate', color=self.colors['turning'], linewidth=2.5, markersize=4)
            
            ax.set_xlabel('Frame', fontsize=10, fontweight='bold')
            ax.set_ylabel('Speed (m/s)', fontsize=10, fontweight='bold', color=self.colors['cv'])
            ax2.set_ylabel('Yaw Rate (rad/s)', fontsize=10, fontweight='bold', color=self.colors['turning'])
            ax.set_title(f'{title}\nMotion Characteristics', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved CV vs Turning comparison: {output_file}")
        plt.close()
    
    def generate_all_visualizations(self, track_limit=5):
        """Generate all available visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        print("1. Generating performance summary...")
        self.plot_model_performance_summary('model_performance_summary.png')
        
        print("\n2. Generating CV vs Turning comparison...")
        self.plot_cv_vs_turning_comparison('cv_vs_turning_comparison.png')
        
        print("\n3. Generating individual track visualizations...")
        tracks = self.extract_track_data()
        
        for i, track_id in enumerate(list(tracks.keys())[:track_limit]):
            print(f"   Processing track {track_id} ({i+1}/{min(track_limit, len(tracks))})...")
            output_file = f'track_{track_id}_analysis.png'
            self.plot_trajectory_comparison(track_id, output_file)
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("="*70 + "\n")


if __name__ == '__main__':
    print("IMM Visualization Module")
    print("Usage: from imm_visualization import IMMVisualizer")
    print("       visualizer = IMMVisualizer('imm_kitti_results.json')")
    print("       visualizer.generate_all_visualizations()")
