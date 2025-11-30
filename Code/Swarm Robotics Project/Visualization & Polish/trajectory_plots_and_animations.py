"""
Publication-Ready Trajectory Plots & Swarm Animations
Generates high-quality visualizations for papers/presentations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime


class TrajectoryVisualizer:
    """Generate publication-ready trajectory plots and animations."""
    
    @staticmethod
    def plot_3d_trajectories(agent_positions: dict,
                            formation_name: str,
                            save_path: str = None,
                            title: str = None):
        """
        Create publication-quality 3D trajectory plot.
        
        Args:
            agent_positions: {agent_id: (N, 3) positions array}
            formation_name: Name of formation
            save_path: Path to save PNG
            title: Optional custom title
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for agents
        colors = plt.cm.tab20(np.linspace(0, 1, len(agent_positions)))
        
        # Plot trajectories
        for idx, (agent_id, positions) in enumerate(agent_positions.items()):
            ax.plot(positions[:, 0], 
                   positions[:, 1], 
                   positions[:, 2],
                   linewidth=2,
                   label=f'Agent {agent_id}',
                   color=colors[idx],
                   alpha=0.8)
            
            # Start point
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                      marker='o', s=100, color=colors[idx], edgecolors='black', linewidth=2)
            
            # End point
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                      marker='*', s=500, color=colors[idx], edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
        ax.set_title(title or f'{formation_name.upper()} Formation - 3D Trajectories',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.95)
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 3D trajectory plot: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_2d_trajectories(agent_positions: dict,
                            formation_name: str,
                            plane: str = 'xy',
                            save_path: str = None):
        """
        Create publication-quality 2D trajectory projection.
        
        Args:
            agent_positions: {agent_id: (N, 3) positions array}
            formation_name: Name of formation
            plane: 'xy', 'xz', or 'yz'
            save_path: Path to save PNG
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(agent_positions)))
        
        # Map plane to axes
        axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        ax_idx = axis_map.get(plane, (0, 1))
        axis_labels = {'xy': ('X (m)', 'Y (m)'),
                      'xz': ('X (m)', 'Z (m)'),
                      'yz': ('Y (m)', 'Z (m)')}
        
        for idx, (agent_id, positions) in enumerate(agent_positions.items()):
            # Trajectory line
            ax.plot(positions[:, ax_idx[0]], 
                   positions[:, ax_idx[1]],
                   linewidth=2.5,
                   label=f'Agent {agent_id}',
                   color=colors[idx],
                   alpha=0.8,
                   marker='.',
                   markersize=4)
            
            # Start point (circle)
            ax.scatter(positions[0, ax_idx[0]], positions[0, ax_idx[1]],
                      marker='o', s=150, color=colors[idx], 
                      edgecolors='black', linewidth=2.5, zorder=5,
                      label=f'Agent {agent_id} Start' if idx < 5 else '')
            
            # End point (star)
            ax.scatter(positions[-1, ax_idx[0]], positions[-1, ax_idx[1]],
                      marker='*', s=600, color=colors[idx], 
                      edgecolors='black', linewidth=2, zorder=5)
        
        ax.set_xlabel(axis_labels[plane][0], fontsize=13, fontweight='bold')
        ax.set_ylabel(axis_labels[plane][1], fontsize=13, fontweight='bold')
        ax.set_title(f'{formation_name.upper()} Formation - {plane.upper()} Plane Projection',
                    fontsize=14, fontweight='bold', pad=15)
        
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 2D trajectory plot ({plane}): {save_path}")
        
        return fig
    
    @staticmethod
    def animate_swarm_convergence(agent_positions: dict,
                                 formation_name: str,
                                 save_path: str = None,
                                 interval: int = 50):
        """
        Create animated GIF of swarm converging to formation.
        
        Args:
            agent_positions: {agent_id: (N, 3) positions array}
            formation_name: Name of formation
            save_path: Path to save GIF
            interval: Milliseconds between frames
        """
        # Prepare data
        num_steps = max(pos.shape[0] for pos in agent_positions.values())
        agent_ids = sorted(agent_positions.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(agent_ids)))
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        all_positions = np.vstack([agent_positions[aid] for aid in agent_ids])
        x_lim = [all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1]
        y_lim = [all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1]
        z_lim = [all_positions[:, 2].min() - 1, all_positions[:, 2].max() + 1]
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
        
        # Line objects for each agent
        lines = []
        scatter = ax.scatter([], [], [], s=100, c=[], cmap='tab20', edgecolors='black')
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
            ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
            
            title = f'{formation_name.upper()} Formation Convergence - Step {frame}/{num_steps}'
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            
            # Plot trajectories up to current frame
            for idx, agent_id in enumerate(agent_ids):
                positions = agent_positions[agent_id]
                max_step = min(frame + 1, len(positions))
                
                # Trajectory line
                if max_step > 1:
                    ax.plot(positions[:max_step, 0],
                           positions[:max_step, 1],
                           positions[:max_step, 2],
                           color=colors[idx],
                           linewidth=2,
                           alpha=0.7)
                
                # Current position
                if max_step > 0:
                    ax.scatter(positions[max_step-1, 0],
                              positions[max_step-1, 1],
                              positions[max_step-1, 2],
                              color=colors[idx],
                              s=150,
                              edgecolors='black',
                              linewidth=2,
                              zorder=5)
            
            ax.view_init(elev=20, azim=frame % 360)
            return ax,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=num_steps,
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            print(f"Creating animation... this may take a minute...")
            writer = PillowWriter(fps=20)
            anim.save(save_path, writer=writer)
            print(f"✓ Saved swarm animation: {save_path}")
        
        return anim
    
    @staticmethod
    def plot_convergence_heatmap(agent_errors: dict,
                                formation_name: str,
                                save_path: str = None):
        """
        Create heatmap of convergence over time.
        
        Args:
            agent_errors: {agent_id: (N,) error array}
            formation_name: Name of formation
            save_path: Path to save PNG
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        agent_ids = sorted(agent_errors.keys())
        max_steps = max(errors.shape[0] for errors in agent_errors.values())
        
        # Create matrix (pad if needed)
        heatmap_data = np.zeros((len(agent_ids), max_steps))
        for idx, agent_id in enumerate(agent_ids):
            errors = agent_errors[agent_id]
            heatmap_data[idx, :len(errors)] = errors
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', 
                       interpolation='bilinear')
        
        ax.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Agent ID', fontsize=12, fontweight='bold')
        ax.set_title(f'{formation_name.upper()} - Convergence Heatmap (Error Over Time)',
                    fontsize=13, fontweight='bold', pad=15)
        
        # Set y-ticks
        ax.set_yticks(range(len(agent_ids)))
        ax.set_yticklabels(agent_ids)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Position Error (m)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved convergence heatmap: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_distance_evolution(agent_positions: dict,
                               formation_name: str,
                               save_path: str = None):
        """
        Plot how inter-agent distances evolve over time.
        
        Args:
            agent_positions: {agent_id: (N, 3) positions array}
            formation_name: Name of formation
            save_path: Path to save PNG
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        agent_ids = sorted(agent_positions.keys())
        num_steps = max(pos.shape[0] for pos in agent_positions.values())
        
        # Calculate pairwise distances
        min_distances = []
        avg_distances = []
        max_distances = []
        
        for step in range(num_steps):
            distances = []
            for i, aid1 in enumerate(agent_ids):
                if step < agent_positions[aid1].shape[0]:
                    pos1 = agent_positions[aid1][step]
                    for aid2 in agent_ids[i+1:]:
                        if step < agent_positions[aid2].shape[0]:
                            pos2 = agent_positions[aid2][step]
                            dist = np.linalg.norm(pos1 - pos2)
                            distances.append(dist)
            
            if distances:
                min_distances.append(np.min(distances))
                avg_distances.append(np.mean(distances))
                max_distances.append(np.max(distances))
        
        steps = np.arange(len(min_distances))
        
        # Plot 1: Distance envelope
        ax1.fill_between(steps, min_distances, max_distances, alpha=0.3, color='blue', label='Min-Max Range')
        ax1.plot(steps, avg_distances, 'b-', linewidth=2.5, label='Average Distance')
        ax1.axhline(y=0.35, color='red', linestyle='--', linewidth=2, label='Safety Threshold (0.35m)')
        
        ax1.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Inter-Agent Distance (m)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{formation_name.upper()} - Distance Evolution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='best')
        
        # Plot 2: Minimum distance detail
        ax2.plot(steps, min_distances, 'r-', linewidth=2.5, label='Minimum Distance')
        ax2.axhline(y=0.35, color='orange', linestyle='--', linewidth=2, label='Safety Threshold')
        ax2.fill_between(steps, 0, min_distances, where=(np.array(min_distances) < 0.35),
                         alpha=0.3, color='red', label='Below Threshold')
        
        ax2.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Minimum Distance (m)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{formation_name.upper()} - Minimum Distance Detail', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distance evolution plot: {save_path}")
        
        return fig


# Example usage
if __name__ == "__main__":
    print("Trajectory Plotting and Animation Module")
    print("Use with your simulation results to generate publication-ready visualizations")
    print("\nExample:")
    print("""
    visualizer = TrajectoryVisualizer()
    
    # 3D trajectories
    visualizer.plot_3d_trajectories(
        agent_positions={0: positions_array, 1: positions_array, ...},
        formation_name='helix',
        save_path='helix_3d_trajectories.png'
    )
    
    # 2D projections
    visualizer.plot_2d_trajectories(
        agent_positions={...},
        formation_name='helix',
        plane='xy',
        save_path='helix_xy_projection.png'
    )
    
    # Animation
    visualizer.animate_swarm_convergence(
        agent_positions={...},
        formation_name='helix',
        save_path='helix_convergence.gif'
    )
    
    # Convergence heatmap
    visualizer.plot_convergence_heatmap(
        agent_errors={0: error_array, ...},
        formation_name='helix',
        save_path='helix_convergence_heatmap.png'
    )
    
    # Distance evolution
    visualizer.plot_distance_evolution(
        agent_positions={...},
        formation_name='helix',
        save_path='helix_distance_evolution.png'
    )
    """)
