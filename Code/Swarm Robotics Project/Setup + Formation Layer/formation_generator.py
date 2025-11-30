"""
Formation Generator Module
Converts geometric shape specifications into waypoint clouds for swarm agents.
Flat structure version - no package imports.

Updated for Windows path structure and package organization.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class FormationGenerator:
    """Generate waypoint clouds for various geometric formations."""
    
    def __init__(self, num_agents: int = 20, scale: float = 1.0):
        """
        Initialize formation generator.
        
        Args:
            num_agents: Number of agents in the swarm
            scale: Global scale factor for all formations
        """
        self.num_agents = num_agents
        self.scale = scale
        self.waypoints = None
        self.formation_name = None
        
    def generate_formation(self, formation_type: str, **kwargs) -> np.ndarray:
        """
        Generate waypoints for specified formation type.
        
        Args:
            formation_type: One of ['cupid', 'dragon', 'flag', 'helix', 'mandala']
            **kwargs: Formation-specific parameters
            
        Returns:
            np.ndarray: (N, 3) array of waypoint positions [x, y, z]
        """
        formation_map = {
            'cupid': self._generate_cupid,
            'dragon': self._generate_dragon,
            'flag': self._generate_canadian_flag,
            'helix': self._generate_helix,
            'mandala': self._generate_mandala
        }
        
        if formation_type.lower() not in formation_map:
            raise ValueError(f"Unknown formation type: {formation_type}")
        
        self.formation_name = formation_type
        self.waypoints = formation_map[formation_type.lower()](**kwargs)
        
        return self.waypoints
    
    def _generate_cupid(self, height: float = 5.0) -> np.ndarray:
        """Generate heart-shaped (Cupid) formation."""
        waypoints = []
        t = np.linspace(0, 2 * np.pi, self.num_agents)
        
        for ti in t:
            x = 16 * np.sin(ti)**3
            y = 13 * np.cos(ti) - 5 * np.cos(2*ti) - 2 * np.cos(3*ti) - np.cos(4*ti)
            z = height
            waypoints.append([x * self.scale * 0.1, y * self.scale * 0.1, z * self.scale])
        
        return np.array(waypoints)
    
    def _generate_dragon(self, height_range: Tuple[float, float] = (2.0, 8.0)) -> np.ndarray:
        """Generate dragon-shaped formation using sinusoidal body with wings."""
        waypoints = []
        
        n_body = int(0.6 * self.num_agents)
        n_wing = (self.num_agents - n_body) // 2
        
        t_body = np.linspace(0, 4 * np.pi, n_body)
        for i, t in enumerate(t_body):
            x = t * 0.5
            y = 2 * np.sin(t)
            z = height_range[0] + (height_range[1] - height_range[0]) * (i / n_body)
            waypoints.append([x * self.scale, y * self.scale, z * self.scale])
        
        wing_x = 3.0
        wing_y = np.linspace(-2, 2, n_wing)
        wing_z = 5.0
        for y in wing_y:
            waypoints.append([wing_x * self.scale, y * self.scale, wing_z * self.scale])
        
        for y in wing_y:
            waypoints.append([wing_x * self.scale, -y * self.scale, wing_z * self.scale])
        
        return np.array(waypoints)
    
    def _generate_canadian_flag(self, height: float = 5.0) -> np.ndarray:
        """Generate Canadian flag formation (simplified maple leaf + bars)."""
        waypoints = []
        n_per_bar = self.num_agents // 5
        
        for i in range(n_per_bar):
            x = -3.0
            y = -2 + 4 * (i / n_per_bar)
            z = height
            waypoints.append([x * self.scale, y * self.scale, z * self.scale])
        
        for i in range(n_per_bar):
            x = 3.0
            y = -2 + 4 * (i / n_per_bar)
            z = height
            waypoints.append([x * self.scale, y * self.scale, z * self.scale])
        
        n_leaf = self.num_agents - 2 * n_per_bar
        theta = np.linspace(0, 2 * np.pi, n_leaf)
        
        for t in theta:
            r = 1.5 * (1 + 0.3 * np.sin(5 * t))
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height
            waypoints.append([x * self.scale, y * self.scale, z * self.scale])
        
        return np.array(waypoints)
    
    def _generate_helix(self, 
                       height_range: Tuple[float, float] = (0.0, 10.0),
                       radius: float = 2.0,
                       turns: float = 3.0) -> np.ndarray:
        """Generate helical/spiral formation for 3D coordination testing."""
        waypoints = []
        t = np.linspace(0, turns * 2 * np.pi, self.num_agents)
        
        for i, ti in enumerate(t):
            x = radius * np.cos(ti)
            y = radius * np.sin(ti)
            z = height_range[0] + (height_range[1] - height_range[0]) * (i / self.num_agents)
            waypoints.append([x * self.scale, y * self.scale, z * self.scale])
        
        return np.array(waypoints)
    
    def _generate_mandala(self, 
                         height: float = 5.0,
                         num_rings: int = 3) -> np.ndarray:
        """Generate mandala formation with concentric circular symmetry."""
        waypoints = []
        agents_per_ring = self.num_agents // num_rings
        remainder = self.num_agents % num_rings
        
        for ring in range(num_rings):
            n_agents = agents_per_ring + (1 if ring < remainder else 0)
            radius = (ring + 1) * 1.5
            theta = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
            
            for t in theta:
                x = radius * np.cos(t)
                y = radius * np.sin(t)
                z = height
                waypoints.append([x * self.scale, y * self.scale, z * self.scale])
        
        return np.array(waypoints)
    
    def visualize(self, save_path: str = None):
        """Visualize the generated formation in 3D."""
        if self.waypoints is None:
            raise ValueError("No formation generated yet. Call generate_formation() first.")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.waypoints[:, 0], 
                  self.waypoints[:, 1], 
                  self.waypoints[:, 2],
                  c=np.arange(len(self.waypoints)),
                  cmap='viridis',
                  s=100,
                  alpha=0.8)
        
        for i, point in enumerate(self.waypoints):
            ax.text(point[0], point[1], point[2], str(i), fontsize=8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{self.formation_name.capitalize()} Formation ({self.num_agents} agents)')
        
        max_range = np.array([
            self.waypoints[:, 0].max() - self.waypoints[:, 0].min(),
            self.waypoints[:, 1].max() - self.waypoints[:, 1].min(),
            self.waypoints[:, 2].max() - self.waypoints[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (self.waypoints[:, 0].max() + self.waypoints[:, 0].min()) * 0.5
        mid_y = (self.waypoints[:, 1].max() + self.waypoints[:, 1].min()) * 0.5
        mid_z = (self.waypoints[:, 2].max() + self.waypoints[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            # Handle Windows paths
            save_path = os.path.normpath(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Formation visualization saved to {save_path}")
        
        plt.show()
    
    def get_formation_info(self) -> Dict:
        """Return information about the current formation."""
        if self.waypoints is None:
            return {}
        
        return {
            'name': self.formation_name,
            'num_agents': len(self.waypoints),
            'scale': self.scale,
            'bounds': {
                'x': (self.waypoints[:, 0].min(), self.waypoints[:, 0].max()),
                'y': (self.waypoints[:, 1].min(), self.waypoints[:, 1].max()),
                'z': (self.waypoints[:, 2].min(), self.waypoints[:, 2].max())
            },
            'centroid': self.waypoints.mean(axis=0),
            'spread': np.std(self.waypoints, axis=0)
        }
