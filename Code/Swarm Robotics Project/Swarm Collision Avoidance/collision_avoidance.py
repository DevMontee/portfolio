"""
Swarm Collision Avoidance Module
Decentralized collision layer with repulsive potentials between agents.

Location: D:\Swarm Robotics Project\Swarm Collision Avoidance\
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Link to Setup + Formation Layer
SETUP_DIR = r"D:\Swarm Robotics Project\Setup + Formation Layer"
if SETUP_DIR not in sys.path:
    sys.path.insert(0, SETUP_DIR)


class CollisionAvoidanceLayer:
    """
    Decentralized collision avoidance using repulsive potential fields.
    Each agent computes repulsive forces based on neighbors.
    """
    
    def __init__(self,
                 agent_radius: float = 0.2,
                 safety_distance: float = 0.5,
                 repulsion_gain: float = 10.0,
                 repulsion_power: float = 2.0):
        """
        Initialize collision avoidance layer.
        
        Args:
            agent_radius: Physical radius of each agent (m)
            safety_distance: Desired minimum distance between agents (m)
            repulsion_gain: Strength of repulsive force
            repulsion_power: Power law for distance decay (2 = inverse square)
        """
        self.agent_radius = agent_radius
        self.safety_distance = safety_distance
        self.repulsion_gain = repulsion_gain
        self.repulsion_power = repulsion_power
        
        # Statistics
        self.collision_events = 0
        self.avoidance_activations = 0
    
    def compute_repulsive_force(self,
                               agent_pos: np.ndarray,
                               agent_vel: np.ndarray,
                               neighbor_positions: np.ndarray,
                               neighbor_velocities: np.ndarray,
                               target_pos: np.ndarray = None,
                               initial_distance_to_target: float = None) -> np.ndarray:
        """
        Compute decentralized repulsive force for one agent.
        Includes proximity-based decay: reduces repulsion as agent nears target.
        
        Args:
            agent_pos: Agent position (3,)
            agent_vel: Agent velocity (3,)
            neighbor_positions: Other agents' positions (N, 3)
            neighbor_velocities: Other agents' velocities (N, 3)
            target_pos: Agent's target position (for proximity decay)
            initial_distance_to_target: Initial distance to target (for normalization)
            
        Returns:
            Repulsive force vector (3,) to apply to agent
        """
        repulsive_force = np.zeros(3)
        
        if len(neighbor_positions) == 0:
            return repulsive_force
        
        # Calculate proximity-based decay factor
        decay_factor = 1.0
        if target_pos is not None and initial_distance_to_target is not None and initial_distance_to_target > 1e-6:
            distance_to_target = np.linalg.norm(agent_pos - target_pos)
            # Normalize: 1.0 at start, approaches 0.0 as agent reaches target
            progress = 1.0 - (distance_to_target / initial_distance_to_target)
            progress = np.clip(progress, 0.0, 1.0)
            # ULTRA-AGGRESSIVE decay - cubic function suppresses oscillation near target
            # This prevents agents from bouncing around at convergence
            decay_factor = (1.0 - progress) ** 3.0  # Ranges from 1.0 to ~0.001
        
        min_collision_dist = 2 * self.agent_radius
        
        for i, neighbor_pos in enumerate(neighbor_positions):
            # Vector from neighbor to agent
            diff = agent_pos - neighbor_pos
            distance = np.linalg.norm(diff)
            
            # Skip if too far
            if distance > self.safety_distance or distance < 1e-6:
                continue
            
            # Normalize direction
            direction = diff / distance
            
            # Repulsive force magnitude
            # Stronger when closer, decays with distance
            if distance < self.safety_distance:
                # Inverse power law: F ∝ 1/d^n
                magnitude = self.repulsion_gain * (self.safety_distance / distance) ** self.repulsion_power
                
                # Apply proximity-based decay
                magnitude *= decay_factor
                
                # Increase magnitude if collision imminent (but still respect decay)
                if distance < min_collision_dist:
                    magnitude *= 2.0
                    self.collision_events += 1
                
                # Apply directional force
                repulsive_force += magnitude * direction
                self.avoidance_activations += 1
        
        return repulsive_force
    
    def check_collision(self,
                       agent_pos: np.ndarray,
                       neighbor_positions: np.ndarray) -> bool:
        """
        Check if agent is in collision.
        
        Args:
            agent_pos: Agent position
            neighbor_positions: Other agents' positions
            
        Returns:
            True if collision detected
        """
        min_dist = 2 * self.agent_radius
        
        for neighbor_pos in neighbor_positions:
            distance = np.linalg.norm(agent_pos - neighbor_pos)
            if distance < min_dist:
                return True
        
        return False
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.collision_events = 0
        self.avoidance_activations = 0
    
    def get_stats(self) -> Dict:
        """Get collision avoidance statistics."""
        return {
            'collision_events': self.collision_events,
            'avoidance_activations': self.avoidance_activations
        }


class SwarmEnvironmentWithAvoidance:
    """
    Extended SwarmEnvironment with collision avoidance layer.
    Wraps existing SwarmEnvironment and adds decentralized collision handling.
    """
    
    def __init__(self, 
                 swarm_env,
                 enable_collision_avoidance: bool = True,
                 avoidance_gain: float = 10.0,
                 safety_distance: float = 0.5):
        """
        Initialize swarm with collision avoidance.
        
        Args:
            swarm_env: Existing SwarmEnvironment instance
            enable_collision_avoidance: Enable/disable avoidance layer
            avoidance_gain: Strength of repulsive forces
            safety_distance: Minimum safe distance between agents
        """
        self.swarm_env = swarm_env
        self.agents = swarm_env.agents
        self.physics_client = swarm_env.physics_client
        self.time_step = swarm_env.time_step
        
        self.enable_collision_avoidance = enable_collision_avoidance
        
        # Store initial distances for proximity decay (CRITICAL FIX)
        self.initial_distances_to_target = {}
        self.simulation_started = False
        
        self.avoidance = CollisionAvoidanceLayer(
            agent_radius=0.2,
            safety_distance=safety_distance,
            repulsion_gain=avoidance_gain,
            repulsion_power=2.0
        )
    
    def initialize_target_distances(self):
        """
        Initialize and store the starting distance from each agent to its target.
        MUST be called once before run_until_converged_with_avoidance().
        """
        for agent in self.agents:
            initial_dist = np.linalg.norm(agent.target_position - agent.position)
            self.initial_distances_to_target[agent.agent_id] = initial_dist if initial_dist > 1e-6 else 1.0
        self.simulation_started = True
    
    def step_with_avoidance(self):
        """
        Execute one simulation step WITH collision avoidance and proximity decay.
        """
        if not self.enable_collision_avoidance:
            self.swarm_env.step()
            return
        
        # Apply collision avoidance to all agents
        positions = np.array([agent.position for agent in self.agents])
        velocities = np.array([agent.velocity for agent in self.agents])
        
        # Compute repulsive forces for each agent
        repulsive_forces = []
        for i, agent in enumerate(self.agents):
            neighbor_indices = [j for j in range(len(self.agents)) if j != i]
            neighbor_pos = positions[neighbor_indices]
            neighbor_vel = velocities[neighbor_indices]
            
            # CRITICAL FIX: Use STORED initial distance, not current distance
            initial_dist = self.initial_distances_to_target.get(agent.agent_id, 1.0)
            
            # Compute repulsive force with proximity decay
            f_repul = self.avoidance.compute_repulsive_force(
                agent.position,
                agent.velocity,
                neighbor_pos,
                neighbor_vel,
                target_pos=agent.target_position,
                initial_distance_to_target=initial_dist
            )
            repulsive_forces.append(f_repul)
        
        # Store repulsive forces temporarily
        for agent, f_repul in zip(self.agents, repulsive_forces):
            agent.repulsive_force = f_repul
        
        # Apply control WITH repulsive forces
        for agent in self.agents:
            self._apply_control_with_avoidance(agent)
        
        # Step physics
        import pybullet as p
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # Update agent states
        for agent in self.agents:
            agent.update_state(self.physics_client)
    
    def _apply_control_with_avoidance(self, agent):
        """
        Apply PD control + collision avoidance to agent.
        """
        import pybullet as p
        
        if agent.body_id is None:
            return
        
        # PD control gains
        kp = 10.0
        kd = 5.0
        
        # Position error
        error = agent.target_position - agent.position
        desired_velocity = kp * error
        
        # Limit desired velocity
        speed = np.linalg.norm(desired_velocity)
        if speed > agent.max_velocity:
            desired_velocity = desired_velocity / speed * agent.max_velocity
        
        velocity_error = desired_velocity - agent.velocity
        
        # Control force
        force = kp * error + kd * velocity_error
        
        # ADD COLLISION AVOIDANCE FORCE
        if hasattr(agent, 'repulsive_force'):
            force += agent.repulsive_force
        
        # Limit acceleration
        acceleration = force / agent.mass
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > agent.max_acceleration:
            force = force / accel_magnitude * agent.max_acceleration * agent.mass
        
        # Apply force (with gravity compensation)
        gravity_compensation = np.array([0, 0, 9.81 * agent.mass])
        total_force = force + gravity_compensation
        
        p.applyExternalForce(
            agent.body_id,
            -1,
            total_force,
            agent.position,
            p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )
    
    def run_until_converged_with_avoidance(self,
                                           max_steps: int = 5000,
                                           tolerance: float = 0.15,
                                           check_interval: int = 100) -> bool:
        """
        Run simulation with collision avoidance + proximity decay until convergence.
        Proximity decay reduces repulsion as agents approach targets.
        """
        import time
        
        # CRITICAL: Initialize target distances on first run
        if not self.simulation_started:
            self.initialize_target_distances()
        
        print(f"Running simulation with collision avoidance + proximity decay (max {max_steps} steps)...")
        
        for step in range(max_steps):
            self.step_with_avoidance()
            
            # Check convergence periodically
            if step % check_interval == 0:
                converged = all(agent.at_target(tolerance) for agent in self.agents)
                if converged:
                    print(f"✓ Converged at step {step}")
                    return True
                
                # Progress update
                distances = [np.linalg.norm(agent.position - agent.target_position) 
                           for agent in self.agents]
                avg_distance = np.mean(distances)
                max_distance = np.max(distances)
                
                # Get avoidance stats
                stats = self.avoidance.get_stats()
                print(f"Step {step}: avg_dist={avg_distance:.3f}m, "
                      f"decay_active=enabled, avoidance_activations={stats['avoidance_activations']}")
            
            time.sleep(self.time_step / 10)  # Real-time visualization
        
        print(f"⚠ Timeout after {max_steps} steps")
        return False
    
    def get_avoidance_metrics(self) -> Dict:
        """Get collision avoidance metrics."""
        return self.avoidance.get_stats()
    
    def check_collisions(self, min_distance: float = 0.3) -> int:
        """Check for actual collisions."""
        return self.swarm_env.check_collisions(min_distance)
    
    def get_metrics(self) -> Dict:
        """Get all metrics including avoidance."""
        metrics = self.swarm_env.get_metrics()
        avoidance_metrics = self.avoidance.get_stats()
        
        return {
            **metrics,
            'avoidance_activations': avoidance_metrics['avoidance_activations'],
            'collision_events': avoidance_metrics['collision_events']
        }
    
    def close(self):
        """Close simulation."""
        self.swarm_env.close()


def visualize_avoidance_performance(results: List[Dict], save_path: str = None):
    """
    Visualize collision avoidance performance comparison.
    
    Args:
        results: List of result dictionaries (with/without avoidance)
        save_path: Save figure to file
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    labels = [r['label'] for r in results]
    convergence_rates = [r['metrics']['convergence_rate'] for r in results]
    final_errors = [r['metrics']['avg_final_error'] for r in results]
    collisions = [r['metrics'].get('collision_events', 0) for r in results]
    sim_times = [r['sim_time'] for r in results]
    
    # Convergence rate
    ax = axes[0, 0]
    ax.bar(labels, [c*100 for c in convergence_rates], color=['red', 'green'])
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title('Convergence Rate Comparison')
    ax.set_ylim([0, 105])
    
    # Final errors
    ax = axes[0, 1]
    ax.bar(labels, final_errors, color=['red', 'green'])
    ax.set_ylabel('Avg Final Error (m)')
    ax.set_title('Position Error Comparison')
    
    # Collision events
    ax = axes[1, 0]
    ax.bar(labels, collisions, color=['red', 'green'])
    ax.set_ylabel('Collision Events')
    ax.set_title('Collision Detection')
    
    # Simulation time
    ax = axes[1, 1]
    ax.bar(labels, sim_times, color=['red', 'green'])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Simulation Duration')
    
    plt.tight_layout()
    
    if save_path:
        save_path = os.path.normpath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Collision Avoidance Module")
    print("Import in your code: from collision_avoidance import SwarmEnvironmentWithAvoidance")
