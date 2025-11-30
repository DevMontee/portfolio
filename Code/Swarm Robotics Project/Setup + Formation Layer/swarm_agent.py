"""
Swarm Agent Module
PyBullet-based quadrotor agent with physics simulation.
Supports multi-agent swarm with collision detection and dynamics.

Updated for Windows path structure and package organization.
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Tuple, Optional
import time
import os


class QuadrotorAgent:
    """
    Simplified quadrotor agent for swarm simulation.
    Uses PyBullet for physics, simplified as a sphere for computational efficiency.
    """
    
    def __init__(self, 
                 agent_id: int,
                 start_pos: np.ndarray,
                 radius: float = 0.15,
                 mass: float = 0.5,
                 color: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize quadrotor agent.
        
        Args:
            agent_id: Unique identifier for this agent
            start_pos: Initial position [x, y, z]
            radius: Agent collision radius (m)
            mass: Agent mass (kg)
            color: RGBA color tuple (optional)
        """
        self.agent_id = agent_id
        self.radius = radius
        self.mass = mass
        self.body_id = None
        
        # State variables
        self.position = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(3)
        self.target_position = np.array(start_pos, dtype=float)
        
        # Control parameters
        self.max_velocity = 5.0  # m/s
        self.max_acceleration = 10.0  # m/s²
        
        # Energy tracking
        self.energy_consumed = 0.0
        self.distance_traveled = 0.0
        
        # Color assignment (rainbow if not specified)
        if color is None:
            hue = agent_id / 20.0
            self.color = self._hue_to_rgb(hue) + (1.0,)
        else:
            self.color = color
    
    def create_in_simulation(self, physics_client: int):
        """
        Create agent body in PyBullet simulation.
        
        Args:
            physics_client: PyBullet physics client ID
        """
        # Create sphere collision shape
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius,
            physicsClientId=physics_client
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius,
            rgbaColor=self.color,
            physicsClientId=physics_client
        )
        
        # Create multi-body
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.position,
            physicsClientId=physics_client
        )
        
        # Set dynamics properties
        p.changeDynamics(
            self.body_id,
            -1,
            linearDamping=0.5,
            angularDamping=0.5,
            physicsClientId=physics_client
        )
    
    def set_target(self, target_pos: np.ndarray):
        """Set target position for this agent."""
        self.target_position = np.array(target_pos, dtype=float)
    
    def update_state(self, physics_client: int):
        """
        Update agent state from PyBullet simulation.
        
        Args:
            physics_client: PyBullet physics client ID
        """
        if self.body_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=physics_client)
            vel, _ = p.getBaseVelocity(self.body_id, physicsClientId=physics_client)
            
            old_position = self.position.copy()
            self.position = np.array(pos)
            self.velocity = np.array(vel)
            
            # Update energy and distance
            displacement = np.linalg.norm(self.position - old_position)
            self.distance_traveled += displacement
            
            # Simplified energy model: E = m * v^2 / 2 * dt
            speed = np.linalg.norm(self.velocity)
            self.energy_consumed += 0.5 * self.mass * speed**2 * 0.01  # Assuming dt = 0.01
    
    def apply_control(self, physics_client: int, dt: float = 0.01):
        """
        Apply simple PD control to reach target position.
        
        Args:
            physics_client: PyBullet physics client ID
            dt: Time step (s)
        """
        if self.body_id is None:
            return
        
        # PD control gains
        kp = 10.0
        kd = 5.0
        
        # Position error
        error = self.target_position - self.position
        distance_to_target = np.linalg.norm(error)
        
        # Velocity error
        desired_velocity = kp * error
        
        # Limit desired velocity
        speed = np.linalg.norm(desired_velocity)
        if speed > self.max_velocity:
            desired_velocity = desired_velocity / speed * self.max_velocity
        
        velocity_error = desired_velocity - self.velocity
        
        # Control force
        force = kp * error + kd * velocity_error
        
        # Limit acceleration
        acceleration = force / self.mass
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > self.max_acceleration:
            force = force / accel_magnitude * self.max_acceleration * self.mass
        
        # Apply force (with gravity compensation)
        gravity_compensation = np.array([0, 0, 9.81 * self.mass])
        total_force = force + gravity_compensation
        
        p.applyExternalForce(
            self.body_id,
            -1,
            total_force,
            self.position,
            p.WORLD_FRAME,
            physicsClientId=physics_client
        )
    
    def at_target(self, tolerance: float = 0.1) -> bool:
        """
        Check if agent has reached target position.
        
        Args:
            tolerance: Position tolerance (m)
            
        Returns:
            True if within tolerance of target
        """
        return np.linalg.norm(self.position - self.target_position) < tolerance
    
    @staticmethod
    def _hue_to_rgb(hue: float) -> Tuple[float, float, float]:
        """Convert hue (0-1) to RGB color."""
        import colorsys
        return colorsys.hsv_to_rgb(hue, 0.8, 0.9)


class SwarmEnvironment:
    """
    PyBullet environment for multi-agent swarm simulation.
    Manages physics, visualization, and agent coordination.
    """
    
    def __init__(self, 
                 num_agents: int = 20,
                 gui: bool = True,
                 time_step: float = 0.01):
        """
        Initialize swarm environment.
        
        Args:
            num_agents: Number of agents in swarm
            gui: Enable PyBullet GUI
            time_step: Physics simulation time step (s)
        """
        self.num_agents = num_agents
        self.time_step = time_step
        self.agents: List[QuadrotorAgent] = []
        
        # Initialize PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Setup simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # Camera setup for GUI
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 5],
                physicsClientId=self.physics_client
            )
        
        print(f"SwarmEnvironment initialized with {num_agents} agents")
    
    def initialize_agents(self, start_positions: np.ndarray):
        """
        Create and initialize all agents in the swarm.
        
        Args:
            start_positions: (N, 3) array of starting positions
        """
        if len(start_positions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} positions, got {len(start_positions)}")
        
        self.agents = []
        for i, pos in enumerate(start_positions):
            agent = QuadrotorAgent(agent_id=i, start_pos=pos)
            agent.create_in_simulation(self.physics_client)
            self.agents.append(agent)
        
        print(f"Created {len(self.agents)} agents in simulation")
    
    def set_formation_targets(self, target_positions: np.ndarray):
        """
        Set target positions for all agents.
        
        Args:
            target_positions: (N, 3) array of target positions
        """
        if len(target_positions) != len(self.agents):
            raise ValueError(f"Number of targets ({len(target_positions)}) doesn't match agents ({len(self.agents)})")
        
        for agent, target in zip(self.agents, target_positions):
            agent.set_target(target)
    
    def step(self):
        """Execute one simulation step."""
        # Apply control for all agents
        for agent in self.agents:
            agent.apply_control(self.physics_client, self.time_step)
        
        # Step physics
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # Update agent states
        for agent in self.agents:
            agent.update_state(self.physics_client)
    
    def run_until_converged(self, 
                           max_steps: int = 5000,
                           tolerance: float = 0.15,
                           check_interval: int = 100) -> bool:
        """
        Run simulation until all agents reach targets or timeout.
        
        Args:
            max_steps: Maximum simulation steps
            tolerance: Position tolerance for convergence (m)
            check_interval: Steps between convergence checks
            
        Returns:
            True if converged, False if timed out
        """
        print(f"Running simulation (max {max_steps} steps)...")
        
        for step in range(max_steps):
            self.step()
            
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
                print(f"Step {step}: avg_dist={avg_distance:.3f}m, max_dist={max_distance:.3f}m")
            
            # Real-time visualization delay
            time.sleep(self.time_step / 10)  # 10x speedup
        
        print(f"⚠ Timeout after {max_steps} steps")
        return False
    
    def get_metrics(self) -> dict:
        """
        Calculate performance metrics for the swarm.
        
        Returns:
            Dictionary of metrics
        """
        total_energy = sum(agent.energy_consumed for agent in self.agents)
        total_distance = sum(agent.distance_traveled for agent in self.agents)
        
        final_errors = [np.linalg.norm(agent.position - agent.target_position) 
                       for agent in self.agents]
        
        return {
            'total_energy': total_energy,
            'avg_energy_per_agent': total_energy / len(self.agents),
            'total_distance': total_distance,
            'avg_distance_per_agent': total_distance / len(self.agents),
            'max_final_error': max(final_errors),
            'avg_final_error': np.mean(final_errors),
            'convergence_rate': sum(1 for e in final_errors if e < 0.15) / len(self.agents)
        }
    
    def check_collisions(self, min_distance: float = 0.3) -> int:
        """
        Check for inter-agent collisions.
        
        Args:
            min_distance: Minimum safe distance between agents (m)
            
        Returns:
            Number of collision violations
        """
        collisions = 0
        positions = np.array([agent.position for agent in self.agents])
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_distance:
                    collisions += 1
        
        return collisions
    
    def close(self):
        """Cleanup and close simulation."""
        p.disconnect(physicsClientId=self.physics_client)
        print("Environment closed")


def demo_simple_swarm():
    """Demo: Simple swarm moving from grid to circle."""
    print("\n" + "="*60)
    print("DEMO: Simple Swarm Formation")
    print("="*60 + "\n")
    
    # Create environment
    env = SwarmEnvironment(num_agents=9, gui=True)
    
    # Starting positions (3x3 grid)
    grid_size = 3
    spacing = 1.0
    start_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - grid_size/2) * spacing
            y = (j - grid_size/2) * spacing
            z = 1.0
            start_positions.append([x, y, z])
    
    start_positions = np.array(start_positions)
    
    # Target positions (circle)
    target_positions = []
    radius = 2.0
    for i in range(9):
        theta = 2 * np.pi * i / 9
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 3.0
        target_positions.append([x, y, z])
    
    target_positions = np.array(target_positions)
    
    # Initialize
    env.initialize_agents(start_positions)
    env.set_formation_targets(target_positions)
    
    # Run simulation
    converged = env.run_until_converged(max_steps=3000, tolerance=0.2)
    
    # Get metrics
    metrics = env.get_metrics()
    print("\nFinal Metrics:")
    print(f"  Total Energy: {metrics['total_energy']:.2f} J")
    print(f"  Avg Energy/Agent: {metrics['avg_energy_per_agent']:.2f} J")
    print(f"  Total Distance: {metrics['total_distance']:.2f} m")
    print(f"  Avg Distance/Agent: {metrics['avg_distance_per_agent']:.2f} m")
    print(f"  Max Final Error: {metrics['max_final_error']:.3f} m")
    print(f"  Convergence Rate: {metrics['convergence_rate']*100:.1f}%")
    
    # Check collisions
    collisions = env.check_collisions()
    print(f"  Collision Violations: {collisions}")
    
    # Keep window open
    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    demo_simple_swarm()
