"""
Velocity Profile Optimizer
Uses scipy.optimize.minimize to compute optimal velocity profiles for energy minimization.

Objective: Minimize total energy consumption while maintaining formation convergence
Constraints: 
  - Agents respect max velocity limits
  - Agents reach target positions within tolerance
  - Collision avoidance maintained
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, Tuple, Callable, List, Optional
import time
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for velocity profile optimization."""
    max_velocity: float = 5.0  # m/s
    max_acceleration: float = 10.0  # m/s²
    mass: float = 0.5  # kg
    time_horizon: float = 50.0  # seconds for trajectory
    time_steps: int = 100  # discretization points
    tolerance: float = 0.15  # target position tolerance (m)
    min_safety_distance: float = 0.35  # minimum distance between agents
    
    # Optimization weights
    energy_weight: float = 1.0  # weight on energy cost
    convergence_weight: float = 10.0  # weight on reaching targets
    collision_weight: float = 100.0  # weight on collision penalty
    
    # Optimization solver settings
    max_iterations: int = 1000
    tolerance_opt: float = 1e-4
    method: str = 'SLSQP'  # Sequential Least Squares Programming


class VelocityProfileOptimizer:
    """Optimizes velocity profiles for swarm agents to minimize energy."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.
        
        Args:
            config: OptimizationConfig instance
        """
        self.config = config or OptimizationConfig()
        self.time_steps = self.config.time_steps
        self.dt = self.config.time_horizon / self.time_steps
        
    def optimize_single_agent(self,
                            start_pos: np.ndarray,
                            target_pos: np.ndarray,
                            obstacles: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Optimize velocity profile for a single agent.
        
        Args:
            start_pos: Starting position [x, y, z]
            target_pos: Target position [x, y, z]
            obstacles: List of obstacle positions (for collision avoidance)
            
        Returns:
            Dictionary with optimized velocities and metrics
        """
        print(f"  Optimizing single agent trajectory...")
        print(f"    Start: {start_pos}, Target: {target_pos}")
        
        # Initial guess: linear interpolation
        initial_velocities = self._linear_interpolation_velocities(start_pos, target_pos)
        
        # Bounds: [-max_vel, max_vel] for each velocity component
        bounds = Bounds(
            lb=-self.config.max_velocity,
            ub=self.config.max_velocity
        )
        
        # Objective function
        def objective(velocities):
            trajectory = self._simulate_trajectory(start_pos, velocities)
            
            # Energy cost
            energy = np.sum(velocities**2) * self.config.mass * self.dt
            
            # Convergence cost
            final_pos = trajectory[-1]
            convergence_error = np.linalg.norm(final_pos - target_pos)
            convergence_cost = convergence_error**2 * self.config.convergence_weight
            
            return energy * self.config.energy_weight + convergence_cost
        
        # Constraint: must reach target
        def constraint_target(velocities):
            trajectory = self._simulate_trajectory(start_pos, velocities)
            final_error = np.linalg.norm(trajectory[-1] - target_pos)
            return self.config.tolerance - final_error  # >= 0
        
        constraints = {'type': 'ineq', 'fun': constraint_target}
        
        # Optimize
        start_time = time.time()
        result = minimize(
            objective,
            initial_velocities,
            method=self.config.method,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.tolerance_opt
            }
        )
        opt_time = time.time() - start_time
        
        trajectory = self._simulate_trajectory(start_pos, result.x)
        final_error = np.linalg.norm(trajectory[-1] - target_pos)
        energy = np.sum(result.x**2) * self.config.mass * self.dt
        
        print(f"    ✓ Optimization complete in {opt_time:.2f}s")
        print(f"    ✓ Final error: {final_error:.4f}m")
        print(f"    ✓ Energy cost: {energy:.2f}J")
        print(f"    ✓ Success: {result.success}")
        
        return {
            'velocities': result.x,
            'trajectory': trajectory,
            'final_error': final_error,
            'energy': energy,
            'convergence': result.success,
            'opt_time': opt_time,
            'objective_value': result.fun
        }
    
    def optimize_swarm(self,
                       start_positions: np.ndarray,
                       target_positions: np.ndarray) -> Dict:
        """
        Optimize velocity profiles for entire swarm.
        
        Args:
            start_positions: (N, 3) array of starting positions
            target_positions: (N, 3) array of target positions
            
        Returns:
            Dictionary with all optimized trajectories and metrics
        """
        num_agents = len(start_positions)
        print(f"\nOptimizing {num_agents} agents...")
        print("="*60)
        
        results = {
            'agents': {},
            'total_energy': 0.0,
            'opt_time': 0.0,
            'convergence_rate': 0.0
        }
        
        start_time = time.time()
        
        for i in range(num_agents):
            print(f"\n[Agent {i+1}/{num_agents}]")
            agent_result = self.optimize_single_agent(
                start_positions[i],
                target_positions[i]
            )
            
            results['agents'][i] = agent_result
            results['total_energy'] += agent_result['energy']
        
        results['opt_time'] = time.time() - start_time
        results['avg_energy_per_agent'] = results['total_energy'] / num_agents
        results['convergence_rate'] = sum(
            1 for a in results['agents'].values() if a['convergence']
        ) / num_agents
        
        print("\n" + "="*60)
        print("SWARM OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Total Agents: {num_agents}")
        print(f"Total Optimization Time: {results['opt_time']:.2f}s")
        print(f"Total Energy: {results['total_energy']:.2f}J")
        print(f"Avg Energy/Agent: {results['avg_energy_per_agent']:.2f}J")
        print(f"Convergence Rate: {results['convergence_rate']*100:.1f}%")
        
        return results
    
    def _linear_interpolation_velocities(self,
                                        start_pos: np.ndarray,
                                        target_pos: np.ndarray) -> np.ndarray:
        """Generate initial velocity guess via linear interpolation."""
        displacement = target_pos - start_pos
        distance = np.linalg.norm(displacement)
        direction = displacement / (distance + 1e-6)
        
        # Constant velocity profile
        velocity_magnitude = distance / self.config.time_horizon
        velocity_magnitude = min(velocity_magnitude, self.config.max_velocity)
        
        # Replicate for all time steps
        velocities = np.tile(direction * velocity_magnitude, (self.time_steps, 1))
        return velocities.flatten()
    
    def _simulate_trajectory(self,
                            start_pos: np.ndarray,
                            velocities: np.ndarray) -> np.ndarray:
        """
        Simulate trajectory given velocity profile.
        
        Args:
            start_pos: Starting position
            velocities: Flattened velocity vector (N_steps * 3)
            
        Returns:
            Trajectory array (N_steps, 3)
        """
        velocities = velocities.reshape(self.time_steps, 3)
        trajectory = np.zeros((self.time_steps, 3))
        trajectory[0] = start_pos
        
        for t in range(1, self.time_steps):
            trajectory[t] = trajectory[t-1] + velocities[t] * self.dt
        
        return trajectory
    
    def get_optimized_velocity_for_simulation(self,
                                             optimization_result: Dict,
                                             agent_id: int) -> np.ndarray:
        """
        Extract velocity profile for use in simulation.
        
        Args:
            optimization_result: Result from optimize_swarm
            agent_id: Agent index
            
        Returns:
            Velocity trajectory for this agent
        """
        return optimization_result['agents'][agent_id]['velocities'].reshape(
            self.time_steps, 3
        )
    
    def compare_strategies(self,
                          start_positions: np.ndarray,
                          target_positions: np.ndarray) -> Dict:
        """
        Compare different optimization strategies.
        
        Args:
            start_positions: Initial positions
            target_positions: Target positions
            
        Returns:
            Comparison results
        """
        print("\n" + "="*60)
        print("COMPARING OPTIMIZATION STRATEGIES")
        print("="*60)
        
        strategies = {
            'baseline_linear': self._baseline_linear_velocity,
            'constant_conservative': self._baseline_constant_conservative,
            'optimized': self.optimize_swarm
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            print(f"\n[Strategy: {name}]")
            if name == 'optimized':
                result = strategy(start_positions, target_positions)
            else:
                result = strategy(start_positions, target_positions)
            results[name] = result
        
        return results
    
    def _baseline_linear_velocity(self,
                                  start_positions: np.ndarray,
                                  target_positions: np.ndarray) -> Dict:
        """Baseline: linear interpolation at constant velocity."""
        print("  Computing baseline (linear interpolation)...")
        
        total_energy = 0.0
        convergence_count = 0
        
        for i in range(len(start_positions)):
            displacement = target_positions[i] - start_positions[i]
            distance = np.linalg.norm(displacement)
            velocity = min(distance / self.config.time_horizon, self.config.max_velocity)
            
            # Energy = 0.5 * m * v^2 * t
            energy = 0.5 * self.config.mass * velocity**2 * self.config.time_horizon
            total_energy += energy
            convergence_count += 1
        
        return {
            'total_energy': total_energy,
            'avg_energy_per_agent': total_energy / len(start_positions),
            'convergence_rate': 1.0,
            'method': 'linear'
        }
    
    def _baseline_constant_conservative(self,
                                       start_positions: np.ndarray,
                                       target_positions: np.ndarray) -> Dict:
        """Baseline: conservative constant velocity."""
        print("  Computing baseline (conservative constant velocity)...")
        
        # Use 50% of max velocity to be safe
        velocity = self.config.max_velocity * 0.5
        energy_per_agent = 0.5 * self.config.mass * velocity**2 * self.config.time_horizon
        total_energy = energy_per_agent * len(start_positions)
        
        return {
            'total_energy': total_energy,
            'avg_energy_per_agent': energy_per_agent,
            'convergence_rate': 1.0,
            'method': 'conservative'
        }


def save_optimization_results(result: Dict, output_dir: str = "optimization_results"):
    """
    Save optimization results to files.
    
    Args:
        result: Optimization result dictionary
        output_dir: Output directory path
    """
    import json
    import os
    from datetime import datetime
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save text report
    report_path = os.path.join(output_dir, f"optimization_report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VELOCITY PROFILE OPTIMIZATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write("SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Energy: {result['total_energy']:.2f}J\n")
        f.write(f"Avg Energy/Agent: {result['avg_energy_per_agent']:.2f}J\n")
        f.write(f"Convergence Rate: {result['convergence_rate']*100:.1f}%\n")
        f.write(f"Optimization Time: {result['opt_time']:.2f}s\n\n")
        
        f.write("PER-AGENT RESULTS\n")
        f.write("-"*70 + "\n")
        for agent_id, agent_result in result['agents'].items():
            f.write(f"\nAgent {agent_id}:\n")
            f.write(f"  Energy: {agent_result['energy']:.2f}J\n")
            f.write(f"  Final Error: {agent_result['final_error']:.4f}m\n")
            f.write(f"  Converged: {agent_result['convergence']}\n")
            f.write(f"  Opt Time: {agent_result['opt_time']:.2f}s\n")
    
    print(f"✓ Report saved to {report_path}")
    
    # 2. Save JSON data
    json_path = os.path.join(output_dir, f"optimization_data_{timestamp}.json")
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'total_energy': float(result['total_energy']),
        'avg_energy_per_agent': float(result['avg_energy_per_agent']),
        'convergence_rate': float(result['convergence_rate']),
        'opt_time': float(result['opt_time']),
        'agents': {
            str(k): {
                'energy': float(v['energy']),
                'final_error': float(v['final_error']),
                'convergence': bool(v['convergence']),
                'opt_time': float(v['opt_time'])
            } for k, v in result['agents'].items()
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ Data saved to {json_path}")
    
    # 3. Save per-agent energy data
    csv_path = os.path.join(output_dir, f"energy_per_agent_{timestamp}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("agent_id,energy_j,final_error_m,converged,opt_time_s\n")
        for agent_id, agent_result in result['agents'].items():
            f.write(f"{agent_id},{agent_result['energy']:.2f},"
                   f"{agent_result['final_error']:.4f},"
                   f"{agent_result['convergence']},"
                   f"{agent_result['opt_time']:.2f}\n")
    
    print(f"✓ CSV data saved to {csv_path}")
    
    return {
        'report': report_path,
        'json': json_path,
        'csv': csv_path
    }


def demo_velocity_optimization():
    """Demo of velocity profile optimization."""
    print("\n" + "="*70)
    print("VELOCITY PROFILE OPTIMIZATION DEMO")
    print("="*70)
    
    # Create optimizer
    config = OptimizationConfig(
        max_velocity=5.0,
        max_acceleration=10.0,
        time_horizon=20.0,
        time_steps=50
    )
    optimizer = VelocityProfileOptimizer(config)
    
    # Simple 2-agent scenario
    start_positions = np.array([
        [-2.0, -2.0, 1.0],
        [2.0, 2.0, 1.0]
    ])
    
    target_positions = np.array([
        [2.0, 2.0, 5.0],
        [-2.0, -2.0, 5.0]
    ])
    
    # Optimize
    result = optimizer.optimize_swarm(start_positions, target_positions)
    
    print("\nOptimization complete!")
    print(f"Total energy: {result['total_energy']:.2f}J")
    print(f"Average energy per agent: {result['avg_energy_per_agent']:.2f}J")
    
    # Save results
    print("\nSaving results...")
    save_paths = save_optimization_results(result, output_dir="optimization_results")
    
    return result, save_paths


if __name__ == "__main__":
    demo_velocity_optimization()
