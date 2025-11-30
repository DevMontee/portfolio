"""
Contingency Integration Wrapper for Existing SwarmEnvironment
Drop-in replacement - use instead of SwarmEnvironment without modifying original code

Usage:
    Instead of:
        from swarm_agent import SwarmEnvironment
        env = SwarmEnvironment(...)
    
    Use:
        from contingency_swarm_wrapper import SwarmEnvironmentWithContingency
        env = SwarmEnvironmentWithContingency(...)
        
    The wrapper handles everything internally.
"""

import numpy as np
from typing import Optional
import sys


class SwarmEnvironmentWithContingency:
    """
    Wrapper around existing SwarmEnvironment that adds contingency monitoring.
    
    This wrapper doesn't modify your original SwarmEnvironment class.
    It wraps it and intercepts the run_until_converged() calls to add contingency checks.
    """
    
    def __init__(self, 
                 num_agents: int = 20,
                 gui: bool = True,
                 time_step: float = 0.01,
                 enable_contingency: bool = True,
                 formation_name: str = 'unknown'):
        """
        Initialize wrapper with contingency monitoring.
        
        Args:
            num_agents: Number of agents
            gui: Enable PyBullet GUI
            time_step: Physics time step
            enable_contingency: Enable contingency monitoring
            formation_name: Formation type for optimized reserves
        """
        # Import your original SwarmEnvironment
        try:
            from swarm_agent import SwarmEnvironment
        except ImportError:
            print("Error: Could not import SwarmEnvironment from swarm_agent.py")
            print("Make sure swarm_agent.py is in the same directory")
            raise
        
        # Create the original environment
        self.env = SwarmEnvironment(num_agents=num_agents, gui=gui, time_step=time_step)
        
        self.num_agents = num_agents
        self.enable_contingency = enable_contingency
        self.formation_name = formation_name
        
        # Initialize contingency system if enabled
        if enable_contingency:
            self._init_contingency()
        
        print(f"✓ SwarmEnvironmentWithContingency initialized")
        if enable_contingency:
            print(f"  Formation: {formation_name}")
            print(f"  Contingency: ENABLED")
    
    def _init_contingency(self):
        """Initialize contingency monitoring system."""
        try:
            from contingency_reserve import SwarmContingencyManager, ContingencyReserves
            from formation_specific_config import get_formation_reserves
            from contingency_monitoring import ContingencyDashboard
        except ImportError as e:
            print(f"Error importing contingency modules: {e}")
            print("Make sure you've copied the contingency files to your project directory")
            raise
        
        # Energy limit from your agents
        self.energy_limit = 4.5  # Joules per agent
        
        # Get formation-specific reserves if available
        try:
            self.contingency_reserves = get_formation_reserves(self.formation_name)
            print(f"  Using {self.formation_name}-optimized reserves")
        except:
            # Fall back to default reserves
            self.contingency_reserves = ContingencyReserves()
            print(f"  Using default reserves (formation '{self.formation_name}' not recognized)")
        
        # Create contingency manager
        self.contingency_manager = SwarmContingencyManager(
            self.num_agents,
            self.contingency_reserves
        )
        
        # Create dashboard for visualization
        self.dashboard = ContingencyDashboard(self.formation_name, self.num_agents)
    
    # =========================================================================
    # PROXY METHODS - Pass through to original environment
    # =========================================================================
    
    def initialize_agents(self, start_positions: np.ndarray):
        """Initialize agents - delegates to original environment."""
        return self.env.initialize_agents(start_positions)
    
    def set_formation_targets(self, target_positions: np.ndarray):
        """Set formation targets - delegates to original environment."""
        return self.env.set_formation_targets(target_positions)
    
    def step(self):
        """Execute one simulation step - delegates to original environment."""
        return self.env.step()
    
    def get_metrics(self) -> dict:
        """Get simulation metrics - delegates to original environment."""
        return self.env.get_metrics()
    
    def check_collisions(self, min_distance: float = 0.3) -> int:
        """Check for collisions - delegates to original environment."""
        return self.env.check_collisions(min_distance)
    
    def close(self):
        """Close environment - delegates to original environment."""
        return self.env.close()
    
    @property
    def agents(self):
        """Access agents from original environment."""
        return self.env.agents
    
    @property
    def physics_client(self):
        """Access physics client from original environment."""
        return self.env.physics_client
    
    @property
    def time_step(self):
        """Access time_step from original environment."""
        return self.env.time_step
    
    # =========================================================================
    # ENHANCED METHOD: run_until_converged with contingency
    # =========================================================================
    
    def run_until_converged_with_contingency(self,
                                             max_steps: int = 5000,
                                             tolerance: float = 0.15,
                                             check_interval: int = 100) -> bool:
        """
        Run simulation with contingency monitoring.
        
        This is the main enhanced method that adds contingency checks
        to the original run_until_converged() method.
        
        Args:
            max_steps: Maximum simulation steps
            tolerance: Position tolerance for convergence
            check_interval: Steps between convergence checks
            
        Returns:
            True if converged, False if timed out or error
        """
        print(f"\nRunning simulation with contingency monitoring...")
        print(f"  Max steps: {max_steps}")
        print(f"  Convergence tolerance: {tolerance}m")
        
        converged_flag = False
        
        try:
            for step in range(max_steps):
                # Get current agent states
                agent_positions = np.array([agent.position for agent in self.agents])
                agent_energies = np.array([agent.energy_consumed for agent in self.agents])
                target_positions = np.array([agent.target_position for agent in self.agents])
                
                # ✨ CONTINGENCY CHECK 1: Energy reserves
                if self.enable_contingency:
                    energy_warning = self.contingency_manager.update_agent_energy(
                        agent_energies,
                        self.energy_limit
                    )
                    
                    if energy_warning:
                        print(f"  ⚠️  Step {step}: {energy_warning['warning']}")
                        # Reduce agent velocities to save energy
                        for agent in self.agents:
                            agent.max_velocity *= 0.75
                
                # ✨ CONTINGENCY CHECK 2: Collision buffers
                if self.enable_contingency:
                    min_distance = self._compute_min_distance(agent_positions)
                    collision_status = self.contingency_manager.check_collision_buffer(
                        np.array([min_distance]),
                        self.formation_name
                    )
                    
                    if collision_status['status'].value >= 2:  # WARNING or CRITICAL
                        print(f"  ⚠️  Step {step}: {collision_status['status'].name} - "
                              f"distance={min_distance:.3f}m")
                        
                        if collision_status['status'].value == 3:  # CRITICAL
                            print(f"  🛑 CRITICAL - Stopping simulation")
                            return False
                        else:
                            # Slow convergence
                            for agent in self.agents:
                                agent.max_velocity *= 0.75
                
                # Step physics
                self.step()
                
                # ✨ CONTINGENCY CHECK 3: Position errors (periodic)
                if self.enable_contingency and step % 100 == 0 and step > 0:
                    position_errors = np.linalg.norm(agent_positions - target_positions, axis=1)
                    error_status = self.contingency_manager.check_position_errors(
                        position_errors,
                        self.formation_name
                    )
                    
                    if not error_status['acceptable']:
                        print(f"  ⚠️  Step {step}: Position error high")
                
                # ✨ CONTINGENCY CHECK 4: Convergence timeout
                if self.enable_contingency:
                    has_converged = all(agent.at_target(tolerance) for agent in self.agents)
                    timeout_status = self.contingency_manager.check_convergence_timeout(
                        step, has_converged
                    )
                    
                    if timeout_status['timeout_reached']:
                        print(f"  ⏱️  Convergence timeout at step {step}")
                        return False
                
                # ✨ Record metrics for dashboard
                if self.enable_contingency and step % 100 == 0:
                    min_distance = self._compute_min_distance(agent_positions)
                    pos_errors = np.linalg.norm(agent_positions - target_positions, axis=1)
                    
                    self.dashboard.record_state(
                        step=step,
                        contingency_level=self.contingency_manager.contingency_level,
                        energies=agent_energies,
                        min_dist=min_distance,
                        pos_errors=pos_errors,
                        active_agents=self.num_agents,
                        energy_limit=self.energy_limit
                    )
                
                # Check convergence
                if step % check_interval == 0:
                    converged_flag = all(agent.at_target(tolerance) for agent in self.agents)
                    
                    if converged_flag:
                        print(f"✓ Converged at step {step}")
                        
                        # Print contingency summary
                        if self.enable_contingency:
                            report = self.contingency_manager.generate_contingency_report()
                            print(f"\nContingency Report:")
                            print(f"  Final Level: {report['current_level']}")
                            print(f"  Total Events: {len(report['failures'])}")
                        
                        return True
                    
                    # Progress output
                    distances = [np.linalg.norm(agent.position - agent.target_position)
                               for agent in self.agents]
                    avg_dist = np.mean(distances)
                    max_dist = np.max(distances)
                    
                    status_str = ""
                    if self.enable_contingency:
                        status_str = f" | {self.contingency_manager.contingency_level.name}"
                    
                    print(f"  Step {step}: avg={avg_dist:.3f}m, max={max_dist:.3f}m{status_str}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return False
        except Exception as e:
            print(f"\nError during simulation: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"⚠ Timeout after {max_steps} steps")
        
        if self.enable_contingency:
            report = self.contingency_manager.generate_contingency_report()
            print(f"\nFinal Contingency Status:")
            print(f"  Level: {report['current_level']}")
            print(f"  Total Events: {len(report['failures'])}")
        
        return False
    
    def run_until_converged(self,
                           max_steps: int = 5000,
                           tolerance: float = 0.15,
                           check_interval: int = 100) -> bool:
        """
        Run until converged with automatic contingency handling.
        
        If contingency is enabled, uses enhanced monitoring.
        Otherwise, delegates to original environment.
        
        Args:
            max_steps: Maximum steps
            tolerance: Convergence tolerance
            check_interval: Check interval
            
        Returns:
            True if converged, False otherwise
        """
        if self.enable_contingency:
            return self.run_until_converged_with_contingency(
                max_steps=max_steps,
                tolerance=tolerance,
                check_interval=check_interval
            )
        else:
            # Use original environment's method
            return self.env.run_until_converged(
                max_steps=max_steps,
                tolerance=tolerance,
                check_interval=check_interval
            )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def save_contingency_dashboard(self, filepath: str):
        """Save contingency dashboard visualization."""
        if self.enable_contingency and self.dashboard:
            self.dashboard.save_dashboard(filepath)
            print(f"✓ Dashboard saved to {filepath}")
        else:
            print("Contingency monitoring not enabled")
    
    def print_contingency_summary(self):
        """Print contingency monitoring summary."""
        if self.enable_contingency and self.dashboard:
            self.dashboard.print_contingency_summary()
        else:
            print("Contingency monitoring not enabled")
    
    def get_contingency_report(self) -> dict:
        """Get contingency monitoring report."""
        if self.enable_contingency and self.contingency_manager:
            return self.contingency_manager.generate_contingency_report()
        else:
            return {}
    
    def _compute_min_distance(self, positions: np.ndarray) -> float:
        """Compute minimum pairwise distance between agents."""
        if len(positions) < 2:
            return np.inf
        
        min_dist = np.inf
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)
        
        return min_dist


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_basic_usage():
    """Basic example showing how to use the wrapper."""
    print("\n" + "="*70)
    print("EXAMPLE: Using SwarmEnvironmentWithContingency")
    print("="*70)
    
    import numpy as np
    
    # Try to import required modules
    try:
        from formation_generator import FormationGenerator
    except ImportError:
        print("Error: Could not import FormationGenerator")
        print("Make sure formation_generator.py is available")
        return
    
    # Create environment with contingency
    print("\n1. Creating environment with contingency monitoring...")
    env = SwarmEnvironmentWithContingency(
        num_agents=20,
        gui=False,
        enable_contingency=True,
        formation_name='helix'  # Optimized reserves for helix
    )
    
    # Generate formation
    print("\n2. Generating helix formation...")
    gen = FormationGenerator(num_agents=20)
    target_positions = gen.generate_formation('helix')
    
    # Create starting positions (grid)
    print("\n3. Creating starting positions...")
    grid_size = int(np.ceil(np.sqrt(20)))
    start_positions = []
    for i in range(20):
        row = i // grid_size
        col = i % grid_size
        x = (col - grid_size/2) * 0.8
        y = (row - grid_size/2) * 0.8
        z = 0.5
        start_positions.append([x, y, z])
    start_positions = np.array(start_positions)
    
    # Initialize agents
    print("\n4. Initializing agents...")
    env.initialize_agents(start_positions)
    env.set_formation_targets(target_positions)
    
    # Run simulation
    print("\n5. Running simulation with contingency monitoring...")
    converged = env.run_until_converged(
        max_steps=10000,
        tolerance=0.20,
        check_interval=100
    )
    
    # Print results
    print("\n6. Results:")
    print(f"  Converged: {converged}")
    
    # Get metrics
    metrics = env.get_metrics()
    print(f"  Total Energy: {metrics['total_energy']:.2f}J")
    print(f"  Avg Energy/Agent: {metrics['avg_energy_per_agent']:.2f}J")
    print(f"  Collisions: {env.check_collisions()}")
    
    # Print contingency summary
    print("\n7. Contingency Monitoring Summary:")
    env.print_contingency_summary()
    
    # Save dashboard
    print("\n8. Saving contingency dashboard...")
    env.save_contingency_dashboard('helix_contingency_dashboard.png')
    
    # Cleanup
    env.close()
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    example_basic_usage()
