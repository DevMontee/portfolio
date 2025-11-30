"""
Day 1 Integration: Formation Generation + PyBullet Simulation
Demonstrates complete workflow from shape specification to swarm simulation.

Location: D:\Swarm Robotics Project\Setup + Formation Layer\
"""

import numpy as np
import argparse
import time
import sys
import os

# Set working directory to correct location
WORKING_DIR = r"D:\Swarm Robotics Project\Setup + Formation Layer"
os.chdir(WORKING_DIR)
sys.path.insert(0, WORKING_DIR)

from formation_generator import FormationGenerator
from swarm_agent import SwarmEnvironment


class SwarmFormationPlatform:
    """
    Main platform integrating formation generation and swarm simulation.
    """
    
    def __init__(self, num_agents: int = 20, gui: bool = True):
        """
        Initialize swarm formation platform.
        
        Args:
            num_agents: Number of agents in swarm
            gui: Enable visualization
        """
        self.num_agents = num_agents
        self.gui = gui
        self.formation_gen = FormationGenerator(num_agents=num_agents, scale=1.0)
        self.env = None
        
    def generate_starting_positions(self, formation_type: str = 'grid') -> np.ndarray:
        """
        Generate starting positions for agents.
        
        Args:
            formation_type: 'grid', 'random', or 'line'
            
        Returns:
            (N, 3) array of starting positions
        """
        if formation_type == 'grid':
            # Arrange in square grid at ground level
            grid_size = int(np.ceil(np.sqrt(self.num_agents)))
            positions = []
            
            for i in range(self.num_agents):
                row = i // grid_size
                col = i % grid_size
                x = (col - grid_size/2) * 0.8
                y = (row - grid_size/2) * 0.8
                z = 0.5
                positions.append([x, y, z])
            
            return np.array(positions)
        
        elif formation_type == 'random':
            # Random positions in a box
            positions = np.random.uniform(-3, 3, (self.num_agents, 3))
            positions[:, 2] = np.random.uniform(0.5, 2.0, self.num_agents)
            return positions
        
        elif formation_type == 'line':
            # Single line
            positions = []
            for i in range(self.num_agents):
                x = (i - self.num_agents/2) * 0.5
                y = 0
                z = 0.5
                positions.append([x, y, z])
            return np.array(positions)
        
        else:
            raise ValueError(f"Unknown starting formation: {formation_type}")
    
    def run_formation_test(self, 
                          formation_name: str,
                          start_type: str = 'grid',
                          max_steps: int = 5000,
                          visualize_formation: bool = False) -> dict:
        """
        Run complete test for a single formation.
        
        Args:
            formation_name: Name of target formation
            start_type: Starting position type
            max_steps: Maximum simulation steps
            visualize_formation: Show formation visualization before simulation
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*70}")
        print(f"FORMATION TEST: {formation_name.upper()}")
        print('='*70)
        
        # Step 1: Generate formation waypoints
        print(f"\n[1/4] Generating {formation_name} formation...")
        waypoints = self.formation_gen.generate_formation(formation_name)
        info = self.formation_gen.get_formation_info()
        
        print(f"  ✓ Generated {info['num_agents']} waypoints")
        print(f"  ✓ Bounds: X[{info['bounds']['x'][0]:.2f}, {info['bounds']['x'][1]:.2f}] "
              f"Y[{info['bounds']['y'][0]:.2f}, {info['bounds']['y'][1]:.2f}] "
              f"Z[{info['bounds']['z'][0]:.2f}, {info['bounds']['z'][1]:.2f}]")
        
        if visualize_formation:
            self.formation_gen.visualize()
        
        # Step 2: Generate starting positions
        print(f"\n[2/4] Setting up simulation...")
        start_positions = self.generate_starting_positions(start_type)
        print(f"  ✓ Starting formation: {start_type}")
        
        # Step 3: Initialize environment
        print(f"  ✓ Initializing PyBullet environment...")
        self.env = SwarmEnvironment(num_agents=self.num_agents, gui=self.gui)
        self.env.initialize_agents(start_positions)
        self.env.set_formation_targets(waypoints)
        
        # Step 4: Run simulation
        print(f"\n[3/4] Running simulation (max {max_steps} steps)...")
        start_time = time.time()
        converged = self.env.run_until_converged(
            max_steps=max_steps,
            tolerance=0.2,
            check_interval=100
        )
        sim_time = time.time() - start_time
        
        # Step 5: Collect metrics
        print(f"\n[4/4] Collecting metrics...")
        metrics = self.env.get_metrics()
        collisions = self.env.check_collisions(min_distance=0.35)
        
        # Compile results
        results = {
            'formation': formation_name,
            'converged': converged,
            'simulation_time': sim_time,
            'collisions': collisions,
            **metrics
        }
        
        # Print summary
        print(f"\n{'─'*70}")
        print("RESULTS SUMMARY")
        print('─'*70)
        print(f"  Convergence: {'✓ SUCCESS' if converged else '✗ TIMEOUT'}")
        print(f"  Simulation Time: {sim_time:.2f} s")
        print(f"  Total Energy: {metrics['total_energy']:.2f} J")
        print(f"  Avg Energy/Agent: {metrics['avg_energy_per_agent']:.2f} J")
        print(f"  Total Distance: {metrics['total_distance']:.2f} m")
        print(f"  Max Final Error: {metrics['max_final_error']:.3f} m")
        print(f"  Avg Final Error: {metrics['avg_final_error']:.3f} m")
        print(f"  Convergence Rate: {metrics['convergence_rate']*100:.1f}%")
        print(f"  Collision Violations: {collisions}")
        print('='*70)
        
        return results
    
    def run_all_formations(self, save_results: bool = True):
        """
        Run tests for all 5 formations.
        
        Args:
            save_results: Save results to file
        """
        formations = ['cupid', 'dragon', 'flag', 'helix', 'mandala']
        all_results = []
        
        print("\n" + "="*70)
        print("SWARM FORMATION PLATFORM - COMPREHENSIVE TEST")
        print("Testing 5 formations with 20 agents each")
        print("="*70)
        
        for i, formation in enumerate(formations, 1):
            print(f"\n\n{'#'*70}")
            print(f"TEST {i}/5: {formation.upper()}")
            print('#'*70)
            
            results = self.run_formation_test(
                formation_name=formation,
                start_type='grid',
                max_steps=5000,
                visualize_formation=False
            )
            
            all_results.append(results)
            
            # Close environment between tests
            if self.env:
                self.env.close()
                self.env = None
            
            # Brief pause between tests
            if i < len(formations):
                print(f"\nNext test in 3 seconds...")
                time.sleep(3)
        
        # Summary report
        self.print_summary_report(all_results)
        
        if save_results:
            self.save_results(all_results)
    
    def print_summary_report(self, results: list):
        """Print summary report for all formations."""
        print("\n\n" + "="*70)
        print("FINAL SUMMARY REPORT")
        print("="*70)
        
        print(f"\n{'Formation':<15} {'Conv':<8} {'Time(s)':<10} {'Energy(J)':<12} "
              f"{'Dist(m)':<10} {'Colls':<8}")
        print("─"*70)
        
        for r in results:
            conv_str = "✓" if r['converged'] else "✗"
            print(f"{r['formation'].capitalize():<15} {conv_str:<8} "
                  f"{r['simulation_time']:<10.2f} {r['avg_energy_per_agent']:<12.2f} "
                  f"{r['avg_distance_per_agent']:<10.2f} {r['collisions']:<8}")
        
        print("="*70)
        
        # Statistics
        convergence_rate = sum(1 for r in results if r['converged']) / len(results) * 100
        avg_energy = np.mean([r['avg_energy_per_agent'] for r in results])
        avg_collisions = np.mean([r['collisions'] for r in results])
        
        print(f"\nOverall Statistics:")
        print(f"  Convergence Rate: {convergence_rate:.1f}%")
        print(f"  Avg Energy/Agent: {avg_energy:.2f} J")
        print(f"  Avg Collisions: {avg_collisions:.1f}")
        print("="*70 + "\n")
    
    def save_results(self, results: list, filename: str = "day1_results.txt"):
        """
        Save results to file (Windows path compatible).
        
        Args:
            results: List of results dictionaries
            filename: Output filename
        """
        # Use correct directory path
        filepath = os.path.join(r"D:\Swarm Robotics Project\Setup + Formation Layer", filename)
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DAY 1 TEST RESULTS - Swarm Formation Platform\n")
            f.write("="*70 + "\n\n")
            
            for r in results:
                f.write(f"\nFormation: {r['formation'].upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"  Converged: {r['converged']}\n")
                f.write(f"  Simulation Time: {r['simulation_time']:.2f} s\n")
                f.write(f"  Total Energy: {r['total_energy']:.2f} J\n")
                f.write(f"  Avg Energy/Agent: {r['avg_energy_per_agent']:.2f} J\n")
                f.write(f"  Total Distance: {r['total_distance']:.2f} m\n")
                f.write(f"  Max Final Error: {r['max_final_error']:.3f} m\n")
                f.write(f"  Avg Final Error: {r['avg_final_error']:.3f} m\n")
                f.write(f"  Convergence Rate: {r['convergence_rate']*100:.1f}%\n")
                f.write(f"  Collisions: {r['collisions']}\n")
        
        print(f"✓ Results saved to {filepath}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Swarm Formation Platform - Day 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --formation helix
  python main.py --formation all --no-gui
  python main.py --formation mandala --visualize
        """
    )
    parser.add_argument('--formation', type=str, default='all',
                       choices=['all', 'cupid', 'dragon', 'flag', 'helix', 'mandala'],
                       help='Formation to test (default: all)')
    parser.add_argument('--agents', type=int, default=20,
                       help='Number of agents (default: 20)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Disable GUI (headless mode)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show formation visualization before simulation')
    
    args = parser.parse_args()
    
    # Create platform
    platform = SwarmFormationPlatform(
        num_agents=args.agents,
        gui=not args.no_gui
    )
    
    try:
        if args.formation == 'all':
            # Test all formations
            platform.run_all_formations(save_results=True)
        else:
            # Test single formation
            results = platform.run_formation_test(
                formation_name=args.formation,
                start_type='grid',
                max_steps=5000,
                visualize_formation=args.visualize
            )
            
            if platform.env:
                input("\nPress Enter to close...")
                platform.env.close()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if platform.env:
            platform.env.close()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if platform.env:
            platform.env.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
