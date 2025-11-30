"""
Energy Optimization + Metrics Platform
Complete workflow: Formation generation → Velocity optimization → Simulation → Metrics analysis

Location: D:\Swarm Robotics Project\Energy Optimization + Metrics\
"""

import numpy as np
import sys
import os
import time
import argparse
from typing import Dict, List, Optional

# Reference paths
SETUP_DIR = r"D:\Swarm Robotics Project\Setup + Formation Layer"
COLLISION_DIR = r"D:\Swarm Robotics Project\Swarm Collision Avoidance"
ENERGY_OPT_DIR = r"D:\Swarm Robotics Project\Energy Optimization + Metrics"

# Add to path
for path in [SETUP_DIR, COLLISION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from formation_generator import FormationGenerator
from swarm_agent import SwarmEnvironment, QuadrotorAgent
from velocity_optimizer import VelocityProfileOptimizer, OptimizationConfig
from metrics_dashboard import MetricsCollector, MetricsDashboard, SimulationMetrics


class EnergyOptimizationPlatform:
    """
    Complete platform for energy-optimized swarm formation control.
    
    Workflow:
    1. Generate formation waypoints
    2. Optimize velocity profiles for minimum energy
    3. Run simulation with optimized velocities
    4. Collect and visualize metrics
    """
    
    def __init__(self, 
                 num_agents: int = 20,
                 gui: bool = False,
                 output_dir: str = ENERGY_OPT_DIR):
        """
        Initialize platform.
        
        Args:
            num_agents: Number of agents
            gui: Enable PyBullet visualization
            output_dir: Directory for outputs
        """
        self.num_agents = num_agents
        self.gui = gui
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.formation_gen = FormationGenerator(num_agents=num_agents, scale=1.0)
        self.optimizer = VelocityProfileOptimizer(OptimizationConfig())
        self.dashboard = MetricsDashboard(output_dir=output_dir)
        
        print(f"✓ EnergyOptimizationPlatform initialized")
        print(f"  Output directory: {output_dir}")
    
    def generate_start_positions(self, 
                                formation_type: str = 'grid') -> np.ndarray:
        """Generate starting positions."""
        if formation_type == 'grid':
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
            positions = np.random.uniform(-3, 3, (self.num_agents, 3))
            positions[:, 2] = np.random.uniform(0.5, 2.0, self.num_agents)
            return positions
        
        else:
            raise ValueError(f"Unknown start type: {formation_type}")
    
    def run_optimized_formation(self,
                               formation_name: str,
                               enable_optimization: bool = True,
                               visualize: bool = False) -> Dict:
        """
        Run complete optimized formation control.
        
        Args:
            formation_name: Target formation type
            enable_optimization: Use velocity optimization
            visualize: Show formation before simulation
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*70)
        print(f"ENERGY-OPTIMIZED FORMATION: {formation_name.upper()}")
        print("="*70)
        
        # Step 1: Generate formation
        print(f"\n[1/5] Generating {formation_name} formation...")
        target_positions = self.formation_gen.generate_formation(formation_name)
        start_positions = self.generate_start_positions('grid')
        
        formation_info = self.formation_gen.get_formation_info()
        print(f"  ✓ Generated {len(target_positions)} waypoints")
        print(f"  ✓ Bounds: X[{formation_info['bounds']['x'][0]:.2f}, "
              f"{formation_info['bounds']['x'][1]:.2f}]")
        
        if visualize:
            self.formation_gen.visualize()
        
        # Step 2: Optimize velocities
        if enable_optimization:
            print(f"\n[2/5] Optimizing velocity profiles...")
            opt_result = self.optimizer.optimize_swarm(start_positions, target_positions)
            print(f"  ✓ Optimization complete")
            print(f"  ✓ Total energy: {opt_result['total_energy']:.2f}J")
            print(f"  ✓ Avg energy/agent: {opt_result['avg_energy_per_agent']:.2f}J")
        else:
            print(f"\n[2/5] Skipping velocity optimization (baseline mode)")
            opt_result = None
        
        # Step 3: Initialize simulation
        print(f"\n[3/5] Initializing simulation...")
        env = SwarmEnvironment(num_agents=self.num_agents, gui=self.gui)
        env.initialize_agents(start_positions)
        env.set_formation_targets(target_positions)
        print(f"  ✓ Environment ready")
        
        # Step 4: Run simulation
        print(f"\n[4/5] Running simulation...")
        metrics_collector = MetricsCollector(self.num_agents, formation_name)
        
        start_time = time.time()
        sim_time = time.time()
        
        converged = False
        for step in range(10000):
            env.step()
            
            # Record metrics
            metrics_collector.record_step(env.agents, step, 10000)
            
            # Check convergence
            if step % 100 == 0:
                errors = [np.linalg.norm(a.position - a.target_position) for a in env.agents]
                converged = all(e < 0.15 for e in errors)
                
                if converged:
                    sim_time = time.time() - start_time
                    print(f"  ✓ Converged at step {step}")
                    break
                
                if step % 500 == 0:
                    avg_error = np.mean(errors)
                    print(f"    Step {step}: avg_error={avg_error:.3f}m")
        
        if not converged:
            sim_time = time.time() - start_time
            print(f"  ⚠ Timeout after 10000 steps")
        
        # Step 5: Collect metrics
        print(f"\n[5/5] Collecting metrics...")
        metrics_collector.record_completion(
            env.agents,
            converged,
            sim_time if converged else 10000 * 0.01,
            sim_time,
            10000
        )
        
        metrics = metrics_collector.get_metrics()
        self.dashboard.add_metrics(metrics)
        
        # Print summary
        print(f"\n{'─'*70}")
        print("RESULTS SUMMARY")
        print('─'*70)
        print(f"  Convergence: {'✓ SUCCESS' if converged else '✗ TIMEOUT'}")
        print(f"  Convergence Time: {sim_time:.2f}s")
        print(f"  Energy (Total): {metrics.total_energy:.2f}J")
        print(f"  Energy (Avg/Agent): {metrics.avg_energy_per_agent:.2f}J")
        print(f"  Collisions: {metrics.collision_count}")
        print(f"  Min Distance: {metrics.min_distance_observed:.3f}m")
        print(f"  Avg Final Error: {np.mean(metrics.final_errors):.3f}m")
        
        if opt_result:
            print(f"  Optimization Benefit:")
            baseline_energy = opt_result['total_energy']
            print(f"    Baseline Energy: {baseline_energy:.2f}J")
            print(f"    Optimized Energy: {metrics.total_energy:.2f}J")
            if baseline_energy > 0:
                savings = (baseline_energy - metrics.total_energy) / baseline_energy * 100
                print(f"    Savings: {savings:.1f}%")
        
        print('='*70)
        
        env.close()
        
        return {
            'formation': formation_name,
            'metrics': metrics,
            'optimization': opt_result,
            'converged': converged
        }
    
    def run_comparison_study(self):
        """
        Run comparison of all formations with and without optimization.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE ENERGY OPTIMIZATION STUDY")
        print("="*70)
        
        formations = ['helix', 'mandala', 'cupid', 'dragon', 'flag']
        all_results = []
        
        for i, formation in enumerate(formations, 1):
            print(f"\n\n{'#'*70}")
            print(f"TEST {i}/{len(formations)}: {formation.upper()}")
            print('#'*70)
            
            results = self.run_optimized_formation(
                formation_name=formation,
                enable_optimization=True,
                visualize=False
            )
            
            all_results.append(results)
            time.sleep(2)  # Brief pause between tests
        
        # Generate comparison visualizations
        self.print_final_report(all_results)
        self.generate_visualizations(all_results)
        
        return all_results
    
    def print_final_report(self, results: List[Dict]):
        """Print comprehensive final report."""
        print("\n\n" + "="*70)
        print("FINAL COMPREHENSIVE REPORT")
        print("="*70)
        
        print(f"\n{'Formation':<15} {'Conv':<8} {'Time(s)':<12} {'Energy(J)':<12} "
              f"{'Colls':<8} {'MinDist(m)':<12}")
        print("─"*70)
        
        for r in results:
            m = r['metrics']
            conv_str = "✓" if r['converged'] else "✗"
            print(f"{m.formation_name.capitalize():<15} {conv_str:<8} "
                  f"{m.convergence_time:<12.2f} {m.total_energy:<12.2f} "
                  f"{m.collision_count:<8} {m.min_distance_observed:<12.3f}")
        
        print("="*70)
        
        # Statistics
        metrics_list = [r['metrics'] for r in results]
        convergence_rate = sum(1 for r in results if r['converged']) / len(results) * 100
        avg_energy = np.mean([m.total_energy for m in metrics_list])
        total_collisions = sum(m.collision_count for m in metrics_list)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Convergence Rate: {convergence_rate:.1f}%")
        print(f"  Average Energy: {avg_energy:.2f}J")
        print(f"  Total Collisions: {total_collisions}")
        print(f"  Total Simulations: {len(results)}")
        
        print("\n")
    
    def generate_visualizations(self, results: List[Dict]):
        """Generate all visualizations and reports with organized directory structure."""
        from datetime import datetime
        
        # Create timestamp for organized folders
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_timestamp_dir = os.path.join(self.output_dir, f"results_{timestamp}")
        os.makedirs(results_timestamp_dir, exist_ok=True)
        
        print(f"\n[Generating visualizations and reports...]")
        print(f"[Output directory: {results_timestamp_dir}]")
        
        # Create subdirectories
        dashboards_dir = os.path.join(results_timestamp_dir, "dashboards")
        reports_dir = os.path.join(results_timestamp_dir, "reports")
        data_dir = os.path.join(results_timestamp_dir, "data")
        summary_dir = os.path.join(results_timestamp_dir, "summary")
        
        for dir_path in [dashboards_dir, reports_dir, data_dir, summary_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Process each result
        for r in results:
            metrics = r['metrics']
            formation_name = metrics.formation_name
            
            # 1. Save dashboard visualization
            dashboard_path = os.path.join(dashboards_dir, f"{formation_name}_dashboard.png")
            self.dashboard.plot_single_simulation(metrics, save_path=dashboard_path)
            
            # 2. Save detailed report
            report_path = os.path.join(reports_dir, f"{formation_name}_report.txt")
            self.dashboard.generate_report(metrics, save_path=report_path)
            
            # 3. Save metrics data (JSON)
            json_path = os.path.join(data_dir, f"{formation_name}_metrics.json")
            self.dashboard.export_metrics_json(metrics, save_path=json_path)
            
            # 4. Save CSV summary
            csv_path = os.path.join(data_dir, f"{formation_name}_summary.csv")
            self._save_metrics_csv(metrics, csv_path)
            
            # 5. Save optimization data if available
            if r['optimization']:
                opt_csv_path = os.path.join(data_dir, f"{formation_name}_optimization.csv")
                self._save_optimization_csv(r['optimization'], opt_csv_path)
        
        # 6. Comparison plot across all formations
        comparison_path = os.path.join(summary_dir, "formation_comparison.png")
        metrics_list = [r['metrics'] for r in results]
        self.dashboard.plot_comparison(metrics_list=metrics_list, save_path=comparison_path)
        
        # 7. Create comprehensive summary report
        summary_report_path = os.path.join(summary_dir, "comprehensive_summary.txt")
        self._save_comprehensive_summary(results, summary_report_path)
        
        # 8. Create master index file
        index_path = os.path.join(results_timestamp_dir, "INDEX.txt")
        self._create_index_file(results, results_timestamp_dir, index_path)
        
        print(f"\n✓ All results saved to: {results_timestamp_dir}")
        print(f"  ├── dashboards/ ........... (PNG visualizations)")
        print(f"  ├── reports/ .............. (TXT detailed reports)")
        print(f"  ├── data/ ................. (JSON & CSV metrics)")
        print(f"  ├── summary/ .............. (Cross-formation comparisons)")
        print(f"  └── INDEX.txt ............. (Navigation file)")
        
        return results_timestamp_dir
    
    def _save_metrics_csv(self, metrics, csv_path):
        """Save metrics as CSV for easy analysis."""
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Formation', metrics.formation_name, ''])
            writer.writerow(['Converged', metrics.converged, ''])
            writer.writerow(['Convergence Time', f'{metrics.convergence_time:.2f}', 's'])
            writer.writerow(['Convergence Rate', f'{metrics.convergence_rate*100:.1f}', '%'])
            writer.writerow(['Total Energy', f'{metrics.total_energy:.2f}', 'J'])
            writer.writerow(['Avg Energy/Agent', f'{metrics.avg_energy_per_agent:.2f}', 'J'])
            writer.writerow(['Total Distance', f'{metrics.total_distance:.2f}', 'm'])
            writer.writerow(['Avg Distance/Agent', f'{metrics.avg_distance_per_agent:.2f}', 'm'])
            writer.writerow(['Collision Count', metrics.collision_count, ''])
            writer.writerow(['Min Distance', f'{metrics.min_distance_observed:.3f}', 'm'])
            writer.writerow(['Avg Final Error', f'{np.mean(metrics.final_errors):.4f}', 'm'])
            writer.writerow(['Simulation Time', f'{metrics.simulation_time:.2f}', 's'])
        print(f"  ✓ CSV saved: {os.path.basename(csv_path)}")
    
    def _save_optimization_csv(self, opt_result, csv_path):
        """Save optimization results as CSV."""
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Total Energy', f'{opt_result["total_energy"]:.2f}', 'J'])
            writer.writerow(['Avg Energy/Agent', f'{opt_result["avg_energy_per_agent"]:.2f}', 'J'])
            writer.writerow(['Convergence Rate', f'{opt_result["convergence_rate"]*100:.1f}', '%'])
            writer.writerow(['Optimization Time', f'{opt_result["opt_time"]:.2f}', 's'])
        print(f"  ✓ Optimization CSV saved: {os.path.basename(csv_path)}")
    
    def _save_comprehensive_summary(self, results, summary_path):
        """Create comprehensive summary across all formations."""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("FORMATION COMPARISON TABLE\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Formation':<15} {'Converged':<12} {'Time(s)':<12} {'Energy(J)':<12}\n")
            f.write(f"{'':15} {'':12} {'':12} {'':12}\n")
            
            for r in results:
                m = r['metrics']
                conv = "✓ YES" if m.converged else "✗ NO"
                f.write(f"{m.formation_name:<15} {conv:<12} {m.convergence_time:<12.2f} {m.total_energy:<12.2f}\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("DETAILED STATISTICS\n")
            f.write("-"*70 + "\n\n")
            
            metrics_list = [r['metrics'] for r in results]
            conv_rate = sum(1 for r in results if r['converged']) / len(results) * 100
            avg_energy = np.mean([m.total_energy for m in metrics_list])
            avg_collisions = np.mean([m.collision_count for m in metrics_list])
            
            f.write(f"Total Simulations: {len(results)}\n")
            f.write(f"Convergence Rate: {conv_rate:.1f}%\n")
            f.write(f"Average Energy/Formation: {avg_energy:.2f}J\n")
            f.write(f"Average Collisions/Formation: {avg_collisions:.1f}\n")
            f.write(f"\nBest Energy Formation: {metrics_list[np.argmin([m.total_energy for m in metrics_list])].formation_name}\n")
            f.write(f"Fastest Convergence: {metrics_list[np.argmin([m.convergence_time for m in metrics_list if m.converged] or [999])].formation_name if any(m.converged for m in metrics_list) else 'N/A'}\n")
        
        print(f"  ✓ Summary saved: {os.path.basename(summary_path)}")
    
    def _create_index_file(self, results, base_dir, index_path):
        """Create index file for easy navigation."""
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESULTS DIRECTORY INDEX\n")
            f.write("="*70 + "\n\n")
            
            f.write("DIRECTORY STRUCTURE\n")
            f.write("-"*70 + "\n")
            f.write("├── dashboards/ ..................... 8-panel visualizations (PNG)\n")
            f.write("├── reports/ ........................ Detailed text reports\n")
            f.write("├── data/ ........................... Raw metrics (JSON & CSV)\n")
            f.write("├── summary/ ........................ Cross-formation analyses\n")
            f.write("└── INDEX.txt ....................... This file\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-"*70 + "\n\n")
            
            for r in results:
                m = r['metrics']
                f.write(f"{m.formation_name.upper()}\n")
                f.write(f"  dashboards/{m.formation_name}_dashboard.png\n")
                f.write(f"  reports/{m.formation_name}_report.txt\n")
                f.write(f"  data/{m.formation_name}_metrics.json\n")
                f.write(f"  data/{m.formation_name}_summary.csv\n")
                if r['optimization']:
                    f.write(f"  data/{m.formation_name}_optimization.csv\n")
                f.write("\n")
            
            f.write("SUMMARY FILES\n")
            f.write("-"*70 + "\n")
            f.write("summary/formation_comparison.png ....... Cross-formation comparison\n")
            f.write("summary/comprehensive_summary.txt ...... All formations summary\n\n")
            
            f.write("HOW TO USE THESE FILES\n")
            f.write("-"*70 + "\n")
            f.write("1. PNG FILES: Open in any image viewer or browser\n")
            f.write("2. TXT FILES: Open in any text editor\n")
            f.write("3. JSON FILES: Parse with Python json.load() or any JSON reader\n")
            f.write("4. CSV FILES: Open in Excel, LibreOffice, or parse with Python\n\n")
            
            f.write("NEXT STEPS\n")
            f.write("-"*70 + "\n")
            f.write("1. View dashboards: Open PNG files in dashboards/\n")
            f.write("2. Read detailed reports: Open TXT files in reports/\n")
            f.write("3. Analyze raw data: Use JSON/CSV files in data/\n")
            f.write("4. Compare formations: View summary/formation_comparison.png\n")
        
        print(f"  ✓ Index created: {os.path.basename(index_path)}")
    
    def benchmark_single_formation(self,
                                   formation_name: str,
                                   num_runs: int = 3):
        """
        Benchmark a single formation over multiple runs.
        
        Args:
            formation_name: Formation to test
            num_runs: Number of repeated runs
        """
        print("\n" + "="*70)
        print(f"BENCHMARK: {formation_name.upper()}")
        print(f"Number of runs: {num_runs}")
        print("="*70)
        
        results = []
        for run in range(num_runs):
            print(f"\n[Run {run+1}/{num_runs}]")
            result = self.run_optimized_formation(formation_name, enable_optimization=True)
            results.append(result)
            time.sleep(1)
        
        # Statistics
        metrics_list = [r['metrics'] for r in results]
        energies = [m.total_energy for m in metrics_list]
        times = [m.convergence_time for m in metrics_list]
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"Formation: {formation_name}")
        print(f"Runs: {num_runs}")
        print(f"\nEnergy (J):")
        print(f"  Mean: {np.mean(energies):.2f}")
        print(f"  Std:  {np.std(energies):.2f}")
        print(f"  Min:  {np.min(energies):.2f}")
        print(f"  Max:  {np.max(energies):.2f}")
        print(f"\nConvergence Time (s):")
        print(f"  Mean: {np.mean(times):.2f}")
        print(f"  Std:  {np.std(times):.2f}")
        print(f"  Min:  {np.min(times):.2f}")
        print(f"  Max:  {np.max(times):.2f}")
        print("="*70)
        
        return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Energy Optimization + Metrics Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python energy_optimization_main.py --formation helix
  python energy_optimization_main.py --all
  python energy_optimization_main.py --benchmark cupid
        """
    )
    parser.add_argument('--formation', type=str, default=None,
                       choices=['helix', 'mandala', 'cupid', 'dragon', 'flag'],
                       help='Single formation to test')
    parser.add_argument('--all', action='store_true',
                       help='Test all formations')
    parser.add_argument('--benchmark', type=str, default=None,
                       help='Benchmark a formation')
    parser.add_argument('--agents', type=int, default=20,
                       help='Number of agents')
    parser.add_argument('--gui', action='store_true',
                       help='Enable PyBullet GUI')
    parser.add_argument('--output', type=str, default=ENERGY_OPT_DIR,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create platform
    platform = EnergyOptimizationPlatform(
        num_agents=args.agents,
        gui=args.gui,
        output_dir=args.output
    )
    
    try:
        if args.all:
            results = platform.run_comparison_study()
        
        elif args.benchmark:
            results = platform.benchmark_single_formation(args.benchmark, num_runs=3)
        
        elif args.formation:
            result = platform.run_optimized_formation(
                args.formation,
                enable_optimization=True,
                visualize=False
            )
        
        else:
            print("Usage: python energy_optimization_main.py --help")
            print("Run with --all for comprehensive test, or --formation <name> for single test")
            return
        
        print("\n✓ All simulations complete!")
        print(f"✓ Results saved to {args.output}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
