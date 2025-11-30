"""
Energy Optimization Platform - Integration Examples
Comprehensive examples showing how to use all components together.
"""

import numpy as np
from energy_optimization_main import EnergyOptimizationPlatform
from velocity_optimizer import VelocityProfileOptimizer, OptimizationConfig
from metrics_dashboard import MetricsCollector, MetricsDashboard
from energy_analysis import VelocityProfileComparator, EnergyAnalyzer, EnergyVisualization
from configuration import Presets, get_config_summary
import os


# ============================================================================
# EXAMPLE 1: Quick Start - Single Formation
# ============================================================================

def example_1_quick_start():
    """
    Simple example: Run a single formation with optimization.
    
    This is the quickest way to get started.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Start - Single Formation")
    print("="*70)
    
    # Create platform
    platform = EnergyOptimizationPlatform(
        num_agents=20,
        gui=False,
        output_dir="results/example_1"
    )
    
    # Run single formation
    result = platform.run_optimized_formation(
        formation_name='helix',
        enable_optimization=True,
        visualize=False
    )
    
    # Print summary
    print(f"\n✓ Simulation complete!")
    print(f"  Converged: {result['converged']}")
    print(f"  Energy: {result['metrics'].total_energy:.2f}J")
    print(f"  Collisions: {result['metrics'].collision_count}")
    
    return result


# ============================================================================
# EXAMPLE 2: Comprehensive Study - All Formations
# ============================================================================

def example_2_all_formations():
    """
    Run all 5 formations with comparison plots.
    
    Good for benchmark and comparison analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comprehensive Study - All Formations")
    print("="*70)
    
    platform = EnergyOptimizationPlatform(
        num_agents=20,
        gui=False,
        output_dir="results/example_2"
    )
    
    # Run all formations
    results = platform.run_comparison_study()
    
    return results


# ============================================================================
# EXAMPLE 3: Energy Analysis - Velocity Profile Comparison
# ============================================================================

def example_3_velocity_profile_comparison():
    """
    Compare different velocity profile optimization strategies.
    
    Shows optimization benefit over baseline.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Velocity Profile Comparison")
    print("="*70)
    
    # Define trajectory
    start = np.array([0.0, 0.0, 0.5])
    target = np.array([5.0, 5.0, 5.0])
    
    print(f"\nTrajectory: {start} → {target}")
    
    # Compare velocity profiles
    comparison = VelocityProfileComparator.compare_profiles(
        start, target, num_steps=100
    )
    
    # Print results
    print(f"\n{'Profile':<15} {'Total Energy':<20} {'Energy/Distance':<20}")
    print("─"*70)
    
    for name, energy_data in comparison['energy'].items():
        print(f"{name:<15} {energy_data.total_energy:<20.2f}J "
              f"{energy_data.energy_per_distance:<20.2f}J/m")
    
    # Find most efficient
    most_efficient = min(comparison['energy'].items(),
                        key=lambda x: x[1].total_energy)
    print(f"\n✓ Most efficient profile: {most_efficient[0]}")
    print(f"  Energy: {most_efficient[1].total_energy:.2f}J")
    
    # Visualize
    EnergyVisualization.plot_energy_comparison(comparison['energy'])
    
    return comparison


# ============================================================================
# EXAMPLE 4: Custom Configuration - Preset
# ============================================================================

def example_4_custom_configuration():
    """
    Use custom configuration presets for different scenarios.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Configuration Presets")
    print("="*70)
    
    # Test different presets
    presets = {
        'Default': Presets.get_default(),
        'Fast': Presets.get_fast(),
        'Accurate': Presets.get_accurate(),
        'Energy-Focused': Presets.get_energy_focused(),
        'Safe': Presets.get_safe()
    }
    
    # Show configuration for each preset
    for preset_name, config in presets.items():
        print(f"\n{preset_name} Preset:")
        print(f"  Time Steps: {config['optimization'].time_steps}")
        print(f"  Max Iterations: {config['optimization'].max_iterations}")
        print(f"  Energy Weight: {config['optimization'].energy_weight}")
        print(f"  Convergence Weight: {config['optimization'].convergence_weight}")
    
    print("\n✓ Presets available for easy customization")
    print("  Use: Presets.get_<name>() to load preset")
    
    return presets


# ============================================================================
# EXAMPLE 5: Advanced - Custom Objective Function
# ============================================================================

def example_5_custom_optimization():
    """
    Create custom optimization with custom objective function.
    
    Shows how to extend optimization for specific needs.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Optimization Setup")
    print("="*70)
    
    # Create custom configuration
    config = OptimizationConfig(
        max_velocity=5.0,
        time_horizon=30.0,
        time_steps=80,
        energy_weight=2.0,  # Emphasize energy savings
        convergence_weight=8.0,
        collision_weight=50.0
    )
    
    print(f"\nCustom Configuration:")
    print(f"  Energy Weight: {config.energy_weight}")
    print(f"  Convergence Weight: {config.convergence_weight}")
    print(f"  Collision Weight: {config.collision_weight}")
    print(f"  Time Horizon: {config.time_horizon}s")
    print(f"  Time Steps: {config.time_steps}")
    
    # Create optimizer with custom config
    optimizer = VelocityProfileOptimizer(config)
    
    # Simple 2-agent example
    start_positions = np.array([
        [-2.0, 0.0, 1.0],
        [2.0, 0.0, 1.0]
    ])
    
    target_positions = np.array([
        [0.0, 3.0, 5.0],
        [0.0, -3.0, 5.0]
    ])
    
    print(f"\nOptimizing 2-agent trajectory...")
    result = optimizer.optimize_swarm(start_positions, target_positions)
    
    print(f"\n✓ Optimization Results:")
    print(f"  Total Energy: {result['total_energy']:.2f}J")
    print(f"  Avg Energy/Agent: {result['avg_energy_per_agent']:.2f}J")
    print(f"  Convergence Rate: {result['convergence_rate']*100:.1f}%")
    
    return result


# ============================================================================
# EXAMPLE 6: Metrics Collection - Advanced
# ============================================================================

def example_6_metrics_analysis():
    """
    Detailed metrics collection and analysis.
    
    Shows how to collect detailed metrics and generate reports.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Metrics Collection and Analysis")
    print("="*70)
    
    # This is a mock example showing metrics structure
    # In real use, metrics are collected during simulation
    
    from metrics_dashboard import SimulationMetrics
    
    # Create mock metrics
    metrics = SimulationMetrics(
        formation_name='helix',
        converged=True,
        convergence_time=42.5,
        convergence_rate=0.95,
        total_energy=142.3,
        avg_energy_per_agent=7.12,
        collision_count=0,
        min_distance_observed=0.38,
        total_distance=230.5,
        avg_distance_per_agent=11.53,
        simulation_time=45.2,
        steps_to_convergence=4250
    )
    
    # Populate error data
    metrics.final_errors = [0.08, 0.12, 0.05] + [0.10]*17
    metrics.energy_profile = {i: 142.3/20 + np.random.rand() for i in range(20)}
    metrics.distance_profile = {i: 230.5/20 + np.random.rand() for i in range(20)}
    
    # Create dashboard
    dashboard = MetricsDashboard(output_dir="results/example_6")
    dashboard.add_metrics(metrics)
    
    # Generate outputs
    print(f"\nGenerating reports...")
    
    # Text report
    report = dashboard.generate_report(
        metrics,
        save_path="results/example_6/helix_report.txt"
    )
    print(f"✓ Text report generated")
    
    # Visualization
    dashboard.plot_single_simulation(
        metrics,
        save_path="results/example_6/helix_dashboard.png"
    )
    print(f"✓ Dashboard visualization generated")
    
    # JSON export
    dashboard.export_metrics_json(
        metrics,
        save_path="results/example_6/helix_metrics.json"
    )
    print(f"✓ JSON export generated")
    
    return dashboard, metrics


# ============================================================================
# EXAMPLE 7: Benchmarking
# ============================================================================

def example_7_benchmarking():
    """
    Run benchmark studies with statistics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Benchmarking Single Formation")
    print("="*70)
    
    platform = EnergyOptimizationPlatform(
        num_agents=20,
        output_dir="results/example_7"
    )
    
    # Benchmark helix formation
    results = platform.benchmark_single_formation(
        formation_name='helix',
        num_runs=3
    )
    
    return results


# ============================================================================
# EXAMPLE 8: Scalability Analysis
# ============================================================================

def example_8_scalability():
    """
    Test platform scalability with different agent counts.
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Scalability Analysis")
    print("="*70)
    
    agent_counts = [10, 15, 20, 25]
    results = {}
    
    for num_agents in agent_counts:
        print(f"\nTesting with {num_agents} agents...")
        
        platform = EnergyOptimizationPlatform(
            num_agents=num_agents,
            gui=False,
            output_dir=f"results/example_8/agents_{num_agents}"
        )
        
        result = platform.run_optimized_formation(
            'helix',
            enable_optimization=True
        )
        
        results[num_agents] = result['metrics']
    
    # Print scalability summary
    print(f"\n{'Agents':<10} {'Energy(J)':<15} {'Time(s)':<15} {'Collisions':<15}")
    print("─"*60)
    
    for num_agents in agent_counts:
        m = results[num_agents]
        print(f"{num_agents:<10} {m.total_energy:<15.2f} "
              f"{m.convergence_time:<15.2f} {m.collision_count:<15}")
    
    return results


# ============================================================================
# EXAMPLE 9: Configuration Summary
# ============================================================================

def example_9_configuration_summary():
    """
    Display comprehensive configuration information.
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Configuration Summary")
    print("="*70)
    
    # Get default configuration
    config = Presets.get_default()
    
    # Print summary
    summary = get_config_summary(config)
    print(summary)
    
    return config


# ============================================================================
# MAIN - Run All Examples
# ============================================================================

def run_all_examples():
    """Run all examples sequentially."""
    print("\n" + "="*70)
    print("RUNNING ALL INTEGRATION EXAMPLES")
    print("="*70)
    
    examples = [
        ("Quick Start", example_1_quick_start),
        ("All Formations", example_2_all_formations),
        ("Velocity Profiles", example_3_velocity_profile_comparison),
        ("Custom Config", example_4_custom_configuration),
        ("Custom Optimization", example_5_custom_optimization),
        ("Metrics Analysis", example_6_metrics_analysis),
        ("Benchmarking", example_7_benchmarking),
        ("Scalability", example_8_scalability),
        ("Config Summary", example_9_configuration_summary)
    ]
    
    results = {}
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            print(f"\n[{i}/{len(examples)}] Running {name}...")
            result = example_func()
            results[name] = result
            print(f"✓ {name} complete")
        except KeyboardInterrupt:
            print(f"\nInterrupted at {name}")
            break
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def run_single_example(example_num: int):
    """Run a single example by number."""
    examples = {
        1: example_1_quick_start,
        2: example_2_all_formations,
        3: example_3_velocity_profile_comparison,
        4: example_4_custom_configuration,
        5: example_5_custom_optimization,
        6: example_6_metrics_analysis,
        7: example_7_benchmarking,
        8: example_8_scalability,
        9: example_9_configuration_summary
    }
    
    if example_num not in examples:
        print(f"Example {example_num} not found. Available: 1-{len(examples)}")
        return
    
    print(f"\nRunning Example {example_num}...")
    return examples[example_num]()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            run_single_example(example_num)
        except ValueError:
            print("Usage: python integration_examples.py [example_number]")
            print("Examples: 1-9")
    else:
        # Run all examples
        run_all_examples()
