"""
Advanced Energy Analysis Tools
Detailed energy profiling, optimization strategies, and efficiency metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class EnergyProfile:
    """Detailed energy profile for an agent."""
    agent_id: int
    kinetic_energy_profile: np.ndarray  # Energy at each timestep
    total_kinetic_energy: float
    total_gravitational_work: float
    total_energy: float
    
    # Efficiency metrics
    energy_per_distance: float
    energy_per_step: float
    peak_power: float


class EnergyAnalyzer:
    """Analyzes energy consumption in detail."""
    
    def __init__(self, time_step: float = 0.01, mass: float = 0.5):
        """
        Initialize analyzer.
        
        Args:
            time_step: Physics time step
            mass: Agent mass
        """
        self.time_step = time_step
        self.mass = mass
    
    def analyze_agent_trajectory(self,
                                trajectory: np.ndarray,
                                velocities: np.ndarray) -> EnergyProfile:
        """
        Analyze energy consumption for agent trajectory.
        
        Args:
            trajectory: (N, 3) trajectory positions
            velocities: (N, 3) velocities
            
        Returns:
            EnergyProfile object
        """
        # Kinetic energy: 0.5 * m * v^2
        speed = np.linalg.norm(velocities, axis=1)
        kinetic_energy = 0.5 * self.mass * speed**2
        total_kinetic = np.sum(kinetic_energy) * self.time_step
        
        # Gravitational work: m * g * Δz
        height_changes = np.diff(trajectory[:, 2])
        gravitational_work = self.mass * 9.81 * np.sum(np.abs(height_changes))
        
        # Distance traveled
        displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        total_distance = np.sum(displacements)
        
        # Total energy
        total_energy = total_kinetic + gravitational_work
        
        # Efficiency metrics
        energy_per_distance = total_energy / max(total_distance, 0.001)
        energy_per_step = total_energy / len(trajectory)
        peak_power = np.max(kinetic_energy) if len(kinetic_energy) > 0 else 0
        
        return EnergyProfile(
            agent_id=-1,  # Set by caller
            kinetic_energy_profile=kinetic_energy,
            total_kinetic_energy=total_kinetic,
            total_gravitational_work=gravitational_work,
            total_energy=total_energy,
            energy_per_distance=energy_per_distance,
            energy_per_step=energy_per_step,
            peak_power=peak_power
        )
    
    def compare_velocity_profiles(self,
                                 trajectories: Dict[str, np.ndarray],
                                 velocities_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Compare energy consumption across different velocity profiles.
        
        Args:
            trajectories: {profile_name: trajectory}
            velocities_dict: {profile_name: velocities}
            
        Returns:
            Comparison results
        """
        results = {}
        
        for profile_name, trajectory in trajectories.items():
            velocities = velocities_dict[profile_name]
            profile = self.analyze_agent_trajectory(trajectory, velocities)
            results[profile_name] = profile
        
        return results
    
    def identify_energy_bottlenecks(self,
                                   kinetic_profile: np.ndarray,
                                   threshold_percentile: float = 90.0) -> Dict:
        """
        Identify timesteps with high energy consumption.
        
        Args:
            kinetic_profile: Energy profile over time
            threshold_percentile: Percentile for bottleneck detection
            
        Returns:
            Bottleneck information
        """
        threshold = np.percentile(kinetic_profile, threshold_percentile)
        bottleneck_steps = np.where(kinetic_profile > threshold)[0]
        
        return {
            'threshold': threshold,
            'bottleneck_steps': bottleneck_steps.tolist(),
            'num_bottlenecks': len(bottleneck_steps),
            'fraction': len(bottleneck_steps) / len(kinetic_profile)
        }
    
    def suggest_optimization_strategy(self,
                                     energy_profile: Dict[str, EnergyProfile]) -> Dict:
        """
        Suggest optimization strategy based on energy analysis.
        
        Args:
            energy_profile: Per-agent energy profiles
            
        Returns:
            Optimization recommendations
        """
        agents = list(energy_profile.values())
        total_energy = sum(a.total_energy for a in agents)
        
        # Identify high-energy agents
        sorted_agents = sorted(agents, key=lambda a: a.total_energy, reverse=True)
        high_energy_agents = [a for a in sorted_agents[:len(agents)//4]]
        high_energy_fraction = sum(a.total_energy for a in high_energy_agents) / total_energy
        
        recommendations = {
            'focus_on_high_energy_agents': len(high_energy_agents),
            'high_energy_fraction': high_energy_fraction,
            'strategy': []
        }
        
        # Analyze efficiency
        avg_efficiency = np.mean([a.energy_per_distance for a in agents])
        max_efficiency = max(a.energy_per_distance for a in agents)
        
        if max_efficiency > 1.5 * avg_efficiency:
            recommendations['strategy'].append(
                'Focus on velocity profile optimization for inefficient agents'
            )
        
        # Peak power analysis
        avg_peak_power = np.mean([a.peak_power for a in agents])
        if avg_peak_power > 100:
            recommendations['strategy'].append(
                'Consider smoother acceleration profiles to reduce peak power'
            )
        
        # Gravitational work
        total_grav = sum(a.total_gravitational_work for a in agents)
        if total_grav > total_energy * 0.3:
            recommendations['strategy'].append(
                'Consider lower altitude formations to reduce gravitational work'
            )
        
        return recommendations


class VelocityProfileComparator:
    """Compares different velocity profile optimization strategies."""
    
    @staticmethod
    def linear_profile(start: np.ndarray,
                       target: np.ndarray,
                       num_steps: int) -> np.ndarray:
        """Linear velocity profile."""
        displacement = target - start
        time_total = num_steps * 0.01
        velocity = displacement / time_total
        return np.tile(velocity, (num_steps, 1))
    
    @staticmethod
    def trapezoidal_profile(start: np.ndarray,
                            target: np.ndarray,
                            num_steps: int,
                            accel_fraction: float = 0.2) -> np.ndarray:
        """Trapezoidal velocity profile (smooth acceleration/deceleration)."""
        displacement = target - start
        distance = np.linalg.norm(displacement)
        direction = displacement / (distance + 1e-6)
        
        # Divide profile into 3 phases: accel, cruise, decel
        accel_steps = int(num_steps * accel_fraction)
        decel_steps = int(num_steps * accel_fraction)
        cruise_steps = num_steps - accel_steps - decel_steps
        
        # Compute max velocity needed
        time_total = num_steps * 0.01
        max_velocity = distance / time_total * 2  # Scaling for accel/decel
        
        velocities = []
        
        # Acceleration phase
        for i in range(accel_steps):
            v = max_velocity * (i / max(accel_steps, 1))
            velocities.append(direction * v)
        
        # Cruise phase
        for i in range(cruise_steps):
            velocities.append(direction * max_velocity)
        
        # Deceleration phase
        for i in range(decel_steps):
            v = max_velocity * (1 - i / max(decel_steps, 1))
            velocities.append(direction * v)
        
        return np.array(velocities)
    
    @staticmethod
    def sinusoidal_profile(start: np.ndarray,
                          target: np.ndarray,
                          num_steps: int) -> np.ndarray:
        """Smooth sinusoidal velocity profile."""
        displacement = target - start
        distance = np.linalg.norm(displacement)
        direction = displacement / (distance + 1e-6)
        
        # Sinusoidal envelope
        time_steps = np.linspace(0, np.pi, num_steps)
        velocity_envelope = np.sin(time_steps)  # 0 → 1 → 0
        
        # Scale by average velocity needed
        time_total = num_steps * 0.01
        avg_velocity = distance / time_total
        
        velocities = []
        for envelope_val in velocity_envelope:
            v = avg_velocity * envelope_val
            velocities.append(direction * v)
        
        return np.array(velocities)
    
    @staticmethod
    def compare_profiles(start: np.ndarray,
                        target: np.ndarray,
                        num_steps: int = 100) -> Dict:
        """
        Compare different velocity profiles.
        
        Args:
            start: Starting position
            target: Target position
            num_steps: Number of time steps
            
        Returns:
            Comparison data
        """
        analyzer = EnergyAnalyzer()
        
        profiles = {
            'linear': VelocityProfileComparator.linear_profile(start, target, num_steps),
            'trapezoidal': VelocityProfileComparator.trapezoidal_profile(start, target, num_steps),
            'sinusoidal': VelocityProfileComparator.sinusoidal_profile(start, target, num_steps)
        }
        
        # Simulate trajectories
        trajectories = {}
        energy_data = {}
        
        for name, velocities in profiles.items():
            # Integrate velocities to get trajectory
            trajectory = np.zeros((num_steps, 3))
            trajectory[0] = start
            for t in range(1, num_steps):
                trajectory[t] = trajectory[t-1] + velocities[t] * 0.01
            
            trajectories[name] = trajectory
            energy_data[name] = analyzer.analyze_agent_trajectory(trajectory, velocities)
        
        return {
            'trajectories': trajectories,
            'profiles': profiles,
            'energy': energy_data
        }


class EnergyVisualization:
    """Visualization tools for energy analysis."""
    
    @staticmethod
    def plot_energy_comparison(energy_profiles: Dict[str, EnergyProfile],
                              save_path: Optional[str] = None):
        """Plot energy comparison across profiles."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(energy_profiles.keys())
        total_energies = [energy_profiles[n].total_energy for n in names]
        kinetic_energies = [energy_profiles[n].total_kinetic_energy for n in names]
        grav_works = [energy_profiles[n].total_gravitational_work for n in names]
        efficiencies = [energy_profiles[n].energy_per_distance for n in names]
        
        # Total energy comparison
        ax = axes[0, 0]
        ax.bar(names, total_energies, color='steelblue', alpha=0.7)
        ax.set_ylabel('Total Energy (J)')
        ax.set_title('Total Energy Consumption')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Energy breakdown
        ax = axes[0, 1]
        ax.bar(names, kinetic_energies, label='Kinetic', alpha=0.7)
        ax.bar(names, grav_works, bottom=kinetic_energies, label='Gravitational', alpha=0.7)
        ax.set_ylabel('Energy (J)')
        ax.set_title('Energy Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Efficiency
        ax = axes[1, 0]
        ax.bar(names, efficiencies, color='green', alpha=0.7)
        ax.set_ylabel('Energy per Distance (J/m)')
        ax.set_title('Energy Efficiency')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Peak power
        ax = axes[1, 1]
        peak_powers = [energy_profiles[n].peak_power for n in names]
        ax.bar(names, peak_powers, color='orange', alpha=0.7)
        ax.set_ylabel('Peak Power (W)')
        ax.set_title('Peak Power Consumption')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved energy comparison to {save_path}")
            
            # Also save data as CSV
            csv_path = save_path.replace('.png', '.csv')
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Profile', 'Total Energy (J)', 'Kinetic Energy (J)', 
                               'Gravitational Work (J)', 'Efficiency (J/m)', 'Peak Power (W)'])
                for name in names:
                    profile = energy_profiles[name]
                    writer.writerow([
                        name,
                        f'{profile.total_energy:.2f}',
                        f'{profile.total_kinetic_energy:.2f}',
                        f'{profile.total_gravitational_work:.2f}',
                        f'{profile.energy_per_distance:.2f}',
                        f'{profile.peak_power:.2f}'
                    ])
            print(f"✓ Saved energy data to {csv_path}")
        
        plt.show()
        return fig


def demo_energy_analysis():
    """Demo of energy analysis tools with automatic result saving."""
    import os
    from datetime import datetime
    
    print("\n" + "="*70)
    print("ENERGY ANALYSIS TOOLS DEMO")
    print("="*70)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"energy_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample trajectory
    start = np.array([0, 0, 0], dtype=float)
    target = np.array([5, 5, 3], dtype=float)
    num_steps = 100
    
    print(f"\nTrajectory: {start} → {target}")
    print(f"Output directory: {output_dir}")
    
    # Compare profiles
    print("\nComparing velocity profiles...")
    comparison = VelocityProfileComparator.compare_profiles(start, target, num_steps)
    
    print(f"\n{'Profile':<15} {'Total Energy':<20} {'Energy/Distance':<20}")
    print("─"*70)
    
    for name, energy_data in comparison['energy'].items():
        print(f"{name:<15} {energy_data.total_energy:<20.2f}J "
              f"{energy_data.energy_per_distance:<20.2f}J/m")
    
    # Save comparison visualization and data
    print("\nSaving results...")
    comparison_png = os.path.join(output_dir, "energy_comparison.png")
    EnergyVisualization.plot_energy_comparison(
        comparison['energy'],
        save_path=comparison_png
    )
    
    # Save detailed report
    report_path = os.path.join(output_dir, "energy_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ENERGY ANALYSIS REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("VELOCITY PROFILE COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(f"Start Position: {start}\n")
        f.write(f"Target Position: {target}\n")
        f.write(f"Number of Steps: {num_steps}\n\n")
        
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Profile':<20} {'Total Energy (J)':<20} {'Efficiency (J/m)':<20}\n")
        f.write("-"*70 + "\n")
        
        for name, energy_data in comparison['energy'].items():
            f.write(f"{name:<20} {energy_data.total_energy:<20.2f} "
                   f"{energy_data.energy_per_distance:<20.2f}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("-"*70 + "\n")
        
        most_efficient = min(comparison['energy'].items(),
                           key=lambda x: x[1].total_energy)
        f.write(f"Most Efficient Profile: {most_efficient[0]}\n")
        f.write(f"Energy Consumed: {most_efficient[1].total_energy:.2f}J\n")
        f.write(f"Efficiency: {most_efficient[1].energy_per_distance:.2f}J/m\n")
    
    print(f"✓ Report saved to {report_path}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"  ├── energy_comparison.png")
    print(f"  ├── energy_comparison.csv")
    print(f"  └── energy_analysis_report.txt")


if __name__ == "__main__":
    demo_energy_analysis()
