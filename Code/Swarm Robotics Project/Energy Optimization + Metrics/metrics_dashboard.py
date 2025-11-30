"""
Metrics Dashboard
Comprehensive metrics collection, analysis, and visualization for swarm simulations.
Tracks energy, collisions, convergence time, and generates reports.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime
import os


@dataclass
class SimulationMetrics:
    """Container for simulation metrics."""
    
    # Identification
    formation_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Convergence metrics
    converged: bool = False
    convergence_time: float = 0.0  # seconds
    convergence_rate: float = 0.0  # percentage (0-1)
    final_errors: List[float] = field(default_factory=list)  # per agent
    
    # Energy metrics
    total_energy: float = 0.0  # Joules
    avg_energy_per_agent: float = 0.0
    energy_profile: Dict[int, float] = field(default_factory=dict)  # agent_id -> energy
    
    # Collision metrics
    collision_count: int = 0
    collision_events: List[Tuple[float, int, int]] = field(default_factory=list)  # (time, agent1, agent2)
    min_distance_observed: float = float('inf')
    
    # Distance metrics
    total_distance: float = 0.0  # total path length
    avg_distance_per_agent: float = 0.0
    distance_profile: Dict[int, float] = field(default_factory=dict)  # agent_id -> distance
    
    # Simulation metrics
    simulation_time: float = 0.0  # wall clock time
    steps_to_convergence: int = 0
    max_steps: int = 0
    
    # Trajectory data (for advanced analysis)
    position_history: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    velocity_history: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    energy_history: List[float] = field(default_factory=list)


class MetricsCollector:
    """Collects metrics during simulation."""
    
    def __init__(self, num_agents: int, formation_name: str):
        """
        Initialize metrics collector.
        
        Args:
            num_agents: Number of agents in swarm
            formation_name: Name of formation
        """
        self.num_agents = num_agents
        self.formation_name = formation_name
        
        self.metrics = SimulationMetrics(formation_name=formation_name)
        self.collision_pairs = set()  # Track unique collision pairs
        self.step_count = 0
        
    def record_step(self,
                   agents: List,
                   step: int,
                   max_steps: int):
        """
        Record metrics for one simulation step.
        
        Args:
            agents: List of agent objects
            step: Current step number
            max_steps: Maximum steps
        """
        self.step_count = step
        
        # Update trajectories
        for agent in agents:
            if agent.agent_id not in self.metrics.position_history:
                self.metrics.position_history[agent.agent_id] = []
                self.metrics.velocity_history[agent.agent_id] = []
            
            self.metrics.position_history[agent.agent_id].append(agent.position.copy())
            self.metrics.velocity_history[agent.agent_id].append(agent.velocity.copy())
        
        # Check for collisions
        positions = np.array([agent.position for agent in agents])
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = np.linalg.norm(positions[i] - positions[j])
                self.metrics.min_distance_observed = min(
                    self.metrics.min_distance_observed, dist
                )
                
                if dist < 0.25:  # collision threshold
                    pair = (min(i, j), max(i, j))
                    if pair not in self.collision_pairs:
                        self.collision_pairs.add(pair)
                        self.metrics.collision_events.append((step, i, j))
                        self.metrics.collision_count += 1
    
    def record_completion(self,
                         agents: List,
                         converged: bool,
                         convergence_time: float,
                         sim_time: float,
                         max_steps: int):
        """
        Record metrics when simulation completes.
        
        Args:
            agents: List of agents
            converged: Whether swarm converged
            convergence_time: Time to convergence (seconds)
            sim_time: Wall clock simulation time
            max_steps: Maximum steps
        """
        self.metrics.converged = converged
        self.metrics.convergence_time = convergence_time
        self.metrics.simulation_time = sim_time
        self.metrics.steps_to_convergence = self.step_count
        self.metrics.max_steps = max_steps
        
        # Collect final metrics from agents
        total_energy = 0.0
        total_distance = 0.0
        final_errors = []
        
        for agent in agents:
            energy = agent.energy_consumed
            distance = agent.distance_traveled
            error = np.linalg.norm(agent.position - agent.target_position)
            
            self.metrics.energy_profile[agent.agent_id] = energy
            self.metrics.distance_profile[agent.agent_id] = distance
            
            total_energy += energy
            total_distance += distance
            final_errors.append(error)
        
        self.metrics.total_energy = total_energy
        self.metrics.avg_energy_per_agent = total_energy / len(agents)
        self.metrics.total_distance = total_distance
        self.metrics.avg_distance_per_agent = total_distance / len(agents)
        self.metrics.final_errors = final_errors
        
        # Convergence rate
        converged_agents = sum(1 for e in final_errors if e < 0.15)
        self.metrics.convergence_rate = converged_agents / len(agents)
    
    def get_metrics(self) -> SimulationMetrics:
        """Get collected metrics."""
        return self.metrics


class MetricsDashboard:
    """Dashboard for visualizing and comparing simulation metrics."""
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize dashboard.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        self.metrics_list: List[SimulationMetrics] = []
        
    def add_metrics(self, metrics: SimulationMetrics):
        """Add metrics from a simulation."""
        self.metrics_list.append(metrics)
    
    def plot_single_simulation(self,
                              metrics: SimulationMetrics,
                              save_path: Optional[str] = None):
        """
        Plot metrics from a single simulation.
        
        Args:
            metrics: SimulationMetrics object
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Convergence over time
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics.position_history:
            agents_with_history = list(metrics.position_history.keys())
            if agents_with_history:
                agent_id = agents_with_history[0]
                positions = np.array(metrics.position_history[agent_id])
                distances = np.linalg.norm(
                    positions - positions[0],
                    axis=1
                )
                ax1.plot(distances, linewidth=2)
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Distance from Start (m)')
                ax1.set_title('Sample Agent Trajectory Distance')
                ax1.grid(True, alpha=0.3)
        
        # 2. Energy per agent
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics.energy_profile:
            agents = sorted(metrics.energy_profile.keys())
            energies = [metrics.energy_profile[a] for a in agents]
            ax2.bar(range(len(agents)), energies, color='steelblue', alpha=0.7)
            ax2.axhline(metrics.avg_energy_per_agent, color='red', linestyle='--',
                       label=f'Avg: {metrics.avg_energy_per_agent:.2f}J')
            ax2.set_xlabel('Agent ID')
            ax2.set_ylabel('Energy (J)')
            ax2.set_title('Energy Consumption per Agent')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Convergence summary
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        summary_text = f"""
CONVERGENCE SUMMARY
{'='*30}
Formation: {metrics.formation_name}
Converged: {'✓ YES' if metrics.converged else '✗ NO'}
Convergence Rate: {metrics.convergence_rate*100:.1f}%
Convergence Time: {metrics.convergence_time:.2f}s
Steps to Conv: {metrics.steps_to_convergence}
        """
        ax3.text(0.1, 0.5, summary_text, fontfamily='monospace',
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Final position errors
        ax4 = fig.add_subplot(gs[1, 0])
        if metrics.final_errors:
            ax4.hist(metrics.final_errors, bins=10, color='coral', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(metrics.final_errors), color='red', linestyle='--',
                       label=f'Mean: {np.mean(metrics.final_errors):.3f}m')
            ax4.set_xlabel('Final Position Error (m)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Final Errors')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Distance traveled per agent
        ax5 = fig.add_subplot(gs[1, 1])
        if metrics.distance_profile:
            agents = sorted(metrics.distance_profile.keys())
            distances = [metrics.distance_profile[a] for a in agents]
            ax5.bar(range(len(agents)), distances, color='lightgreen', alpha=0.7)
            ax5.axhline(metrics.avg_distance_per_agent, color='red', linestyle='--',
                       label=f'Avg: {metrics.avg_distance_per_agent:.2f}m')
            ax5.set_xlabel('Agent ID')
            ax5.set_ylabel('Distance (m)')
            ax5.set_title('Path Length per Agent')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Energy summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        energy_text = f"""
ENERGY SUMMARY
{'='*30}
Total Energy: {metrics.total_energy:.2f}J
Avg/Agent: {metrics.avg_energy_per_agent:.2f}J
Total Distance: {metrics.total_distance:.2f}m
Avg/Agent: {metrics.avg_distance_per_agent:.2f}m
Energy/Distance: {metrics.total_energy/max(metrics.total_distance, 0.001):.2f}J/m
        """
        ax6.text(0.1, 0.5, energy_text, fontfamily='monospace',
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 7. Collision events timeline
        ax7 = fig.add_subplot(gs[2, :2])
        if metrics.collision_events:
            collision_times = [e[0] for e in metrics.collision_events]
            ax7.scatter(collision_times, [1]*len(collision_times), s=100, color='red', alpha=0.7)
            ax7.set_ylim(0.5, 1.5)
            ax7.set_xlim(0, metrics.steps_to_convergence)
            ax7.set_xlabel('Step')
            ax7.set_ylabel('Collisions Detected')
            ax7.set_title(f'Collision Events Timeline ({metrics.collision_count} total)')
            ax7.grid(True, alpha=0.3, axis='x')
        else:
            ax7.text(0.5, 0.5, 'No Collisions Detected ✓',
                    ha='center', va='center', fontsize=14, color='green',
                    transform=ax7.transAxes)
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.axis('off')
        
        # 8. Collision summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        collision_text = f"""
COLLISION SUMMARY
{'='*30}
Total Collisions: {metrics.collision_count}
Min Distance: {metrics.min_distance_observed:.3f}m
Safe Threshold: 0.25m
Status: {'✓ SAFE' if metrics.collision_count == 0 else '✗ UNSAFE'}
        """
        ax8.text(0.1, 0.5, collision_text, fontfamily='monospace',
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round',
                         facecolor='lightgreen' if metrics.collision_count == 0 else 'lightcoral',
                         alpha=0.5))
        
        fig.suptitle(f'{metrics.formation_name.upper()} Formation - Simulation Metrics',
                    fontsize=16, fontweight='bold')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dashboard to {save_path}")
        
        plt.show()
        return fig
    
    def plot_comparison(self,
                       metrics_list: Optional[List[SimulationMetrics]] = None,
                       save_path: Optional[str] = None):
        """
        Plot comparison of multiple simulations.
        
        Args:
            metrics_list: List of SimulationMetrics (uses internal if None)
            save_path: Path to save figure
        """
        if metrics_list is None:
            metrics_list = self.metrics_list
        
        if not metrics_list:
            print("No metrics to compare")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Swarm Simulation Comparison', fontsize=16, fontweight='bold')
        
        formations = [m.formation_name for m in metrics_list]
        
        # 1. Convergence time
        ax = axes[0, 0]
        conv_times = [m.convergence_time if m.converged else m.simulation_time for m in metrics_list]
        colors = ['green' if m.converged else 'red' for m in metrics_list]
        ax.bar(formations, conv_times, color=colors, alpha=0.7)
        ax.set_ylabel('Time (s)')
        ax.set_title('Convergence Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Energy consumption
        ax = axes[0, 1]
        energies = [m.total_energy for m in metrics_list]
        ax.bar(formations, energies, color='steelblue', alpha=0.7)
        ax.set_ylabel('Energy (J)')
        ax.set_title('Total Energy Consumption')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Convergence rate
        ax = axes[0, 2]
        conv_rates = [m.convergence_rate * 100 for m in metrics_list]
        ax.bar(formations, conv_rates, color='green', alpha=0.7)
        ax.set_ylabel('Convergence Rate (%)')
        ax.set_title('Convergence Rate')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(conv_rates):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
        
        # 4. Collision count
        ax = axes[1, 0]
        collisions = [m.collision_count for m in metrics_list]
        colors = ['green' if c == 0 else 'red' for c in collisions]
        ax.bar(formations, collisions, color=colors, alpha=0.7)
        ax.set_ylabel('Collision Count')
        ax.set_title('Collision Events')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Energy efficiency (energy per distance)
        ax = axes[1, 1]
        efficiency = [
            m.total_energy / max(m.total_distance, 0.001)
            for m in metrics_list
        ]
        ax.bar(formations, efficiency, color='orange', alpha=0.7)
        ax.set_ylabel('Energy/Distance (J/m)')
        ax.set_title('Energy Efficiency')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for m in metrics_list:
            table_data.append([
                m.formation_name[:10],
                f"{m.convergence_time:.1f}s" if m.converged else "timeout",
                f"{m.total_energy:.1f}J",
                f"{m.convergence_rate*100:.0f}%",
                f"{m.collision_count}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Formation', 'Time', 'Energy', 'Conv.', 'Coll.'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison to {save_path}")
        
        plt.show()
        return fig
    
    def generate_report(self,
                       metrics: SimulationMetrics,
                       save_path: Optional[str] = None) -> str:
        """
        Generate text report of metrics.
        
        Args:
            metrics: SimulationMetrics object
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = f"""
{'='*70}
SWARM SIMULATION METRICS REPORT
{'='*70}

FORMATION: {metrics.formation_name.upper()}
TIMESTAMP: {metrics.timestamp}

{'─'*70}
CONVERGENCE METRICS
{'─'*70}
Status:                  {'✓ CONVERGED' if metrics.converged else '✗ TIMEOUT'}
Convergence Time:        {metrics.convergence_time:.2f}s
Convergence Rate:        {metrics.convergence_rate*100:.1f}% ({int(metrics.convergence_rate * len(metrics.final_errors))}/{len(metrics.final_errors)} agents)
Steps to Convergence:    {metrics.steps_to_convergence}/{metrics.max_steps}

Final Errors (per agent):
  Mean:                  {np.mean(metrics.final_errors):.4f}m
  Median:                {np.median(metrics.final_errors):.4f}m
  Std Dev:               {np.std(metrics.final_errors):.4f}m
  Max:                   {np.max(metrics.final_errors):.4f}m
  Min:                   {np.min(metrics.final_errors):.4f}m

{'─'*70}
ENERGY METRICS
{'─'*70}
Total Energy:            {metrics.total_energy:.2f}J
Avg Energy per Agent:    {metrics.avg_energy_per_agent:.2f}J
Min Energy:              {np.min(list(metrics.energy_profile.values())):.2f}J
Max Energy:              {np.max(list(metrics.energy_profile.values())):.2f}J

Energy Distribution:
"""
        
        for agent_id in sorted(metrics.energy_profile.keys())[:10]:  # Show first 10
            report += f"  Agent {agent_id:2d}: {metrics.energy_profile[agent_id]:7.2f}J\n"
        if len(metrics.energy_profile) > 10:
            report += f"  ... and {len(metrics.energy_profile) - 10} more agents\n"
        
        report += f"""
{'─'*70}
COLLISION METRICS
{'─'*70}
Total Collision Events:  {metrics.collision_count}
Collision Pairs:         {len(set((e[1], e[2]) for e in metrics.collision_events))}
Min Distance Observed:   {metrics.min_distance_observed:.4f}m
Safety Threshold:        0.25m
Status:                  {'✓ SAFE' if metrics.collision_count == 0 else '✗ UNSAFE'}

"""
        
        if metrics.collision_events:
            report += "Collision Events:\n"
            for step, agent1, agent2 in metrics.collision_events[:10]:
                report += f"  Step {step}: Agents {agent1} ↔ {agent2}\n"
            if len(metrics.collision_events) > 10:
                report += f"  ... and {len(metrics.collision_events) - 10} more events\n"
        else:
            report += "No collision events detected.\n"
        
        report += f"""
{'─'*70}
DISTANCE METRICS
{'─'*70}
Total Distance:          {metrics.total_distance:.2f}m
Avg Distance per Agent:  {metrics.avg_distance_per_agent:.2f}m
Min Distance:            {np.min(list(metrics.distance_profile.values())):.2f}m
Max Distance:            {np.max(list(metrics.distance_profile.values())):.2f}m

{'─'*70}
EFFICIENCY METRICS
{'─'*70}
Energy per Distance:     {metrics.total_energy / max(metrics.total_distance, 0.001):.2f}J/m
Wall Clock Time:         {metrics.simulation_time:.2f}s
Steps per Second:        {metrics.steps_to_convergence / max(metrics.simulation_time, 0.001):.1f}

{'='*70}
"""
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ Saved report to {save_path}")
        
        return report
    
    def export_metrics_json(self,
                           metrics: SimulationMetrics,
                           save_path: str):
        """
        Export metrics to JSON for further analysis.
        
        Args:
            metrics: SimulationMetrics object
            save_path: Path to save JSON
        """
        # Convert to serializable format
        export_data = {
            'formation': metrics.formation_name,
            'timestamp': metrics.timestamp,
            'converged': metrics.converged,
            'convergence_time': metrics.convergence_time,
            'convergence_rate': float(metrics.convergence_rate),
            'total_energy': metrics.total_energy,
            'avg_energy_per_agent': metrics.avg_energy_per_agent,
            'collision_count': metrics.collision_count,
            'min_distance_observed': float(metrics.min_distance_observed),
            'total_distance': metrics.total_distance,
            'avg_distance_per_agent': metrics.avg_distance_per_agent,
            'simulation_time': metrics.simulation_time,
            'steps_to_convergence': metrics.steps_to_convergence,
            'final_errors': [float(e) for e in metrics.final_errors],
            'energy_profile': {str(k): float(v) for k, v in metrics.energy_profile.items()},
            'distance_profile': {str(k): float(v) for k, v in metrics.distance_profile.items()},
            'collision_events': [[int(e[0]), int(e[1]), int(e[2])] for e in metrics.collision_events]
        }
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                   exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported metrics to {save_path}")


def demo_metrics_dashboard():
    """Demo of metrics dashboard."""
    print("\n" + "="*70)
    print("METRICS DASHBOARD DEMO")
    print("="*70)
    
    # Create mock metrics for demonstration
    metrics1 = SimulationMetrics(
        formation_name='helix',
        converged=True,
        convergence_time=45.3,
        convergence_rate=0.95,
        total_energy=156.2,
        avg_energy_per_agent=7.81,
        collision_count=0,
        min_distance_observed=0.42,
        total_distance=234.5,
        avg_distance_per_agent=11.73
    )
    metrics1.final_errors = [0.08, 0.12, 0.05, 0.15] + [0.10] * 16
    metrics1.energy_profile = {i: 156.2/20 + np.random.rand() for i in range(20)}
    metrics1.distance_profile = {i: 234.5/20 + np.random.rand() for i in range(20)}
    
    metrics2 = SimulationMetrics(
        formation_name='mandala',
        converged=True,
        convergence_time=52.1,
        convergence_rate=0.90,
        total_energy=182.3,
        avg_energy_per_agent=9.12,
        collision_count=2,
        min_distance_observed=0.31,
        total_distance=256.3,
        avg_distance_per_agent=12.82
    )
    metrics2.final_errors = [0.12, 0.18, 0.08, 0.20] + [0.14] * 16
    metrics2.energy_profile = {i: 182.3/20 + np.random.rand() for i in range(20)}
    metrics2.distance_profile = {i: 256.3/20 + np.random.rand() for i in range(20)}
    
    # Create dashboard
    dashboard = MetricsDashboard()
    dashboard.add_metrics(metrics1)
    dashboard.add_metrics(metrics2)
    
    # Generate reports
    print("\nGenerating reports...")
    report1 = dashboard.generate_report(metrics1)
    print(report1)
    
    return dashboard, [metrics1, metrics2]


if __name__ == "__main__":
    demo_metrics_dashboard()
