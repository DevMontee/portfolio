"""
Real-time Contingency Monitoring Dashboard
Extends your existing formation dashboards with contingency metrics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime


class ContingencyDashboard:
    """Real-time visualization of contingency status and reserves"""
    
    def __init__(self, formation_name, num_agents=20):
        self.formation_name = formation_name
        self.num_agents = num_agents
        self.history = {
            'steps': [],
            'contingency_level': [],
            'energy_usage': [],
            'min_distance': [],
            'position_errors': [],
            'active_agents': [],
            'reserves_status': []
        }
    
    def record_state(self, step, contingency_level, energies, min_dist, 
                    pos_errors, active_agents, energy_limit=4.5):
        """Record contingency state at each simulation step"""
        self.history['steps'].append(step)
        self.history['contingency_level'].append(contingency_level.value)
        self.history['energy_usage'].append(energies.mean())
        self.history['min_distance'].append(min_dist)
        self.history['position_errors'].append(pos_errors.mean())
        self.history['active_agents'].append(active_agents)
        
        # Calculate reserve status (% of available space/energy)
        reserve_status = 100 - (energies.mean() / energy_limit * 100)
        self.history['reserves_status'].append(reserve_status)
    
    def create_contingency_dashboard(self, figsize=(16, 12)):
        """Create comprehensive contingency monitoring dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Contingency Level Timeline
        ax1 = fig.add_subplot(gs[0, :2])
        level_colors = ['green', 'yellow', 'orange', 'red']
        level_names = ['NOMINAL', 'CAUTION', 'WARNING', 'CRITICAL']
        
        for i, level in enumerate(self.history['contingency_level']):
            color = level_colors[level]
            ax1.scatter(self.history['steps'][i], level, c=color, s=100, zorder=2)
        
        ax1.set_ylim(-0.5, 3.5)
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(level_names)
        ax1.set_xlabel('Simulation Step')
        ax1.set_title(f'{self.formation_name} - Contingency Level Timeline', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add color bands for reference
        ax1.axhspan(0, 1, alpha=0.1, color='green', label='Safe')
        ax1.axhspan(1, 2, alpha=0.1, color='yellow', label='Caution')
        ax1.axhspan(2, 3, alpha=0.1, color='orange', label='Warning')
        ax1.axhspan(3, 3.5, alpha=0.1, color='red', label='Critical')
        
        # 2. Contingency Status Summary Box
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        current_level = self.history['contingency_level'][-1] if self.history['contingency_level'] else 0
        current_level_name = level_names[int(current_level)]
        max_level_reached = max(self.history['contingency_level']) if self.history['contingency_level'] else 0
        
        summary_text = f"""
CONTINGENCY STATUS
================
Current Level: {current_level_name}
Max Level: {level_names[int(max_level_reached)]}
Total Steps: {len(self.history['steps'])}
Warnings Issued: {sum(1 for x in self.history['contingency_level'] if x >= 2)}
Critical Events: {sum(1 for x in self.history['contingency_level'] if x >= 3)}
"""
        ax2.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Energy Usage vs Reserve Threshold
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.history['steps'], self.history['energy_usage'], 'b-', label='Avg Energy', linewidth=2)
        ax3.axhline(y=4.5 * 0.85, color='orange', linestyle='--', label='Warning Threshold (85%)', linewidth=2)
        ax3.axhline(y=4.5 * 0.90, color='red', linestyle='--', label='Critical Threshold (90%)', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Energy (J)')
        ax3.set_title('Energy Consumption with Reserve Buffers', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Safety Buffer Margin (min distance)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.history['steps'], self.history['min_distance'], 'g-', label='Min Distance', linewidth=2)
        ax4.axhline(y=0.25, color='yellow', linestyle='--', label='Safe Threshold', linewidth=2)
        ax4.axhline(y=0.30, color='green', linestyle='--', label='Buffer Threshold (5cm)', linewidth=2)
        ax4.axhline(y=0.20, color='red', linestyle='--', label='Critical', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Distance (m)')
        ax4.set_title('Collision Buffer Margin', fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.15, 0.35])
        
        # 5. Position Error Monitoring
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(self.history['steps'], np.array(self.history['position_errors']) * 100, 'b-', 
                label='Mean Position Error', linewidth=2)
        ax5.axhline(y=15, color='orange', linestyle='--', label='Warning (15cm)', linewidth=2)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Error (cm)')
        ax5.set_title('Position Error vs Threshold', fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Agent Status (active/inactive)
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(self.history['steps'], self.history['active_agents'], 'b-', linewidth=2)
        ax6.axhline(y=self.num_agents * 0.8, color='orange', linestyle='--', label='80% threshold', linewidth=2)
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Active Agents')
        ax6.set_title('Agent Availability', fontweight='bold')
        ax6.set_ylim([0, self.num_agents * 1.05])
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. Reserve Margin (Energy + Distance)
        ax7 = fig.add_subplot(gs[2, 1])
        
        # Normalize metrics to 0-100 scale
        energy_margin = [100 - (e / 4.5 * 100) for e in self.history['energy_usage']]
        distance_margin = [(d - 0.20) / (0.35 - 0.20) * 100 for d in self.history['min_distance']]
        distance_margin = [max(0, min(100, m)) for m in distance_margin]
        
        ax7.fill_between(self.history['steps'], energy_margin, alpha=0.5, label='Energy Reserve')
        ax7.fill_between(self.history['steps'], distance_margin, alpha=0.5, label='Distance Reserve')
        ax7.axhline(y=15, color='red', linestyle='--', label='Critical Reserve', linewidth=2)
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Reserve Margin (%)')
        ax7.set_title('Combined Reserve Margin', fontweight='bold')
        ax7.set_ylim([0, 100])
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Risk Assessment Heatmap
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calculate risk scores
        avg_contingency = np.mean(self.history['contingency_level']) if self.history['contingency_level'] else 0
        max_energy_percent = (np.max(self.history['energy_usage']) / 4.5 * 100) if self.history['energy_usage'] else 0
        min_distance_value = np.min(self.history['min_distance']) if self.history['min_distance'] else 0.5
        agent_loss_percent = (1 - np.min(self.history['active_agents']) / self.num_agents * 100) if self.history['active_agents'] else 0
        
        risk_score = (
            avg_contingency * 25 +
            max_energy_percent * 0.25 +
            (1 - min(min_distance_value / 0.35, 1)) * 25 +
            agent_loss_percent * 0.25
        )
        
        risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 60 else "HIGH"
        risk_color = "green" if risk_score < 30 else "orange" if risk_score < 60 else "red"
        
        risk_text = f"""
RISK ASSESSMENT
===============
Overall Risk: {risk_level}
Risk Score: {risk_score:.1f}%

Metrics:
- Avg Contingency: {avg_contingency:.2f}
- Max Energy: {max_energy_percent:.1f}%
- Min Distance: {min_distance_value:.3f}m
- Agent Loss: {agent_loss_percent:.1f}%
"""
        ax8.text(0.1, 0.5, risk_text, fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.3))
        
        plt.suptitle(f'{self.formation_name} Formation - Contingency Reserve Dashboard', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        return fig
    
    def save_dashboard(self, filepath):
        """Save dashboard figure"""
        fig = self.create_contingency_dashboard()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to {filepath}")
        plt.close(fig)
    
    def print_contingency_summary(self):
        """Print text summary of contingency events"""
        print("\n" + "=" * 70)
        print(f"CONTINGENCY SUMMARY: {self.formation_name} Formation")
        print("=" * 70)
        
        if not self.history['steps']:
            print("No data recorded")
            return
        
        level_names = ['NOMINAL', 'CAUTION', 'WARNING', 'CRITICAL']
        level_colors = {'NOMINAL': '\033[92m', 'CAUTION': '\033[93m', 
                       'WARNING': '\033[94m', 'CRITICAL': '\033[91m', 'END': '\033[0m'}
        
        # Count contingency events
        level_counts = [0, 0, 0, 0]
        for level in self.history['contingency_level']:
            level_counts[int(level)] += 1
        
        print("\nContingency Level Distribution:")
        for i, count in enumerate(level_counts):
            color = level_colors.get(level_names[i], '')
            reset = level_colors['END']
            pct = count / len(self.history['steps']) * 100
            print(f"  {color}{level_names[i]:10s}{reset}: {count:5d} steps ({pct:5.1f}%)")
        
        # Energy statistics
        print("\nEnergy Statistics:")
        print(f"  Min: {np.min(self.history['energy_usage']):.2f} J")
        print(f"  Max: {np.max(self.history['energy_usage']):.2f} J")
        print(f"  Mean: {np.mean(self.history['energy_usage']):.2f} J")
        print(f"  Reserve Threshold: 3.825 J (85% of 4.5J)")
        
        # Distance statistics
        print("\nDistance (Safety Buffer) Statistics:")
        print(f"  Min: {np.min(self.history['min_distance']):.4f} m")
        print(f"  Max: {np.max(self.history['min_distance']):.4f} m")
        print(f"  Mean: {np.mean(self.history['min_distance']):.4f} m")
        print(f"  Safe Threshold: 0.25 m, Buffer Threshold: 0.30 m")
        
        # Position error statistics
        print("\nPosition Error Statistics:")
        errors_cm = np.array(self.history['position_errors']) * 100
        print(f"  Min: {np.min(errors_cm):.3f} cm")
        print(f"  Max: {np.max(errors_cm):.3f} cm")
        print(f"  Mean: {np.mean(errors_cm):.3f} cm")
        print(f"  Warning Threshold: 15 cm")
        
        # Agent availability
        print("\nAgent Availability:")
        print(f"  Min Active: {np.min(self.history['active_agents'])} agents")
        print(f"  Total: {self.num_agents} agents")
        agent_loss = (1 - np.min(self.history['active_agents']) / self.num_agents) * 100
        print(f"  Max Loss: {agent_loss:.1f}%")
        
        print("\n" + "=" * 70)


# Example usage with dashboard integration
if __name__ == "__main__":
    # Simulate contingency monitoring data
    dashboard = ContingencyDashboard("CUPID", num_agents=20)
    
    # Generate synthetic data (in real use, this comes from simulation)
    steps = np.linspace(0, 10000, 100, dtype=int)
    
    for i, step in enumerate(steps):
        # Synthetic contingency levels
        if step < 2000:
            level = 0  # NOMINAL
        elif step < 5000:
            level = 1 if np.random.random() < 0.3 else 0  # CAUTION occasionally
        elif step < 8000:
            level = 2 if np.random.random() < 0.4 else 1  # WARNING
        else:
            level = 2 if np.random.random() < 0.5 else 3  # WARNING/CRITICAL
        
        # Synthetic metrics
        energies = np.random.normal(2.8 + step/10000 * 1.5, 0.3, 20)
        energies = np.clip(energies, 0, 4.5)
        min_dist = 0.294 - step/10000 * 0.05 + np.random.normal(0, 0.01)
        pos_errors = np.random.normal(0.008 + step/10000 * 0.002, 0.005, 20)
        active_agents = 20 - int(step / 2000)
        
        dashboard.record_state(step, type('obj', (object,), {'value': int(level)})(),
                             energies, min_dist, pos_errors, active_agents)
    
    # Display results
    dashboard.print_contingency_summary()
    dashboard.save_dashboard('/home/claude/contingency_dashboard.png')
    print("\n✅ Dashboard created at /home/claude/contingency_dashboard.png")
