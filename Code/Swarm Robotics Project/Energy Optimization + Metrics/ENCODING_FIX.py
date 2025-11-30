"""
Metrics Dashboard - FIXED VERSION
Windows-compatible encoding for report generation
"""

# Add this at the beginning of generate_report method to fix Windows encoding:

def generate_report_fixed(self, metrics, save_path=None):
    """
    Generate text report of metrics (Windows-compatible version).
    
    Args:
        metrics: SimulationMetrics object
        save_path: Path to save report
        
    Returns:
        Report text
    """
    import os
    
    # Use ASCII-safe characters for Windows compatibility
    check = "[OK]"
    cross = "[FAIL]"
    
    report = f"""
{'='*70}
SWARM SIMULATION METRICS REPORT
{'='*70}

FORMATION: {metrics.formation_name.upper()}
TIMESTAMP: {metrics.timestamp}

{'-'*70}
CONVERGENCE METRICS
{'-'*70}
Status:                  {check if metrics.converged else cross} {'CONVERGED' if metrics.converged else 'TIMEOUT'}
Convergence Time:        {metrics.convergence_time:.2f}s
Convergence Rate:        {metrics.convergence_rate*100:.1f}% ({int(metrics.convergence_rate * len(metrics.final_errors))}/{len(metrics.final_errors)} agents)
Steps to Convergence:    {metrics.steps_to_convergence}/{metrics.max_steps}

Final Errors (per agent):
  Mean:                  {np.mean(metrics.final_errors):.4f}m
  Median:                {np.median(metrics.final_errors):.4f}m
  Std Dev:               {np.std(metrics.final_errors):.4f}m
  Max:                   {np.max(metrics.final_errors):.4f}m
  Min:                   {np.min(metrics.final_errors):.4f}m

{'-'*70}
ENERGY METRICS
{'-'*70}
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
{'-'*70}
COLLISION METRICS
{'-'*70}
Total Collision Events:  {metrics.collision_count}
Collision Pairs:         {len(set((e[1], e[2]) for e in metrics.collision_events))}
Min Distance Observed:   {metrics.min_distance_observed:.4f}m
Safety Threshold:        0.35m
Status:                  {check if metrics.collision_count == 0 else cross} {'SAFE' if metrics.collision_count == 0 else 'UNSAFE'}

"""
    
    if metrics.collision_events:
        report += "Collision Events:\n"
        for step, agent1, agent2 in metrics.collision_events[:10]:
            report += f"  Step {step}: Agents {agent1} ++ {agent2}\n"
        if len(metrics.collision_events) > 10:
            report += f"  ... and {len(metrics.collision_events) - 10} more events\n"
    else:
        report += "No collision events detected.\n"
    
    report += f"""
{'-'*70}
DISTANCE METRICS
{'-'*70}
Total Distance:          {metrics.total_distance:.2f}m
Avg Distance per Agent:  {metrics.avg_distance_per_agent:.2f}m
Min Distance:            {np.min(list(metrics.distance_profile.values())):.2f}m
Max Distance:            {np.max(list(metrics.distance_profile.values())):.2f}m

{'-'*70}
EFFICIENCY METRICS
{'-'*70}
Energy per Distance:     {metrics.total_energy / max(metrics.total_distance, 0.001):.2f}J/m
Wall Clock Time:         {metrics.simulation_time:.2f}s
Steps per Second:        {metrics.steps_to_convergence / max(metrics.simulation_time, 0.001):.1f}

{'='*70}
"""
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                   exist_ok=True)
        # FIX: Use UTF-8 encoding explicitly for Windows compatibility
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[OK] Report saved to {save_path}")
    
    return report
