"""
Integration Script: Swarm Simulation → Visualization Pipeline
Connects your PyBullet swarm simulation with publication-ready trajectory plots
"""

import numpy as np
import pickle
import json
from pathlib import Path
from trajectory_plots_and_animations import TrajectoryVisualizer


class SimulationResultsProcessor:
    """Extract and structure simulation data for visualization."""
    
    @staticmethod
    def extract_from_simulation(simulation_data, formation_name):
        """
        Convert raw simulation data into visualizer format.
        
        Args:
            simulation_data: Dict with keys 'agent_positions', 'agent_errors', etc.
            formation_name: Name of the formation (helix, mandala, etc.)
        
        Returns:
            Tuple of (agent_positions, agent_errors) dicts ready for visualizer
        """
        # Format: {agent_id: (N_steps, 3) array of [x, y, z]}
        agent_positions = {}
        agent_errors = {}
        
        # If simulation_data has position history for each agent
        if isinstance(simulation_data, dict):
            for agent_id, history in simulation_data.items():
                if isinstance(history, np.ndarray):
                    agent_positions[agent_id] = history  # Assume (N, 3) shape
                elif isinstance(history, dict) and 'positions' in history:
                    agent_positions[agent_id] = np.array(history['positions'])
                    if 'errors' in history:
                        agent_errors[agent_id] = np.array(history['errors'])
        
        return agent_positions, agent_errors
    
    @staticmethod
    def load_pickle_results(pickle_file):
        """Load simulation results from pickle file."""
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_json_results(json_file):
        """Load simulation results from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists to numpy arrays
        for agent_id in data:
            if 'positions' in data[agent_id]:
                data[agent_id]['positions'] = np.array(data[agent_id]['positions'])
            if 'errors' in data[agent_id]:
                data[agent_id]['errors'] = np.array(data[agent_id]['errors'])
        
        return data


def visualize_simulation_results(results_file, formation_name, output_dir='./visualizations'):
    """
    Main pipeline: Load simulation results and generate all visualizations.
    
    Args:
        results_file: Path to pickle or JSON file with simulation results
        formation_name: Name of formation (helix, mandala, dragon, etc.)
        output_dir: Where to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading simulation results from: {results_file}")
    
    # Load results
    if results_file.endswith('.pkl'):
        sim_data = SimulationResultsProcessor.load_pickle_results(results_file)
    elif results_file.endswith('.json'):
        sim_data = SimulationResultsProcessor.load_json_results(results_file)
    else:
        raise ValueError("Results file must be .pkl or .json")
    
    # Extract formatted data
    print("Processing simulation data...")
    agent_positions, agent_errors = SimulationResultsProcessor.extract_from_simulation(
        sim_data, formation_name
    )
    
    print(f"Found {len(agent_positions)} agents with trajectory data")
    
    # Create visualizer instance
    visualizer = TrajectoryVisualizer()
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # 3D Trajectories
    print("  → 3D trajectories...")
    visualizer.plot_3d_trajectories(
        agent_positions=agent_positions,
        formation_name=formation_name,
        save_path=str(output_path / f"{formation_name}_3d_trajectories.png")
    )
    
    # 2D Projections (all planes)
    for plane in ['xy', 'xz', 'yz']:
        print(f"  → 2D {plane} projection...")
        visualizer.plot_2d_trajectories(
            agent_positions=agent_positions,
            formation_name=formation_name,
            plane=plane,
            save_path=str(output_path / f"{formation_name}_{plane}_projection.png")
        )
    
    # Animation (optional - can take a while)
    print("  → Swarm convergence animation (this may take a minute)...")
    visualizer.animate_swarm_convergence(
        agent_positions=agent_positions,
        formation_name=formation_name,
        save_path=str(output_path / f"{formation_name}_convergence.gif"),
        interval=50
    )
    
    # Convergence heatmap (if error data exists)
    if agent_errors:
        print("  → Convergence heatmap...")
        visualizer.plot_convergence_heatmap(
            agent_errors=agent_errors,
            formation_name=formation_name,
            save_path=str(output_path / f"{formation_name}_convergence_heatmap.png")
        )
    
    # Distance evolution
    print("  → Inter-agent distance evolution...")
    visualizer.plot_distance_evolution(
        agent_positions=agent_positions,
        formation_name=formation_name,
        save_path=str(output_path / f"{formation_name}_distance_evolution.png")
    )
    
    print(f"\n✓ All visualizations saved to: {output_path}")
    return agent_positions, agent_errors


# ============================================================================
# HOW TO USE THIS WITH YOUR SWARM SIMULATION
# ============================================================================

if __name__ == "__main__":
    """
    SETUP INSTRUCTIONS:
    
    1. Modify your swarm simulation to save results in this format:
    
       Dictionary structure (pickle or JSON):
       {
           0: {'positions': [[x, y, z], [x, y, z], ...], 'errors': [e1, e2, ...]},
           1: {'positions': [[x, y, z], [x, y, z], ...], 'errors': [e1, e2, ...]},
           ...
       }
       
       OR just numpy arrays:
       {
           0: np.array([[x, y, z], [x, y, z], ...]),
           1: np.array([[x, y, z], [x, y, z], ...]),
           ...
       }
    
    2. After simulation completes, save this dictionary:
    
       import pickle
       results = {agent_id: {'positions': pos_array, 'errors': err_array} for ...}
       with open('helix_results.pkl', 'wb') as f:
           pickle.dump(results, f)
    
    3. Call this script with your results file:
    
       python swarm_simulation_to_visualization.py helix_results.pkl helix
    """
    
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python swarm_simulation_to_visualization.py <results_file> <formation_name>")
        print("\nExample:")
        print("  python swarm_simulation_to_visualization.py helix_results.pkl helix")
        print("\nSupported results formats: .pkl (pickle) or .json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    formation_name = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './visualizations'
    
    try:
        visualize_simulation_results(results_file, formation_name, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
