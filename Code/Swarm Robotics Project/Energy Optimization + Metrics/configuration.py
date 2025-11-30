"""
Configuration Module
Centralized configuration for Energy Optimization + Metrics Platform
"""

import os
from dataclasses import dataclass
from typing import Dict, List


# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Reference directories (update these to match your system)
BASE_PROJECT_DIR = r"D:\Swarm Robotics Project"
SETUP_DIR = os.path.join(BASE_PROJECT_DIR, "Setup + Formation Layer")
COLLISION_DIR = os.path.join(BASE_PROJECT_DIR, "Swarm Collision Avoidance")
ENERGY_OPT_DIR = os.path.join(BASE_PROJECT_DIR, "Energy Optimization + Metrics")

# Output directories
OUTPUTS_DIR = os.path.join(ENERGY_OPT_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, "visualizations")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")

# Create directories if they don't exist
for dir_path in [OUTPUTS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR, DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

@dataclass
class SimulationConfig:
    """Simulation environment configuration."""
    
    # Physics parameters
    time_step: float = 0.01  # seconds
    gravity: float = 9.81  # m/s^2
    
    # Agent parameters
    agent_mass: float = 0.5  # kg
    agent_radius: float = 0.15  # m
    max_velocity: float = 5.0  # m/s
    max_acceleration: float = 10.0  # m/s^2
    
    # Control parameters
    kp_position: float = 10.0  # proportional gain
    kd_velocity: float = 5.0  # derivative gain
    
    # Convergence parameters
    position_tolerance: float = 0.15  # m
    convergence_check_interval: int = 100  # steps
    max_simulation_steps: int = 10000
    
    # Collision parameters
    collision_threshold: float = 0.35  # m
    min_safe_distance: float = 0.40  # m
    enable_collision_avoidance: bool = True
    
    # Visualization
    enable_gui: bool = False
    camera_distance: float = 15  # meters
    camera_yaw: float = 45  # degrees
    camera_pitch: float = -30  # degrees


# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

@dataclass
class OptimizationConfig:
    """Velocity profile optimization configuration."""
    
    # Velocity constraints
    max_velocity: float = 5.0  # m/s
    max_acceleration: float = 10.0  # m/s^2
    
    # Trajectory parameters
    mass: float = 0.5  # kg
    time_horizon: float = 50.0  # seconds
    time_steps: int = 100  # discretization points
    
    # Convergence tolerance
    tolerance: float = 0.15  # position tolerance (m)
    min_safety_distance: float = 0.35  # collision avoidance (m)
    
    # Optimization objective weights
    # These control the relative importance of different objectives
    # Higher weight = stronger emphasis on that objective
    energy_weight: float = 1.0  # minimize energy consumption
    convergence_weight: float = 10.0  # ensure agents reach targets
    collision_weight: float = 250.0  # avoid collisions
    smoothness_weight: float = 0.1  # smooth trajectories (optional)
    
    # Optimization solver settings
    max_iterations: int = 1000
    tolerance_opt: float = 1e-4
    method: str = 'SLSQP'  # Sequential Least Squares Programming
    
    # Strategy comparison
    compare_baseline: bool = True  # Compare with baseline strategies
    baseline_methods: List[str] = None  # 'linear', 'conservative', 'optimized'
    
    def __post_init__(self):
        if self.baseline_methods is None:
            self.baseline_methods = ['linear', 'conservative', 'optimized']


# ============================================================================
# METRICS COLLECTION PARAMETERS
# ============================================================================

@dataclass
class MetricsConfig:
    """Metrics collection and reporting configuration."""
    
    # Collection options
    collect_position_history: bool = True
    collect_velocity_history: bool = True
    collect_energy_profile: bool = True
    track_collisions: bool = True
    
    # Reporting options
    generate_dashboard: bool = True
    generate_text_report: bool = True
    export_json: bool = True
    create_comparison_plots: bool = True
    
    # Report paths
    dashboard_dpi: int = 300
    report_format: str = 'txt'  # 'txt' or 'md'
    
    # Visualization options
    plot_convergence: bool = True
    plot_energy_profile: bool = True
    plot_collision_timeline: bool = True
    plot_efficiency: bool = True
    
    # Histogram bins
    error_histogram_bins: int = 10
    energy_histogram_bins: int = 15


# ============================================================================
# FORMATION PARAMETERS
# ============================================================================

@dataclass
class FormationConfig:
    """Formation-specific parameters."""
    
    formations: List[str] = None  # Formations to test
    num_agents: int = 20  # Default number of agents
    formation_scale: float = 1.0  # Scale factor
    
    # Formation-specific heights
    helix_height_range: tuple = (0.0, 10.0)
    mandala_height: float = 5.0
    cupid_height: float = 5.0
    dragon_height_range: tuple = (2.0, 8.0)
    flag_height: float = 5.0
    
    def __post_init__(self):
        if self.formations is None:
            self.formations = ['helix', 'mandala', 'cupid', 'dragon', 'flag']


# ============================================================================
# ENERGY ANALYSIS PARAMETERS
# ============================================================================

@dataclass
class EnergyAnalysisConfig:
    """Energy analysis configuration."""
    
    # Analysis options
    analyze_kinetic_energy: bool = True
    analyze_gravitational_work: bool = True
    compute_efficiency: bool = True
    identify_bottlenecks: bool = True
    
    # Bottleneck detection
    bottleneck_percentile: float = 90.0  # Identify top 10% high-energy timesteps
    
    # Comparison
    compare_velocity_profiles: bool = True
    profile_types: List[str] = None  # 'linear', 'trapezoidal', 'sinusoidal'
    
    def __post_init__(self):
        if self.profile_types is None:
            self.profile_types = ['linear', 'trapezoidal', 'sinusoidal']


# ============================================================================
# BENCHMARK PARAMETERS
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark testing configuration."""
    
    # Benchmark options
    num_runs: int = 3  # Number of repeated runs per test
    test_all_formations: bool = True
    test_scalability: bool = False  # Test with different agent counts
    agent_counts: List[int] = None  # Agent counts for scalability testing
    
    # Statistics
    compute_mean: bool = True
    compute_std: bool = True
    compute_min_max: bool = True
    
    def __post_init__(self):
        if self.agent_counts is None:
            self.agent_counts = [10, 20, 30, 40]


# ============================================================================
# ADVANCED PARAMETERS
# ============================================================================

class AdvancedConfig:
    """Advanced configuration options."""
    
    # Numerical parameters
    EPSILON = 1e-10  # Small value for numerical stability
    MIN_VELOCITY = 1e-4  # Minimum velocity threshold
    
    # Energy model parameters
    AERODYNAMIC_DRAG_COEFFICIENT = 0.1
    ROLLING_RESISTANCE = 0.05
    
    # Optimization tolerances
    POSITION_EPSILON = 1e-3  # Position tolerance for convergence checks
    VELOCITY_EPSILON = 1e-4  # Velocity threshold
    
    # Visualization
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    COLORMAP = 'viridis'
    
    # Performance
    PARALLEL_OPTIMIZATION = False  # Use parallel processing
    NUM_WORKERS = 4  # Number of parallel workers


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class Presets:
    """Pre-configured setting combinations."""
    
    @staticmethod
    def get_default() -> Dict:
        """Get default configuration."""
        return {
            'simulation': SimulationConfig(),
            'optimization': OptimizationConfig(),
            'metrics': MetricsConfig(),
            'formation': FormationConfig(),
            'energy_analysis': EnergyAnalysisConfig(),
            'benchmark': BenchmarkConfig()
        }
    
    @staticmethod
    def get_fast() -> Dict:
        """Get fast (less accurate but quick) configuration."""
        config = Presets.get_default()
        config['optimization'].time_steps = 50
        config['simulation'].max_simulation_steps = 5000
        config['optimization'].max_iterations = 500
        return config
    
    @staticmethod
    def get_accurate() -> Dict:
        """Get accurate (slower but more precise) configuration."""
        config = Presets.get_default()
        config['optimization'].time_steps = 200
        config['simulation'].max_simulation_steps = 15000
        config['optimization'].max_iterations = 2000
        config['optimization'].convergence_weight = 20.0
        return config
    
    @staticmethod
    def get_energy_focused() -> Dict:
        """Configuration focusing on energy minimization."""
        config = Presets.get_default()
        config['optimization'].energy_weight = 10.0  # High emphasis on energy
        config['optimization'].convergence_weight = 5.0  # Lower convergence requirement
        config['simulation'].max_simulation_steps = 20000  # More time allowed
        return config
    
    @staticmethod
    def get_safe() -> Dict:
        """Configuration focused on collision avoidance."""
        config = Presets.get_default()
        config['optimization'].collision_weight = 500.0  # Very high collision weight
        config['simulation'].collision_threshold = 0.50  # More conservative
        config['simulation'].enable_collision_avoidance = True
        return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config_from_file(filepath: str) -> Dict:
    """Load configuration from JSON file."""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


def save_config_to_file(config: Dict, filepath: str):
    """Save configuration to JSON file."""
    import json
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        # Convert dataclass objects to dict
        config_dict = {}
        for key, value in config.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        json.dump(config_dict, f, indent=2)


def get_config_summary(config: Dict) -> str:
    """Get string summary of configuration."""
    summary = "CONFIGURATION SUMMARY\n"
    summary += "="*60 + "\n\n"
    
    for section_name, section_config in config.items():
        summary += f"{section_name.upper()}\n"
        summary += "─"*60 + "\n"
        
        if hasattr(section_config, '__dict__'):
            for key, value in section_config.__dict__.items():
                summary += f"  {key:<30} = {value}\n"
        else:
            summary += f"  {section_config}\n"
        summary += "\n"
    
    return summary


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# Global configuration instance
DEFAULT_CONFIG = Presets.get_default()


# Example usage
if __name__ == "__main__":
    print("CONFIGURATION MODULE")
    print("="*60)
    
    # Display default configuration
    config = Presets.get_default()
    print(get_config_summary(config))
    
    # Display preset options
    print("\nAVAILABLE PRESETS:")
    print("  - get_default()      : Standard balanced configuration")
    print("  - get_fast()         : Quick (less accurate)")
    print("  - get_accurate()     : Precise (slower)")
    print("  - get_energy_focused(): Minimize energy")
    print("  - get_safe()         : Collision avoidance focused")
    
    # Save example configuration
    config_path = os.path.join(ENERGY_OPT_DIR, "config_default.json")
    save_config_to_file(config, config_path)
    print(f"\n✓ Configuration saved to {config_path}")
