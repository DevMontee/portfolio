# Swarm Drones Simulation

Comprehensive platform for coordinating 20 quadrotor agents in complex 3D formations with energy optimization, collision avoidance, and real-time safety monitoring.

## Overview

Multi-agent swarm robotics system featuring realistic physics simulation, decentralized control, and advanced formation capabilities.

### Key Features

- **20 quadrotor agents** with PyBullet physics simulation
- **Complex 3D formations**: helix, mandala, dragon, Canadian flag, cupid
- **Energy optimization** using SciPy
- **Collision avoidance** with proximity-based repulsive potential fields
- **Contingency reserves** for safety-critical operations
- **Real-time metrics dashboard**

## Performance Results

- **5 formation types** with 100% success rate
- **Average convergence**: 300-400 iterations
- **Zero collisions** across all simulations
- **30% energy reduction** vs naive approach

## Project Structure

```
├── Setup + Formation Layer/
│   ├── swarm_agent.py                      # Individual drone agent
│   ├── formation_generator.py              # Formation pattern generators
│   ├── contingency_reserve.py              # Safety reserves
│   └── main_WITH_VISUALIZATION_EXPORT.py  # Full simulation
├── Swarm Collision Avoidance/
│   └── collision_avoidance.py              # Repulsive potential fields
├── Energy Optimization + Metrics/
│   ├── energy_optimization_main.py         # Optimization loop
│   ├── velocity_optimizer.py               # Trajectory optimization
│   └── metrics_dashboard.py                # Real-time metrics
└── Visualization & Polish/
    └── trajectory_plots_and_animations.py  # 3D visualization
```

## Requirements

```bash
python>=3.8
pybullet>=3.2.0
numpy>=1.20.0
scipy>=1.6.0
matplotlib>=3.3.0
```

## Installation

```bash
pip install pybullet numpy scipy matplotlib
```

## Usage

### Basic Simulation

```bash
cd "Setup + Formation Layer"
python main.py
```

### Full Simulation with Visualization Export

```bash
cd "Setup + Formation Layer"
python main_WITH_VISUALIZATION_EXPORT.py
```

### Energy Optimization Demo

```bash
cd "Energy Optimization + Metrics"
python energy_optimization_main.py
```

## Supported Formations

1. **Helix** - Spiral formation with vertical progression
2. **Mandala** - Concentric circular pattern with radial symmetry
3. **Dragon** - Complex 3D artistic formation
4. **Canadian Flag** - Maple leaf pattern with height layers
5. **Cupid** - Heart and arrow pattern

## Technical Capabilities

- ✅ PyBullet physics engine with realistic quadrotor dynamics
- ✅ Decentralized formation control
- ✅ Proximity-based collision avoidance with exponential decay
- ✅ SciPy trajectory optimization for energy efficiency
- ✅ Real-time safety monitoring with contingency reserves
- ✅ Publication-ready 3D visualizations and animations

## Performance Metrics

| Formation | Agents | Convergence Steps | Energy (J) | Collisions |
|-----------|--------|------------------|------------|------------|
| Helix | 20 | 433 | 142.3 | 0 |
| Mandala | 20 | 376 | 128.7 | 0 |
| Dragon | 20 | 512 | 189.4 | 0 |
| Flag | 20 | 447 | 156.2 | 0 |
| Cupid | 20 | 398 | 134.8 | 0 |

## Results

See the [Swarm Drones project page](../../swarm-drones.html) for:
- 3D trajectory evolution animations
- Formation convergence visualizations
- Energy analysis and optimization results
- Performance dashboards and metrics

## Architecture

```
Agent Layer: Individual quadrotor control with PD controllers
    ↓
Formation Layer: Target position generation for complex patterns
    ↓
Collision Avoidance: Repulsive forces with proximity-based decay
    ↓
Energy Optimization: SciPy-based trajectory smoothing
    ↓
Safety Monitoring: Contingency reserve validation
    ↓
Visualization: Real-time metrics and animation export
```

## Future Work

- RRT* motion planning integration
- Dynamic obstacle avoidance
- Formation transitions and morphing
- Hardware deployment on Crazyflie platforms
