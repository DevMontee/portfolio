# Sensor Fusion for Autonomous Driving

Multi-sensor 3D object tracking system using Extended Kalman Filter and Interacting Multiple Models (IMM) for autonomous vehicles.

## Overview

This project implements a robust tracking system that combines camera and LiDAR data for 3D object tracking on the KITTI benchmark dataset.

### Key Features

- **Extended Kalman Filter (EKF)** for state estimation
- **Interacting Multiple Models (IMM)** for handling different motion patterns
- **Advanced occlusion handling** maintaining tracks through 5+ second occlusions
- **Asynchronous sensor fusion** handling different update rates
- **Global data association** for optimal track-detection matching

## Performance Results

- **MOTA**: 78.4% on KITTI benchmark
- **Position Error**: 0.42m average
- **Track Retention**: 92% through 5+ second occlusions
- **Curved Trajectory Error Reduction**: 54% vs baseline

## Project Structure

```
├── Environment Setup & Data Pipeline/
│   └── kitti_dataloader.py              # KITTI dataset loader
├── Detection Pipeline/
│   └── detection_pipeline.py            # Object detection
├── Kalman Filter Implementation/
│   └── kalman_filter.py                 # EKF implementation
├── IMM Filter/
│   ├── imm_filter.py                    # IMM tracker
│   └── imm_with_kitti.py               # KITTI integration
├── Data Association/
│   └── data_association.py              # Track-detection matching
├── Occlusion Handling/
│   ├── occlusion_handler.py             # Occlusion management
│   └── test_occlusion_robustness.py    # Robustness testing
└── Parameter Tuning/
    └── tune_parameters_REAL_FIXED.py    # Hyperparameter optimization
```

## Requirements

```bash
python>=3.8
numpy>=1.20.0
opencv-python>=4.5.0
matplotlib>=3.3.0
scipy>=1.6.0
```

## Installation

```bash
pip install numpy opencv-python matplotlib scipy
```

## Dataset

Uses the **KITTI Vision Benchmark Suite**:
- Download from: http://www.cvlibs.net/datasets/kitti/
- Required: Object detection labels, Velodyne point clouds, camera calibration

## Usage

### Run IMM Tracking with Visualization

```bash
python "IMM Filter/run_imm_kitti_with_visualization.py"
```

### Test Occlusion Robustness

```bash
python "Occlusion Handling/test_occlusion_robustness.py"
```

### Parameter Tuning

```bash
python "Parameter Tuning/tune_parameters_REAL_FIXED.py"
```

## Key Capabilities

- ✅ Camera-LiDAR sensor fusion
- ✅ Multiple motion model handling (CV, CA, CTRV)
- ✅ Occlusion-robust tracking
- ✅ Asynchronous measurement updates
- ✅ Automated parameter tuning

## Results

See the [Sensor Fusion project page](../../sensor-fusion.html) for:
- Real-time tracking demonstrations
- Performance analysis and metrics
- Occlusion handling visualizations
- Comparison with published methods

## Technical Details

Implements state-of-the-art techniques for autonomous driving perception including Kalman filtering, IMM estimation, and robust data association for reliable multi-object tracking.
