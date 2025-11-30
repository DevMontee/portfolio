"""

Implements:
  1. Extended Kalman Filter (EKF) for 3D tracking
  2. Constant velocity motion model
  3. Predict step with process noise tuning
  4. Test on synthetic straight-line motion
  
Usage:
  python kalman_filter.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 3D vehicle tracking.
    
    State vector (6D):
      x = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]^T
    
    Motion model: constant velocity
      pos(t+1) = pos(t) + vel(t) * dt
      vel(t+1) = vel(t)
    """
    
    def __init__(self, dt: float, process_noise: np.ndarray = None, 
                 measurement_noise: np.ndarray = None):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            dt: Time step (seconds)
            process_noise: Process noise covariance matrix Q (6x6)
                          If None, uses default values
            measurement_noise: Measurement noise covariance matrix R (3x3)
                              If None, uses default values
        """
        self.dt = dt
        
        # State: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        self.x = np.zeros((6, 1))  # State vector
        self.P = np.eye(6)  # State covariance matrix
        
        # State transition matrix (constant velocity model)
        self.F = self._build_transition_matrix(dt)
        
        # Measurement matrix (we measure position only)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]])  # 3x6
        
        # Process noise covariance - tunable parameter
        if process_noise is None:
            # Default: assume constant acceleration model
            # Process noise on position and velocity
            q_pos = 0.01  # Position noise
            q_vel = 0.1   # Velocity noise
            self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        else:
            self.Q = process_noise
        
        # Measurement noise covariance
        if measurement_noise is None:
            # Default: 0.1m std deviation on each position measurement
            r_meas = 0.1
            self.R = np.diag([r_meas, r_meas, r_meas])  # 3x3
        else:
            self.R = measurement_noise
    
    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Build state transition matrix for constant velocity model.
        
        F = [I  dt*I]  where I is 3x3 identity
            [0   I  ]
        """
        F = np.eye(6)
        F[0, 3] = dt  # pos_x += vel_x * dt
        F[1, 4] = dt  # pos_y += vel_y * dt
        F[2, 5] = dt  # pos_z += vel_z * dt
        return F
    
    def predict(self) -> np.ndarray:
        """
        Predict step: advance state using motion model.
        
        Returns:
            Predicted state vector (6,1)
        """
        # Predict state: x = F * x
        self.x = self.F @ self.x
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.flatten()
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step: correct state using measurement.
        
        Args:
            z: Measurement vector (3,) - position measurements
            
        Returns:
            Updated state vector (6,)
        """
        # Ensure z is 2D
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        
        # Innovation: y = z - H * x
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K * H) * P
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.x.flatten()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state estimate.
        
        Returns:
            state - State vector [x, y, z, vx, vy, vz] as 1D array (6,)
        """
        return self.x.flatten()
    
    def set_state(self, x: np.ndarray, P: np.ndarray = None) -> None:
        """
        Set state and optionally covariance.
        
        Args:
            x: State vector (6,)
            P: Covariance matrix (6,6), if None uses default
        """
        self.x = x.reshape(-1, 1)
        if P is not None:
            self.P = P
        else:
            self.P = np.eye(6)
    
    def get_position(self) -> np.ndarray:
        """Get position part of state."""
        return self.x[:3].flatten()
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity part of state."""
        return self.x[3:6].flatten()
    
    def get_position_uncertainty(self) -> float:
        """Get standard deviation of position uncertainty."""
        return np.sqrt(np.trace(self.P[:3, :3]))


def test_synthetic_straight_line():
    """
    Test EKF on synthetic straight-line motion.
    
    Object moves in a straight line with constant velocity.
    We simulate measurements with Gaussian noise.
    """
    print("\n" + "="*70)
    print("TEST 1: SYNTHETIC STRAIGHT-LINE MOTION")
    print("="*70)
    
    # Setup
    dt = 0.1  # 100ms time step
    total_time = 10.0  # 10 seconds
    num_steps = int(total_time / dt)
    
    # Ground truth: constant velocity motion
    true_pos_0 = np.array([0.0, 0.0, 0.0])  # Starting position
    true_vel = np.array([1.0, 0.5, 0.2])    # Constant velocity
    
    # Measurement noise
    meas_noise_std = 0.1  # 10cm std dev
    
    # Create EKF
    ekf = ExtendedKalmanFilter(dt=dt)
    
    # Initialize with noisy first measurement
    first_measurement = true_pos_0 + np.random.randn(3) * meas_noise_std
    ekf.set_state(np.array([first_measurement[0], first_measurement[1], 
                           first_measurement[2], 0, 0, 0]))
    
    # Storage for results
    time_steps = []
    true_positions = []
    true_velocities = []
    measurements = []
    filter_positions = []
    filter_velocities = []
    position_uncertainties = []
    
    print(f"\nParameters:")
    print(f"  - Time step (dt): {dt} s")
    print(f"  - Total time: {total_time} s")
    print(f"  - Number of steps: {num_steps}")
    print(f"  - True velocity: {true_vel}")
    print(f"  - Measurement noise std: {meas_noise_std} m")
    print(f"  - Process noise diagonal: {np.diag(ekf.Q)}")
    
    print(f"\nRunning filter for {num_steps} steps...")
    
    for step in range(num_steps):
        t = step * dt
        time_steps.append(t)
        
        # Generate ground truth
        true_pos = true_pos_0 + true_vel * t
        true_positions.append(true_pos)
        true_velocities.append(true_vel)
        
        # Generate measurement with noise
        measurement = true_pos + np.random.randn(3) * meas_noise_std
        measurements.append(measurement)
        
        # EKF: Predict
        predicted_state = ekf.predict()
        
        # EKF: Update
        updated_state = ekf.update(measurement)
        
        filter_pos = ekf.get_position()
        filter_vel = ekf.get_velocity()
        pos_unc = ekf.get_position_uncertainty()
        
        filter_positions.append(filter_pos)
        filter_velocities.append(filter_vel)
        position_uncertainties.append(pos_unc)
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}/{num_steps}: "
                  f"pos error: {np.linalg.norm(filter_pos - true_pos):.4f}m, "
                  f"vel error: {np.linalg.norm(filter_vel - true_vel):.4f}m/s")
    
    # Convert to arrays
    time_steps = np.array(time_steps)
    true_positions = np.array(true_positions)
    measurements = np.array(measurements)
    filter_positions = np.array(filter_positions)
    filter_velocities = np.array(filter_velocities)
    position_uncertainties = np.array(position_uncertainties)
    
    # Compute errors
    position_errors = np.linalg.norm(filter_positions - true_positions, axis=1)
    velocity_errors = np.linalg.norm(filter_velocities - true_vel, axis=1)
    measurement_errors = np.linalg.norm(measurements - true_positions, axis=1)
    
    # Statistics
    print(f"\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"\nPosition Error:")
    print(f"  Mean: {np.mean(position_errors):.4f} m")
    print(f"  Std:  {np.std(position_errors):.4f} m")
    print(f"  Max:  {np.max(position_errors):.4f} m")
    
    print(f"\nVelocity Error:")
    print(f"  Mean: {np.mean(velocity_errors):.4f} m/s")
    print(f"  Std:  {np.std(velocity_errors):.4f} m/s")
    print(f"  Max:  {np.max(velocity_errors):.4f} m/s")
    
    print(f"\nMeasurement Error (raw):")
    print(f"  Mean: {np.mean(measurement_errors):.4f} m")
    print(f"  Std:  {np.std(measurement_errors):.4f} m")
    
    print(f"\nFilter Improvement:")
    improvement = (np.mean(measurement_errors) - np.mean(position_errors)) / np.mean(measurement_errors) * 100
    print(f"  Filter reduces measurement error by {improvement:.1f}%")
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Position X
    ax = axes[0, 0]
    ax.plot(time_steps, true_positions[:, 0], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(time_steps, measurements[:, 0], 'r.', label='Measurements', alpha=0.5, markersize=4)
    ax.plot(time_steps, filter_positions[:, 0], 'g-', label='Filter Estimate', linewidth=2)
    ax.fill_between(time_steps, 
                    filter_positions[:, 0] - 2*position_uncertainties,
                    filter_positions[:, 0] + 2*position_uncertainties,
                    alpha=0.2, color='green', label='±2σ Uncertainty')
    ax.set_ylabel('X Position (m)')
    ax.set_title('Position X')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position Y
    ax = axes[0, 1]
    ax.plot(time_steps, true_positions[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(time_steps, measurements[:, 1], 'r.', label='Measurements', alpha=0.5, markersize=4)
    ax.plot(time_steps, filter_positions[:, 1], 'g-', label='Filter Estimate', linewidth=2)
    ax.fill_between(time_steps,
                    filter_positions[:, 1] - 2*position_uncertainties,
                    filter_positions[:, 1] + 2*position_uncertainties,
                    alpha=0.2, color='green', label='±2σ Uncertainty')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Position Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position Z
    ax = axes[1, 0]
    ax.plot(time_steps, true_positions[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(time_steps, measurements[:, 2], 'r.', label='Measurements', alpha=0.5, markersize=4)
    ax.plot(time_steps, filter_positions[:, 2], 'g-', label='Filter Estimate', linewidth=2)
    ax.fill_between(time_steps,
                    filter_positions[:, 2] - 2*position_uncertainties,
                    filter_positions[:, 2] + 2*position_uncertainties,
                    alpha=0.2, color='green', label='±2σ Uncertainty')
    ax.set_ylabel('Z Position (m)')
    ax.set_title('Position Z')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position Error
    ax = axes[1, 1]
    ax.plot(time_steps, position_errors, 'r-', linewidth=2)
    ax.axhline(np.mean(position_errors), color='r', linestyle='--', 
               label=f'Mean: {np.mean(position_errors):.4f}m')
    ax.set_ylabel('Error (m)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity
    ax = axes[2, 0]
    ax.plot(time_steps, filter_velocities[:, 0], 'b-', label='Vel X', linewidth=2)
    ax.plot(time_steps, filter_velocities[:, 1], 'g-', label='Vel Y', linewidth=2)
    ax.plot(time_steps, filter_velocities[:, 2], 'r-', label='Vel Z', linewidth=2)
    ax.axhline(true_vel[0], color='b', linestyle='--', alpha=0.5)
    ax.axhline(true_vel[1], color='g', linestyle='--', alpha=0.5)
    ax.axhline(true_vel[2], color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Estimated Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Uncertainty
    ax = axes[2, 1]
    ax.plot(time_steps, position_uncertainties, 'purple', linewidth=2)
    ax.set_ylabel('Std Dev (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Position Uncertainty')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_test_synthetic.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: kalman_filter_test_synthetic.png")
    plt.show()
    
    return ekf, {
        'time': time_steps,
        'true_pos': true_positions,
        'measurements': measurements,
        'filter_pos': filter_positions,
        'filter_vel': filter_velocities,
        'pos_errors': position_errors,
        'vel_errors': velocity_errors,
    }


def test_process_noise_tuning():
    """
    Test sensitivity to process noise parameters.
    """
    print("\n" + "="*70)
    print("TEST 2: PROCESS NOISE TUNING")
    print("="*70)
    
    dt = 0.1
    total_time = 5.0
    num_steps = int(total_time / dt)
    
    # Ground truth
    true_pos_0 = np.array([0.0, 0.0, 0.0])
    true_vel = np.array([2.0, 1.0, 0.5])
    meas_noise_std = 0.1
    
    # Test different process noise values
    process_noise_configs = [
        ("Low (0.001, 0.01)", np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])),
        ("Medium (0.01, 0.1)", np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])),
        ("High (0.1, 1.0)", np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])),
    ]
    
    results = {}
    
    for config_name, Q in process_noise_configs:
        print(f"\nTesting: {config_name}")
        
        ekf = ExtendedKalmanFilter(dt=dt, process_noise=Q)
        
        # Initialize
        first_measurement = true_pos_0 + np.random.randn(3) * meas_noise_std
        ekf.set_state(np.array([first_measurement[0], first_measurement[1], 
                               first_measurement[2], 0, 0, 0]))
        
        pos_errors = []
        
        for step in range(num_steps):
            t = step * dt
            true_pos = true_pos_0 + true_vel * t
            measurement = true_pos + np.random.randn(3) * meas_noise_std
            
            ekf.predict()
            ekf.update(measurement)
            
            filter_pos = ekf.get_position()
            error = np.linalg.norm(filter_pos - true_pos)
            pos_errors.append(error)
        
        results[config_name] = np.array(pos_errors)
        print(f"  Mean error: {np.mean(pos_errors):.4f}m")
        print(f"  Max error:  {np.max(pos_errors):.4f}m")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(num_steps) * dt
    
    for config_name, errors in results.items():
        ax.plot(time_steps, errors, linewidth=2, label=config_name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Effect of Process Noise on Filter Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_noise_tuning.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: kalman_filter_noise_tuning.png")
    plt.show()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DAY 4: EXTENDED KALMAN FILTER IMPLEMENTATION")
    print("="*70)
    
    # Test 1: Synthetic straight-line motion
    ekf, data = test_synthetic_straight_line()
    
    # Test 2: Process noise tuning
    test_process_noise_tuning()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
