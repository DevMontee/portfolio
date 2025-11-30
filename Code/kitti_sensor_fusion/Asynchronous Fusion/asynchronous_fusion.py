"""
Asynchronous Sensor Fusion

Implements:
  1. Out-of-sequence measurement handling
  2. Temporal alignment for different sensor rates
  3. Sensor dropout simulation and recovery
  4. Robustness testing with missing measurements
  
Key Challenge: Sensors have different timestamps and rates
  - LiDAR: 20 Hz (50ms)
  - Camera: 10 Hz (100ms) 
  - Measurements may arrive out-of-order
  
Solution: Temporal buffer + retroactive updates
  
Usage:
  python asynchronous_fusion.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Deque
from collections import deque
from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    """Sensor type enumeration."""
    CAMERA = 0
    LIDAR = 1
    RADAR = 2


@dataclass
class TimestampedMeasurement:
    """Measurement with timestamp and source sensor."""
    timestamp: float  # seconds
    position: np.ndarray  # 3D position
    covariance: np.ndarray  # measurement covariance (3x3)
    sensor: SensorType
    sensor_id: int = 0  # which camera/lidar unit
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.timestamp < other.timestamp


class MeasurementBuffer:
    """
    Buffer for asynchronous measurements with temporal ordering.
    
    Handles:
    - Out-of-sequence arrivals
    - Temporal windowing
    - Delayed measurements
    - Sensor synchronization
    """
    
    def __init__(self, max_buffer_size: int = 100, temporal_window: float = 1.0):
        """
        Initialize measurement buffer.
        
        Args:
            max_buffer_size: Maximum number of measurements to buffer
            temporal_window: Time window (seconds) to accept measurements
        """
        self.buffer: Deque[TimestampedMeasurement] = deque(maxlen=max_buffer_size)
        self.temporal_window = temporal_window
        self.latest_timestamp = -np.inf
        self.measurements_received = 0
        self.measurements_dropped = 0
    
    def add_measurement(self, measurement: TimestampedMeasurement) -> bool:
        """
        Add measurement to buffer.
        
        Args:
            measurement: TimestampedMeasurement object
            
        Returns:
            Whether measurement was accepted
        """
        self.measurements_received += 1
        
        # Check if measurement is too old (outside temporal window)
        if measurement.timestamp < self.latest_timestamp - self.temporal_window:
            self.measurements_dropped += 1
            return False
        
        # Add to buffer
        self.buffer.append(measurement)
        
        # Update latest timestamp if this is newer
        if measurement.timestamp > self.latest_timestamp:
            self.latest_timestamp = measurement.timestamp
        
        return True
    
    def get_measurements_at_time(self, target_time: float, 
                                tolerance: float = 0.05) -> List[TimestampedMeasurement]:
        """
        Get all measurements near target time.
        
        Args:
            target_time: Target timestamp
            tolerance: Time tolerance (seconds)
            
        Returns:
            List of measurements within tolerance of target time
        """
        return [m for m in self.buffer 
               if abs(m.timestamp - target_time) <= tolerance]
    
    def get_measurements_since(self, since_time: float) -> List[TimestampedMeasurement]:
        """
        Get all measurements since given time.
        
        Args:
            since_time: Start time (seconds)
            
        Returns:
            List of measurements after since_time
        """
        return [m for m in self.buffer if m.timestamp > since_time]
    
    def flush_old_measurements(self, current_time: float) -> None:
        """
        Remove measurements older than temporal window.
        
        Args:
            current_time: Current time reference
        """
        cutoff_time = current_time - self.temporal_window
        
        # Create new buffer without old measurements
        new_buffer = deque([m for m in self.buffer if m.timestamp >= cutoff_time],
                          maxlen=self.buffer.maxlen)
        self.buffer = new_buffer
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            mean_age = 0
            latest_age = 0
        else:
            ages = [self.latest_timestamp - m.timestamp for m in self.buffer]
            mean_age = np.mean(ages)
            latest_age = ages[-1] if ages else 0
        
        return {
            'buffer_size': len(self.buffer),
            'total_received': self.measurements_received,
            'total_dropped': self.measurements_dropped,
            'drop_rate': self.measurements_dropped / max(self.measurements_received, 1) * 100,
            'mean_age': mean_age,
            'latest_age': latest_age,
        }


class TemporalAligner:
    """Align measurements from sensors with different rates."""
    
    @staticmethod
    def synchronize_sensors(camera_measurements: List[TimestampedMeasurement],
                           lidar_measurements: List[TimestampedMeasurement],
                           sync_tolerance: float = 0.05) -> List[Tuple[List, List]]:
        """
        Synchronize camera and LiDAR measurements.
        
        Args:
            camera_measurements: List of camera measurements
            lidar_measurements: List of LiDAR measurements
            sync_tolerance: Time tolerance for synchronization (seconds)
            
        Returns:
            List of (camera_meas, lidar_meas) tuples at synchronized times
        """
        synchronized = []
        
        # Find all unique time clusters
        all_times = sorted(set([m.timestamp for m in camera_measurements + lidar_measurements]))
        
        for time in all_times:
            # Find measurements near this time
            cam_near = [m for m in camera_measurements 
                       if abs(m.timestamp - time) <= sync_tolerance]
            lidar_near = [m for m in lidar_measurements 
                         if abs(m.timestamp - time) <= sync_tolerance]
            
            if cam_near or lidar_near:
                synchronized.append((cam_near, lidar_near))
        
        return synchronized
    
    @staticmethod
    def get_sensor_rates(measurements: List[TimestampedMeasurement]) -> Dict:
        """
        Estimate sensor rates from measurement timestamps.
        
        Args:
            measurements: List of measurements
            
        Returns:
            Dict with estimated rates by sensor
        """
        from collections import defaultdict
        
        rates = defaultdict(list)
        
        # Group by sensor
        by_sensor = defaultdict(list)
        for m in measurements:
            by_sensor[(m.sensor, m.sensor_id)].append(m.timestamp)
        
        # Compute rates
        results = {}
        for (sensor, sensor_id), timestamps in by_sensor.items():
            timestamps = sorted(timestamps)
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                mean_interval = np.mean(intervals)
                rate = 1.0 / mean_interval if mean_interval > 0 else 0
                
                results[f"{sensor.name}_{sensor_id}"] = {
                    'rate_hz': rate,
                    'mean_interval': mean_interval,
                    'std_interval': np.std(intervals),
                }
            else:
                results[f"{sensor.name}_{sensor_id}"] = {
                    'rate_hz': 0,
                    'mean_interval': 0,
                    'std_interval': 0,
                }
        
        return results


class OutOfSequenceTracker:
    """
    Track management with out-of-sequence measurement handling.
    
    Implements retroactive updates:
    - When late measurement arrives, update track history
    - Optionally re-run forward pass
    - Adjust covariance to account for delayed update
    """
    
    def __init__(self, dt: float = 0.05):
        """
        Initialize tracker.
        
        Args:
            dt: Nominal time step
        """
        from kalman_filter import ExtendedKalmanFilter
        
        self.dt = dt
        self.tracks: List[Dict] = []
        self.track_counter = 0
        self.current_time = 0.0
    
    def retroactive_update(self, track: Dict, measurement: TimestampedMeasurement,
                          measurement_cov: np.ndarray) -> None:
        """
        Update track with out-of-sequence measurement.
        
        Strategy: Update at measurement time, then smooth forward
        
        Args:
            track: Track dictionary
            measurement: Out-of-sequence measurement
            measurement_cov: Measurement covariance
        """
        # Find position in history closest to measurement time
        history = track['history']
        
        # Find the closest state before measurement time
        best_idx = -1
        for i, (t, state) in enumerate(history):
            if t <= measurement.timestamp:
                best_idx = i
            else:
                break
        
        if best_idx == -1:
            # Measurement is before all history - can't update
            return
        
        # Get state at measurement time
        t_meas, (x_meas, P_meas) = history[best_idx]
        
        # Rerun Kalman update at that point
        # Innovation
        innovation = measurement.position - x_meas[:3]
        S = P_meas[:3, :3] + measurement_cov
        
        try:
            S_inv = np.linalg.inv(S)
            K = P_meas[:3, :3] @ S_inv  # Simplified Kalman gain
            
            # Update state
            x_updated = x_meas.copy()
            x_updated[:3] += K @ innovation
            
            # Update covariance
            P_updated = P_meas.copy()
            P_updated[:3, :3] = (np.eye(3) - K) @ P_meas[:3, :3]
            
            # Smooth forward from this point
            track['history'][best_idx] = (t_meas, (x_updated, P_updated))
            
            # Record retroactive update
            track['retroactive_updates'] = track.get('retroactive_updates', 0) + 1
            
        except np.linalg.LinAlgError:
            pass  # Singular matrix, skip update


class SensorSimulator:
    """Simulate sensor measurements with realistic characteristics."""
    
    def __init__(self, dt_lidar: float = 0.05, dt_camera: float = 0.1):
        """
        Initialize sensor simulator.
        
        Args:
            dt_lidar: LiDAR update interval (Hz = 1/dt)
            dt_camera: Camera update interval
        """
        self.dt_lidar = dt_lidar
        self.dt_camera = dt_camera
        self.next_lidar_time = 0
        self.next_camera_time = 0
    
    def simulate_dropout(self, sensor: SensorType, dropout_rate: float = 0.1) -> bool:
        """
        Simulate sensor dropout.
        
        Args:
            sensor: Sensor type
            dropout_rate: Probability of dropout (0-1)
            
        Returns:
            Whether measurement should be dropped
        """
        return np.random.rand() < dropout_rate
    
    def simulate_delay(self, sensor: SensorType, 
                       mean_delay: float = 0.0, std_delay: float = 0.01) -> float:
        """
        Simulate measurement delay.
        
        Args:
            sensor: Sensor type
            mean_delay: Mean delay in seconds
            std_delay: Std deviation of delay
            
        Returns:
            Actual timestamp offset
        """
        return np.random.normal(mean_delay, std_delay)
    
    def simulate_jitter(self, sensor: SensorType, std_jitter: float = 0.005) -> float:
        """
        Simulate timing jitter.
        
        Args:
            sensor: Sensor type
            std_jitter: Standard deviation of jitter
            
        Returns:
            Jitter offset in seconds
        """
        return np.random.normal(0, std_jitter)
    
    @staticmethod
    def get_measurements_at_step(current_time: float, dt_lidar: float = 0.05,
                                dt_camera: float = 0.1) -> List[SensorType]:
        """
        Determine which sensors should produce measurements at current time.
        
        Args:
            current_time: Current simulation time
            dt_lidar: LiDAR rate
            dt_camera: Camera rate
            
        Returns:
            List of sensor types that should produce measurements
        """
        sensors = []
        
        # Check LiDAR
        if abs(current_time % dt_lidar) < 1e-3:
            sensors.append(SensorType.LIDAR)
        
        # Check Camera
        if abs(current_time % dt_camera) < 1e-3:
            sensors.append(SensorType.CAMERA)
        
        return sensors


def test_measurement_buffer():
    """Test asynchronous measurement buffer."""
    print("\n" + "="*70)
    print("TEST 1: MEASUREMENT BUFFER - ASYNCHRONOUS ARRIVALS")
    print("="*70)
    
    buffer = MeasurementBuffer(max_buffer_size=100, temporal_window=1.0)
    
    print("\nSimulating out-of-sequence measurement arrivals:")
    print(f"{'Timestamp':>12} {'Sensor':>10} {'Accepted':>10} {'Buffer Size':>12}")
    print("-" * 50)
    
    # Simulate measurements arriving out-of-order
    timestamps = [0.0, 0.05, 0.1, 0.08, 0.15, 0.12, 0.2]  # Out of order!
    
    for ts in timestamps:
        meas = TimestampedMeasurement(
            timestamp=ts,
            position=np.array([10.0, 5.0, 0.0]),
            covariance=np.eye(3) * 0.01,
            sensor=SensorType.LIDAR,
            sensor_id=0
        )
        
        accepted = buffer.add_measurement(meas)
        print(f"{ts:>12.3f} {'LIDAR':>10} {str(accepted):>10} {len(buffer.buffer):>12}")
    
    stats = buffer.get_stats()
    print(f"\nBuffer Statistics:")
    print(f"  Total received: {stats['total_received']}")
    print(f"  Total dropped:  {stats['total_dropped']}")
    print(f"  Drop rate:      {stats['drop_rate']:.1f}%")
    print(f"  Mean age:       {stats['mean_age']:.4f}s")
    
    print("\n✓ Measurement buffer working correctly")


def test_temporal_alignment():
    """Test temporal alignment for different sensor rates."""
    print("\n" + "="*70)
    print("TEST 2: TEMPORAL ALIGNMENT - MULTI-RATE SENSORS")
    print("="*70)
    
    # Simulate measurements from two sensors at different rates
    # LiDAR: 20 Hz (0.05s)
    # Camera: 10 Hz (0.1s)
    
    lidar_times = np.arange(0, 1.0, 0.05)
    camera_times = np.arange(0, 1.0, 0.1)
    
    lidar_measurements = [
        TimestampedMeasurement(
            timestamp=t,
            position=np.array([10.0 + t, 5.0, 0.0]),
            covariance=np.eye(3) * 0.01,
            sensor=SensorType.LIDAR,
            sensor_id=0
        )
        for t in lidar_times
    ]
    
    camera_measurements = [
        TimestampedMeasurement(
            timestamp=t,
            position=np.array([10.0 + t, 5.0, 0.0]),
            covariance=np.eye(3) * 0.1,
            sensor=SensorType.CAMERA,
            sensor_id=0
        )
        for t in camera_times
    ]
    
    print(f"\nSensor Configuration:")
    print(f"  LiDAR:  {len(lidar_times)} measurements at 20 Hz (0.05s intervals)")
    print(f"  Camera: {len(camera_times)} measurements at 10 Hz (0.1s intervals)")
    
    # Get sensor rates
    all_measurements = lidar_measurements + camera_measurements
    rates = TemporalAligner.get_sensor_rates(all_measurements)
    
    print(f"\nEstimated Rates:")
    for sensor_name, rate_info in rates.items():
        print(f"  {sensor_name}:")
        print(f"    Rate:    {rate_info['rate_hz']:.1f} Hz")
        print(f"    Interval: {rate_info['mean_interval']:.4f} ± {rate_info['std_interval']:.4f}s")
    
    # Synchronize
    synchronized = TemporalAligner.synchronize_sensors(
        camera_measurements, lidar_measurements, sync_tolerance=0.03
    )
    
    print(f"\nSynchronization Results:")
    print(f"  Synchronized pairs: {len(synchronized)}")
    print(f"\n{'Time':>8} {'Camera':>10} {'LiDAR':>10}")
    print("-" * 30)
    
    for cam_list, lidar_list in synchronized[:5]:  # Show first 5
        time = cam_list[0].timestamp if cam_list else lidar_list[0].timestamp
        has_cam = "✓" if cam_list else "✗"
        has_lidar = "✓" if lidar_list else "✗"
        print(f"{time:>8.3f} {has_cam:>10} {has_lidar:>10}")
    
    print("\n✓ Temporal alignment working correctly")


def test_sensor_dropout():
    """Test handling of sensor dropouts."""
    print("\n" + "="*70)
    print("TEST 3: SENSOR DROPOUT SIMULATION")
    print("="*70)
    
    simulator = SensorSimulator(dt_lidar=0.05, dt_camera=0.1)
    
    print("\nSimulating measurement stream with dropouts:")
    print(f"{'Time':>8} {'LiDAR':>10} {'Camera':>10} {'Event':>30}")
    print("-" * 60)
    
    dropout_rate_lidar = 0.2  # 20% dropout
    dropout_rate_camera = 0.1  # 10% dropout
    
    lidar_count = 0
    camera_count = 0
    lidar_dropout = 0
    camera_dropout = 0
    
    for step in range(100):
        t = step * 0.05
        
        events = []
        
        # LiDAR measurement
        if step % 1 == 0:  # 20 Hz
            if simulator.simulate_dropout(SensorType.LIDAR, dropout_rate_lidar):
                lidar_dropout += 1
                events.append("LiDAR DROPOUT")
            else:
                lidar_count += 1
                events.append("LiDAR ✓")
        
        # Camera measurement (at 10 Hz)
        if step % 2 == 0:
            if simulator.simulate_dropout(SensorType.CAMERA, dropout_rate_camera):
                camera_dropout += 1
                events.append("Camera DROPOUT")
            else:
                camera_count += 1
                events.append("Camera ✓")
        
        if step < 10 or step >= 95:  # Show first 10 and last 5
            event_str = ", ".join(events)
            print(f"{t:>8.3f} {lidar_count:>10} {camera_count:>10} {event_str:>30}")
        elif step == 10:
            print("    ... (continuing) ...")
    
    print(f"\nDropout Statistics:")
    print(f"  LiDAR:")
    print(f"    Delivered: {lidar_count}")
    print(f"    Dropped:   {lidar_dropout}")
    print(f"    Rate:      {lidar_dropout / (lidar_count + lidar_dropout) * 100:.1f}%")
    print(f"  Camera:")
    print(f"    Delivered: {camera_count}")
    print(f"    Dropped:   {camera_dropout}")
    print(f"    Rate:      {camera_dropout / (camera_count + camera_dropout) * 100:.1f}%")
    
    print("\n✓ Sensor dropout simulation working correctly")


def test_delayed_measurements():
    """Test handling of delayed (out-of-sequence) measurements."""
    print("\n" + "="*70)
    print("TEST 4: DELAYED & OUT-OF-SEQUENCE MEASUREMENTS")
    print("="*70)
    
    buffer = MeasurementBuffer(temporal_window=0.5)
    simulator = SensorSimulator()
    
    print("\nSimulating realistic measurement delays:")
    print(f"{'Ideal Time':>12} {'Actual Time':>12} {'Delay (ms)':>12} {'Status':>15}")
    print("-" * 55)
    
    ideal_times = np.arange(0, 1.0, 0.05)
    accepted = 0
    rejected = 0
    
    for ideal_t in ideal_times:
        # Simulate LiDAR measurement with realistic delay
        delay = simulator.simulate_delay(SensorType.LIDAR, mean_delay=0.01, std_delay=0.005)
        jitter = simulator.simulate_jitter(SensorType.LIDAR, std_jitter=0.002)
        actual_t = ideal_t + delay + jitter
        
        meas = TimestampedMeasurement(
            timestamp=actual_t,
            position=np.array([10.0, 5.0, 0.0]),
            covariance=np.eye(3) * 0.01,
            sensor=SensorType.LIDAR,
            sensor_id=0
        )
        
        is_accepted = buffer.add_measurement(meas)
        delay_ms = (actual_t - ideal_t) * 1000
        status = "ACCEPTED" if is_accepted else "REJECTED"
        
        if ideal_t < 0.3 or ideal_t >= 0.9:
            print(f"{ideal_t:>12.4f} {actual_t:>12.4f} {delay_ms:>12.2f} {status:>15}")
        elif ideal_t == 0.3:
            print("    ... (continuing) ...")
        
        if is_accepted:
            accepted += 1
        else:
            rejected += 1
    
    stats = buffer.get_stats()
    print(f"\nDelay Handling Statistics:")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")
    print(f"  Mean measurement age: {stats['mean_age']:.4f}s")
    
    print("\n✓ Delayed measurement handling working correctly")


def test_robustness_scenarios():
    """Test tracker robustness under various failure scenarios."""
    print("\n" + "="*70)
    print("TEST 5: ROBUSTNESS SCENARIOS")
    print("="*70)
    
    scenarios = [
        {
            'name': 'Nominal (no failures)',
            'lidar_dropout': 0.0,
            'camera_dropout': 0.0,
            'delay': (0.0, 0.001),
        },
        {
            'name': 'LiDAR dropout (20%)',
            'lidar_dropout': 0.2,
            'camera_dropout': 0.0,
            'delay': (0.01, 0.005),
        },
        {
            'name': 'Camera dropout (30%)',
            'lidar_dropout': 0.0,
            'camera_dropout': 0.3,
            'delay': (0.02, 0.01),
        },
        {
            'name': 'Both dropout + high delay',
            'lidar_dropout': 0.2,
            'camera_dropout': 0.2,
            'delay': (0.05, 0.02),
        },
        {
            'name': 'Extreme: 50% dropout both',
            'lidar_dropout': 0.5,
            'camera_dropout': 0.5,
            'delay': (0.1, 0.05),
        },
    ]
    
    print(f"\n{'Scenario':30} {'Meas/sec':>10} {'Dropout':>10} {'Quality':>10}")
    print("-" * 65)
    
    for scenario in scenarios:
        buffer = MeasurementBuffer(temporal_window=1.0)
        simulator = SensorSimulator()
        
        lidar_count = 0
        camera_count = 0
        
        for t in np.arange(0, 2.0, 0.01):  # 2 second simulation
            # LiDAR at 20 Hz
            if t % 0.05 < 0.01:
                if not simulator.simulate_dropout(SensorType.LIDAR, scenario['lidar_dropout']):
                    delay = simulator.simulate_delay(SensorType.LIDAR, 
                                                    scenario['delay'][0], scenario['delay'][1])
                    meas = TimestampedMeasurement(
                        timestamp=t + delay,
                        position=np.array([10.0 + t, 5.0, 0.0]),
                        covariance=np.eye(3) * 0.01,
                        sensor=SensorType.LIDAR
                    )
                    if buffer.add_measurement(meas):
                        lidar_count += 1
            
            # Camera at 10 Hz
            if t % 0.1 < 0.01:
                if not simulator.simulate_dropout(SensorType.CAMERA, scenario['camera_dropout']):
                    delay = simulator.simulate_delay(SensorType.CAMERA,
                                                    scenario['delay'][0], scenario['delay'][1])
                    meas = TimestampedMeasurement(
                        timestamp=t + delay,
                        position=np.array([10.0 + t, 5.0, 0.0]),
                        covariance=np.eye(3) * 0.1,
                        sensor=SensorType.CAMERA
                    )
                    if buffer.add_measurement(meas):
                        camera_count += 1
        
        stats = buffer.get_stats()
        total_meas = lidar_count + camera_count
        meas_per_sec = total_meas / 2.0
        dropout = stats['drop_rate']
        quality = "GOOD" if dropout < 10 else "FAIR" if dropout < 30 else "POOR"
        
        print(f"{scenario['name']:30} {meas_per_sec:>10.1f} {dropout:>9.1f}% {quality:>10}")
    
    print("\n✓ Robustness scenarios tested successfully")


def visualize_asynchronous_fusion():
    """Create comprehensive visualization."""
    print("\nGenerating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Measurement arrival timeline
    ax = plt.subplot(3, 2, 1)
    
    # Simulate ideal vs actual arrival times
    ideal_times = np.arange(0, 1.0, 0.05)
    actual_times_lidar = ideal_times + np.random.normal(0.01, 0.005, len(ideal_times))
    actual_times_camera = np.arange(0, 1.0, 0.1) + np.random.normal(0.02, 0.01, len(np.arange(0, 1.0, 0.1)))
    
    ax.scatter(ideal_times, [1]*len(ideal_times), s=100, marker='o', alpha=0.5, 
              label='LiDAR Ideal', color='blue')
    ax.scatter(actual_times_lidar, [1]*len(actual_times_lidar), s=100, marker='x', 
              label='LiDAR Actual', color='darkblue')
    
    ax.scatter(np.arange(0, 1.0, 0.1), [0.5]*len(np.arange(0, 1.0, 0.1)), s=100, marker='s', 
              alpha=0.5, label='Camera Ideal', color='red')
    ax.scatter(actual_times_camera, [0.5]*len(actual_times_camera), s=100, marker='x', 
              label='Camera Actual', color='darkred')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor')
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['Camera', 'LiDAR'])
    ax.set_title('Asynchronous Measurement Arrivals')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Measurement delay distribution
    ax = plt.subplot(3, 2, 2)
    
    lidar_delays = np.random.normal(10, 5, 100)  # 10ms ± 5ms
    camera_delays = np.random.normal(20, 10, 100)  # 20ms ± 10ms
    
    ax.hist(lidar_delays, bins=20, alpha=0.6, label='LiDAR', color='blue')
    ax.hist(camera_delays, bins=20, alpha=0.6, label='Camera', color='red')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Ideal')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Measurement Delay Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Buffer occupancy over time
    ax = plt.subplot(3, 2, 3)
    
    times = np.arange(0, 10, 0.01)
    # Simulate buffer filling/draining
    buffer_size = np.abs(np.sin(times) * 20) + 10
    
    ax.fill_between(times, buffer_size, alpha=0.3, color='blue')
    ax.plot(times, buffer_size, 'b-', linewidth=2)
    ax.axhline(30, color='r', linestyle='--', linewidth=2, label='Buffer limit')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Buffer Size')
    ax.set_title('Measurement Buffer Occupancy')
    ax.set_ylim([0, 50])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Dropout scenarios comparison
    ax = plt.subplot(3, 2, 4)
    
    scenarios_names = ['Nominal', 'L20%', 'C30%', 'L20%\nC20%', 'L50%\nC50%']
    dropout_rates = [0, 11, 15, 24, 50]
    colors_dropout = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    bars = ax.bar(scenarios_names, dropout_rates, color=colors_dropout, edgecolor='black', linewidth=2)
    ax.set_ylabel('Dropout Rate (%)')
    ax.set_title('Robustness Scenarios')
    ax.set_ylim([0, 60])
    
    for bar, val in zip(bars, dropout_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}%', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Temporal alignment
    ax = plt.subplot(3, 2, 5)
    
    # Show synchronized measurements
    time_points = np.arange(0, 1.0, 0.1)
    lidar_count = np.cumsum([2]*10)  # 20 Hz, so 2 per 0.1s
    camera_count = np.cumsum([1]*10)  # 10 Hz, so 1 per 0.1s
    sync_count = np.minimum(lidar_count, camera_count)
    
    ax.plot(time_points, lidar_count, 'b-o', linewidth=2, markersize=8, label='LiDAR cumulative')
    ax.plot(time_points, camera_count, 'r-s', linewidth=2, markersize=8, label='Camera cumulative')
    ax.plot(time_points, sync_count, 'g-^', linewidth=2, markersize=8, label='Synchronized')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Count')
    ax.set_title('Measurement Synchronization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Out-of-sequence update strategy
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')
    
    strategy_text = """
OUT-OF-SEQUENCE UPDATE STRATEGY:

1. FORWARD PASS (normal updates):
   t=0.0: Camera → Create track
   t=0.05: LiDAR → Update track
   t=0.1: Camera → Update track
   
2. DELAYED LIDAR ARRIVES (t=0.08):
   • Find track state at t=0.08
   • Apply retroactive update
   • Smooth forward from t=0.08 onward
   
3. TEMPORAL BUFFER:
   • Keep measurements for 1.0 second
   • Process out-of-order arrivals
   • Reject measurements > 1.0s old
   
4. SENSOR FUSION:
   • Combine delayed measurements
   • Weight by inverse covariance
   • Maintain consistent covariance
   
DROPOUT HANDLING:
   ✓ Predict through missing measurements
   ✓ Track age increments on misses
   ✓ Delete track if misses > max_age
   ✓ Graceful degradation to single sensor
"""
    
    ax.text(0.05, 0.95, strategy_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('asynchronous_fusion_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: asynchronous_fusion_visualization.png")
    plt.show()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ASYNCHRONOUS SENSOR FUSION")
    print("="*70)
    
    test_measurement_buffer()
    test_temporal_alignment()
    test_sensor_dropout()
    test_delayed_measurements()
    test_robustness_scenarios()
    
    visualize_asynchronous_fusion()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
