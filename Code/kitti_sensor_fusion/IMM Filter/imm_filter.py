"""
IMM (Interacting Multiple Model) Filter
Implements multiple motion models with probabilistic model switching
Supports constant velocity and coordinated turn models
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class IMMConfig:
    """IMM Filter Configuration"""
    dt: float = 0.1  # Time step (seconds)
    # Process noise for constant velocity model
    q_pos_cv: float = 0.1
    q_vel_cv: float = 0.01
    # Process noise for turning model
    q_pos_turn: float = 0.2
    q_yaw_rate: float = 0.1
    # Measurement noise
    r_camera: float = 0.5
    r_lidar: float = 0.5
    # Model Markov chain
    p_cv_to_cv: float = 0.95  # Stay in CV model
    p_cv_to_turn: float = 0.05  # Switch from CV to turn
    p_turn_to_turn: float = 0.90  # Stay in turn model
    p_turn_to_cv: float = 0.10  # Switch from turn to CV
    # Initial model probabilities
    mu_cv_init: float = 0.5
    mu_turn_init: float = 0.5


class ExtendedKalmanFilter:
    """Generic Extended Kalman Filter"""
    
    def __init__(self, x: np.ndarray, P: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Args:
            x: Initial state vector
            P: Initial covariance
            Q: Process noise covariance
            R: Measurement noise covariance
        """
        self.x = x.copy()
        self.P = P.copy()
        self.Q = Q
        self.R = R
    
    def predict(self, F: np.ndarray) -> None:
        """Predict step"""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z: np.ndarray, H: np.ndarray) -> None:
        """Update step"""
        y = z - H @ self.x  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
    
    def likelihood(self, z: np.ndarray, H: np.ndarray) -> float:
        """Calculate measurement likelihood"""
        try:
            y = z - H @ self.x
            S = H @ self.P @ H.T + self.R
            
            det = np.linalg.det(S)
            if det <= 0:
                return 1e-6
            
            # Ensure S is invertible
            try:
                S_inv = np.linalg.inv(S)
            except:
                S_inv = np.linalg.pinv(S)
            
            # Mahalanobis distance
            mahal_dist = y.T @ S_inv @ y
            
            # Gaussian likelihood
            exponent = -0.5 * mahal_dist
            normalization = 1.0 / np.sqrt((2 * np.pi) ** len(y) * max(det, 1e-10))
            likelihood = normalization * np.exp(max(exponent, -100))  # Prevent underflow
            
            return max(likelihood, 1e-6)  # Minimum threshold
        except:
            return 1e-6


class IMMFilter:
    """Interacting Multiple Model Filter"""
    
    def __init__(self, config: IMMConfig):
        """
        Initialize IMM filter with CV and turning models
        
        Args:
            config: IMM configuration
        """
        self.config = config
        self.dt = config.dt
        
        # Initialize two models: Constant Velocity and Turning
        self.num_models = 2
        self.cv_model = 0
        self.turn_model = 1
        
        # Initial state: [x, y, z, vx, vy, vz]
        state_dim = 6
        x_init = np.zeros(state_dim)
        P_init = np.eye(state_dim) * 10.0
        
        # Constant Velocity Model
        Q_cv = np.eye(state_dim) * 0.01
        Q_cv[0:3, 0:3] *= config.q_pos_cv
        Q_cv[3:6, 3:6] *= config.q_vel_cv
        
        R = np.eye(3) * 0.5
        R[0:3, 0:3] = np.diag([config.r_camera, config.r_camera, config.r_lidar])
        
        self.filters = [
            ExtendedKalmanFilter(x_init, P_init, Q_cv, R),  # CV
        ]
        
        # Turning Model (with yaw rate): [x, y, z, vx, vy, vz, yaw_rate]
        state_dim_turn = 7
        x_init_turn = np.zeros(state_dim_turn)
        P_init_turn = np.eye(state_dim_turn) * 10.0
        P_init_turn[6, 6] = 5.0  # Lower uncertainty on yaw rate
        
        Q_turn = np.eye(state_dim_turn) * 0.01
        Q_turn[0:3, 0:3] *= config.q_pos_turn
        Q_turn[3:6, 3:6] *= config.q_vel_cv
        Q_turn[6, 6] *= config.q_yaw_rate
        
        R_turn = np.eye(3) * 0.5
        R_turn[0:3, 0:3] = np.diag([config.r_camera, config.r_camera, config.r_lidar])
        
        self.filters.append(ExtendedKalmanFilter(x_init_turn, P_init_turn, Q_turn, R_turn))
        
        # Model probabilities
        self.mu = np.array([config.mu_cv_init, config.mu_turn_init])
        
        # Transition matrix
        self.M = np.array([
            [config.p_cv_to_cv, config.p_turn_to_cv],
            [config.p_cv_to_turn, config.p_turn_to_turn]
        ])
        
        # Likelihoods
        self.likelihoods = np.ones(self.num_models)
    
    def predict(self) -> None:
        """Predict step for all models"""
        
        # Constant Velocity Model
        F_cv = np.eye(6)
        F_cv[0:3, 3:6] = np.eye(3) * self.dt
        self.filters[self.cv_model].predict(F_cv)
        
        # Turning Model (Coordinated Turn)
        x_turn = self.filters[self.turn_model].x
        yaw_rate = x_turn[6]
        
        # Kinematic model for turning
        F_turn = np.eye(7)
        
        if abs(yaw_rate) > 1e-6:
            # Curved trajectory
            v = np.sqrt(x_turn[3]**2 + x_turn[4]**2)
            
            F_turn[0, 3] = self.dt  # x += vx * dt
            F_turn[1, 4] = self.dt  # y += vy * dt
            F_turn[2, 5] = self.dt  # z += vz * dt
            
            # Update velocity due to turning
            sin_omega = np.sin(yaw_rate * self.dt)
            cos_omega = np.cos(yaw_rate * self.dt)
            
            F_turn[3, 3] = cos_omega
            F_turn[3, 4] = -sin_omega
            F_turn[4, 3] = sin_omega
            F_turn[4, 4] = cos_omega
        else:
            # Straight line (constant velocity)
            F_turn[0:3, 3:6] = np.eye(3) * self.dt
        
        self.filters[self.turn_model].predict(F_turn)
    
    def update(self, z: np.ndarray) -> None:
        """
        Update step with measurement
        
        Args:
            z: Measurement [x, y, z]
        """
        H = np.eye(3, 6)  # For CV model
        H_turn = np.eye(3, 7)  # For turn model
        
        # Calculate likelihoods
        likelihood_cv = self.filters[self.cv_model].likelihood(z, H)
        likelihood_turn = self.filters[self.turn_model].likelihood(z, H_turn)
        
        self.likelihoods[self.cv_model] = likelihood_cv
        self.likelihoods[self.turn_model] = likelihood_turn
        
        # Update model probabilities using mode-matched likelihoods
        # Normalize likelihoods for numerical stability
        max_likelihood = max(likelihood_cv, likelihood_turn)
        if max_likelihood > 0:
            self.likelihoods[self.cv_model] = likelihood_cv / max_likelihood
            self.likelihoods[self.turn_model] = likelihood_turn / max_likelihood
        
        # Model probability update
        predicted_mu = self.M.T @ self.mu
        c_bar = np.sum(predicted_mu * self.likelihoods)
        
        if c_bar > 1e-10:
            self.mu = (predicted_mu * self.likelihoods) / c_bar
        else:
            # Fallback: equal probability
            self.mu = np.array([0.5, 0.5])
        
        # Ensure probabilities sum to 1
        self.mu = self.mu / np.sum(self.mu)
        
        # Update each filter
        self.filters[self.cv_model].update(z, H)
        self.filters[self.turn_model].update(z, H_turn)
    
    def estimate(self) -> np.ndarray:
        """
        Get combined state estimate
        
        Returns:
            Estimated state [x, y, z, vx, vy, vz, yaw_rate]
        """
        # Get CV estimate (6D)
        x_cv = self.filters[self.cv_model].x
        
        # Get Turn estimate (7D)
        x_turn = self.filters[self.turn_model].x
        
        # Combine: weighted by model probabilities
        x_combined = np.zeros(7)
        x_combined[0:6] = self.mu[self.cv_model] * x_cv + self.mu[self.turn_model] * x_turn[0:6]
        x_combined[6] = self.mu[self.turn_model] * x_turn[6]
        
        return x_combined
    
    def get_covariance(self) -> np.ndarray:
        """Get combined covariance"""
        # Simplified: return larger covariance
        return self.filters[self.cv_model].P
    
    def get_model_probabilities(self) -> Dict[str, float]:
        """Get current model probabilities"""
        return {
            'constant_velocity': float(self.mu[self.cv_model]),
            'turning': float(self.mu[self.turn_model])
        }


class IMMTracker:
    """Multi-object tracker using IMM filters"""
    
    def __init__(self, config: IMMConfig = None):
        """
        Initialize IMM tracker
        
        Args:
            config: IMM configuration
        """
        self.config = config or IMMConfig()
        self.tracks: Dict[int, IMMFilter] = {}
        self.next_id = 0
        self.gate_threshold = 8.0
        self.max_age = 30
        self.age_threshold = 3
    
    def predict(self) -> None:
        """Predict all tracks"""
        for imm_filter in self.tracks.values():
            imm_filter.predict()
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with detections
        
        Args:
            detections: List of detection dictionaries with x, y, z
        
        Returns:
            List of tracked objects
        """
        self.predict()
        
        # Simple nearest neighbor association
        tracked_objects = []
        used_detections = set()
        
        for track_id, imm_filter in list(self.tracks.items()):
            x = imm_filter.estimate()
            track_pos = x[0:3]
            
            # Find nearest detection
            best_dist = np.inf
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                
                det_pos = np.array([det['x'], det['y'], det['z']])
                dist = np.linalg.norm(track_pos - det_pos)
                
                if dist < self.gate_threshold and dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx
            
            # Update or age out
            if best_det_idx >= 0:
                det = detections[best_det_idx]
                z = np.array([det['x'], det['y'], det['z']])
                imm_filter.update(z)
                used_detections.add(best_det_idx)
                
                tracked_objects.append({
                    'id': track_id,
                    'x': x[0],
                    'y': x[1],
                    'z': x[2],
                    'vx': x[3],
                    'vy': x[4],
                    'vz': x[5],
                    'yaw_rate': x[6],
                    'model_probs': imm_filter.get_model_probabilities(),
                    'confidence': 0.95
                })
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                track_id = self.next_id
                self.next_id += 1
                
                config = IMMConfig(dt=0.1)
                self.tracks[track_id] = IMMFilter(config)
                
                z = np.array([det['x'], det['y'], det['z']])
                self.tracks[track_id].update(z)
        
        return tracked_objects
    
    def get_state(self) -> Dict:
        """Get full tracker state"""
        return {
            'num_tracks': len(self.tracks),
            'tracks': {
                track_id: {
                    'state': imm_filter.estimate(),
                    'model_probs': imm_filter.get_model_probabilities()
                }
                for track_id, imm_filter in self.tracks.items()
            }
        }
