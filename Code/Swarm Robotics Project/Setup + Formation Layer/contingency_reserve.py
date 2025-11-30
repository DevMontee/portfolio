"""
Contingency Reserve System for Swarm Robotics Formations
Implements safety buffers, energy reserves, and failure recovery
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum


class ContingencyLevel(Enum):
    NOMINAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class ContingencyReserves:
    """Configuration for contingency reserves"""
    energy_reserve_percent: float = 0.15  # 15% energy buffer
    collision_buffer_margin: float = 0.05  # 5cm additional safety margin
    max_position_error_threshold: float = 0.15  # 15cm before contingency triggered
    convergence_timeout: int = 12000  # steps before timeout recovery
    min_safe_distance: float = 0.30  # absolute minimum safety distance


class SwarmContingencyManager:
    """Manages contingency reserves and failure detection for swarm formations"""
    
    def __init__(self, num_agents: int, reserves: ContingencyReserves = None):
        self.num_agents = num_agents
        self.reserves = reserves or ContingencyReserves()
        self.agent_energy = np.zeros(num_agents)
        self.agent_status = np.ones(num_agents, dtype=bool)  # True = operational
        self.contingency_level = ContingencyLevel.NOMINAL
        self.failure_history = []
        self.energy_warning_issued = False
        
    def update_agent_energy(self, energies: np.ndarray, energy_limit: float):
        """Track energy consumption and enforce contingency reserves"""
        self.agent_energy = energies
        
        # Calculate energy reserve threshold
        reserve_threshold = energy_limit * self.reserves.energy_reserve_percent
        
        # Check for low energy agents
        low_energy_agents = np.where(energies > (energy_limit - reserve_threshold))[0]
        
        if len(low_energy_agents) > 0:
            self.contingency_level = ContingencyLevel.WARNING
            if not self.energy_warning_issued:
                self.energy_warning_issued = True
                return {
                    'warning': 'LOW_ENERGY',
                    'affected_agents': low_energy_agents.tolist(),
                    'energy_percent': (energies[low_energy_agents] / energy_limit * 100).tolist(),
                    'action': 'REDUCE_MOVEMENT_SPEED'
                }
        else:
            self.energy_warning_issued = False
            self.contingency_level = ContingencyLevel.NOMINAL
            
        return None
    
    def check_collision_buffer(self, distances: np.ndarray, formation_name: str) -> Dict:
        """
        Verify collision detection has adequate buffer.
        Distances should be minimum distances between agent pairs
        """
        safe_threshold = self.reserves.min_safe_distance
        nominal_threshold = 0.25  # From dashboards
        buffer_threshold = nominal_threshold + self.reserves.collision_buffer_margin
        
        critical_pairs = np.where(distances < safe_threshold)[0]
        warning_pairs = np.where((distances >= safe_threshold) & (distances < buffer_threshold))[0]
        
        status = {
            'formation': formation_name,
            'buffer_margin': self.reserves.collision_buffer_margin,
            'min_distance': distances.min() if len(distances) > 0 else 0,
            'status': ContingencyLevel.NOMINAL
        }
        
        if len(critical_pairs) > 0:
            status['status'] = ContingencyLevel.CRITICAL
            status['critical_violations'] = critical_pairs.tolist()
            status['action'] = 'PAUSE_FORMATION_EMERGENCY_STOP'
            
        elif len(warning_pairs) > 0:
            status['status'] = ContingencyLevel.WARNING
            status['buffer_violations'] = warning_pairs.tolist()
            status['action'] = 'REDUCE_CONVERGENCE_SPEED'
        
        return status
    
    def check_convergence_timeout(self, current_step: int, converged: bool) -> Dict:
        """
        Trigger contingency if formation fails to converge within timeout.
        Based on dashboard data: most formations converge 0.69-100.00s
        """
        status = {
            'step': current_step,
            'converged': converged,
            'timeout_reached': False,
            'action': None
        }
        
        if not converged and current_step > self.reserves.convergence_timeout:
            status['timeout_reached'] = True
            status['action'] = 'RESET_TO_HOME_POSITION'
            status['contingency_level'] = ContingencyLevel.CRITICAL
            self.failure_history.append({
                'type': 'CONVERGENCE_TIMEOUT',
                'step': current_step,
                'action_taken': 'RESET'
            })
        
        return status
    
    def check_position_errors(self, position_errors: np.ndarray, 
                             formation_name: str) -> Dict:
        """
        Monitor final position errors. Trigger contingency if exceeds threshold.
        Dashboard data: errors range from 0.003-0.060m mean
        """
        max_error = position_errors.max()
        mean_error = position_errors.mean()
        
        status = {
            'formation': formation_name,
            'max_error': max_error,
            'mean_error': mean_error,
            'threshold': self.reserves.max_position_error_threshold,
            'acceptable': max_error < self.reserves.max_position_error_threshold
        }
        
        if max_error >= self.reserves.max_position_error_threshold:
            status['contingency_level'] = ContingencyLevel.WARNING
            status['action'] = 'TRIGGER_FINE_POSITION_CORRECTION'
            status['affected_agents'] = np.where(
                position_errors > self.reserves.max_position_error_threshold
            )[0].tolist()
        
        return status
    
    def simulate_agent_failure(self, failed_agent_id: int, current_step: int) -> Dict:
        """Handle graceful degradation when agent fails"""
        self.agent_status[failed_agent_id] = False
        operational_agents = np.sum(self.agent_status)
        
        failure_info = {
            'failed_agent': failed_agent_id,
            'step': current_step,
            'operational_agents': operational_agents,
            'formation_degradation': (1 - operational_agents / self.num_agents) * 100
        }
        
        if operational_agents / self.num_agents < 0.8:
            failure_info['contingency_action'] = 'ABORT_FORMATION'
            failure_info['contingency_level'] = ContingencyLevel.CRITICAL
        elif operational_agents / self.num_agents < 0.95:
            failure_info['contingency_action'] = 'ADJUST_FORMATION_SHAPE'
            failure_info['contingency_level'] = ContingencyLevel.WARNING
        else:
            failure_info['contingency_action'] = 'CONTINUE_WITH_DEGRADED_PERFORMANCE'
            failure_info['contingency_level'] = ContingencyLevel.CAUTION
        
        self.failure_history.append(failure_info)
        return failure_info
    
    def get_energy_reserve_guidance(self, current_energies: np.ndarray, 
                                    energy_limit: float) -> Dict:
        """Recommend speed/acceleration adjustments to maintain reserves"""
        reserve_threshold = energy_limit * self.reserves.energy_reserve_percent
        available_energy = energy_limit - reserve_threshold
        
        guidance = {
            'reserve_threshold_energy': reserve_threshold,
            'available_energy': available_energy,
            'current_energies': current_energies.tolist(),
            'agents_near_limit': np.sum(current_energies > (energy_limit - reserve_threshold)),
            'avg_energy_percent': (current_energies.mean() / energy_limit * 100)
        }
        
        if guidance['avg_energy_percent'] > 80:
            guidance['recommendation'] = 'REDUCE_CONVERGENCE_SPEED_TO_60_PERCENT'
        elif guidance['avg_energy_percent'] > 70:
            guidance['recommendation'] = 'REDUCE_CONVERGENCE_SPEED_TO_75_PERCENT'
        else:
            guidance['recommendation'] = 'MAINTAIN_CURRENT_SPEED'
        
        return guidance
    
    def generate_contingency_report(self) -> Dict:
        """Comprehensive contingency status report"""
        return {
            'current_level': self.contingency_level.name,
            'total_failures': len(self.failure_history),
            'operational_agents': np.sum(self.agent_status),
            'total_agents': self.num_agents,
            'failures': self.failure_history,
            'energy_reserves': {
                'percent': self.reserves.energy_reserve_percent * 100,
                'collision_buffer_cm': self.reserves.collision_buffer_margin * 100
            }
        }


class FormationContingencyValidator:
    """Validates formations against contingency criteria before execution"""
    
    @staticmethod
    def validate_formation_feasibility(formation_config: Dict, 
                                       num_agents: int,
                                       reserves: ContingencyReserves) -> Tuple[bool, List[str]]:
        """Check if formation can be executed with contingency reserves"""
        issues = []
        
        # Check if formation spacing allows for safety buffers
        min_spacing = formation_config.get('min_agent_spacing', 0)
        required_spacing = reserves.min_safe_distance
        
        if min_spacing < required_spacing:
            issues.append(
                f"Formation spacing {min_spacing}m < required {required_spacing}m with buffers"
            )
        
        # Check estimated energy feasibility
        estimated_total_energy = formation_config.get('estimated_energy_consumption', 0)
        energy_limit_per_agent = formation_config.get('energy_limit_per_agent', 0)
        reserve_threshold = energy_limit_per_agent * reserves.energy_reserve_percent
        
        if estimated_total_energy > (energy_limit_per_agent - reserve_threshold) * num_agents:
            issues.append("Estimated energy consumption exceeds reserves")
        
        # Check convergence time
        est_convergence_time = formation_config.get('estimated_convergence_time', 0)
        timeout_steps = reserves.convergence_timeout
        
        if est_convergence_time > timeout_steps:
            issues.append(f"Estimated convergence {est_convergence_time}s > timeout {timeout_steps}s")
        
        is_feasible = len(issues) == 0
        return is_feasible, issues


# Example usage demonstrating contingency implementation
if __name__ == "__main__":
    # Initialize contingency manager for 20 agents (as in dashboards)
    reserves = ContingencyReserves(
        energy_reserve_percent=0.15,
        collision_buffer_margin=0.05,
        max_position_error_threshold=0.15
    )
    
    manager = SwarmContingencyManager(num_agents=20, reserves=reserves)
    
    # Simulate energy monitoring
    simulated_energies = np.array([2.5, 2.8, 3.0, 3.2, 3.5, 2.9, 3.1, 3.3, 3.4, 3.6,
                                   2.7, 3.0, 3.2, 3.4, 3.5, 3.2, 3.0, 2.9, 3.1, 3.3])
    energy_limit = 4.5  # Joules, from dashboard data
    
    energy_warning = manager.update_agent_energy(simulated_energies, energy_limit)
    if energy_warning:
        print("⚠️ Energy Contingency Alert:")
        print(f"   Affected agents: {energy_warning['affected_agents']}")
        print(f"   Energy %: {energy_warning['energy_percent']}")
        print(f"   Action: {energy_warning['action']}\n")
    
    # Simulate collision buffer check for Helix formation
    min_distances = np.array([0.294, 0.301, 0.289, 0.310, 0.305, 0.298])
    collision_status = manager.check_collision_buffer(min_distances, "HELIX")
    print(f"Collision Buffer Status ({collision_status['formation']}):")
    print(f"   Min distance: {collision_status['min_distance']:.3f}m")
    print(f"   Buffer margin: {collision_status['buffer_margin']:.3f}m")
    print(f"   Status: {collision_status['status'].name}\n")
    
    # Simulate position error check
    position_errors = np.random.normal(0.008, 0.005, 20)
    position_errors[position_errors < 0] = 0
    error_status = manager.check_position_errors(position_errors, "MANDALA")
    print(f"Position Error Status ({error_status['formation']}):")
    print(f"   Max error: {error_status['max_error']:.4f}m")
    print(f"   Mean error: {error_status['mean_error']:.4f}m")
    print(f"   Acceptable: {error_status['acceptable']}\n")
    
    # Simulate convergence timeout
    timeout_status = manager.check_convergence_timeout(current_step=500, converged=True)
    print(f"Convergence Timeout Status:")
    print(f"   Converged: {timeout_status['converged']}")
    print(f"   Timeout reached: {timeout_status['timeout_reached']}\n")
    
    # Get energy reserve guidance
    guidance = manager.get_energy_reserve_guidance(simulated_energies, energy_limit)
    print(f"Energy Reserve Guidance:")
    print(f"   Avg energy: {guidance['avg_energy_percent']:.1f}%")
    print(f"   Recommendation: {guidance['recommendation']}\n")
    
    # Generate comprehensive report
    report = manager.generate_contingency_report()
    print(f"Contingency Status Report:")
    print(f"   Current level: {report['current_level']}")
    print(f"   Operational agents: {report['operational_agents']}/{report['total_agents']}")
