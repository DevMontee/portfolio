"""
Integration Guide: Contingency System with PyBullet Swarm Formations
Practical implementation for your formation control pipeline
"""

import numpy as np
from contingency_reserve import SwarmContingencyManager, ContingencyReserves


class SimulationWithContingency:
    """Wraps your PyBullet simulation with contingency monitoring"""
    
    def __init__(self, formation_controller, num_agents=20):
        self.formation_controller = formation_controller
        self.num_agents = num_agents
        
        # Initialize contingency system
        self.reserves = ContingencyReserves(
            energy_reserve_percent=0.15,
            collision_buffer_margin=0.05,
            max_position_error_threshold=0.15,
            convergence_timeout=12000
        )
        self.contingency = SwarmContingencyManager(num_agents, self.reserves)
        
        # Tracking
        self.contingency_events = []
        self.simulation_safe = True
        
    def run_formation_with_contingency(self, formation_name, steps=10000, 
                                       energy_limit=4.5, formation_target='cupid'):
        """
        Execute formation control with real-time contingency monitoring
        Mirrors your dashboard metrics collection
        """
        metrics = {
            'step': [],
            'agent_positions': [],
            'agent_energies': [],
            'min_distances': [],
            'position_errors': [],
            'contingency_events': []
        }
        
        for step in range(steps):
            if not self.simulation_safe:
                print(f"🛑 Simulation halted at step {step} due to critical contingency")
                break
            
            # Get current state from PyBullet
            positions = self.formation_controller.get_agent_positions()  # (N, 3)
            energies = self.formation_controller.get_agent_energies()    # (N,)
            
            # ==== CONTINGENCY CHECK 1: Energy Reserves ====
            energy_warning = self.contingency.update_agent_energy(energies, energy_limit)
            if energy_warning and step % 500 == 0:  # Log every 500 steps
                print(f"⚠️  Step {step}: {energy_warning['warning']}")
                metrics['contingency_events'].append({
                    'step': step,
                    'type': energy_warning['warning'],
                    'action': energy_warning['action']
                })
                
                # Apply contingency action
                if energy_warning['warning'] == 'LOW_ENERGY':
                    self.formation_controller.reduce_convergence_speed(0.75)
            
            # ==== CONTINGENCY CHECK 2: Collision Buffers ====
            min_dist = self._compute_minimum_distance(positions)
            collision_status = self.contingency.check_collision_buffer(
                np.array([min_dist]), formation_name
            )
            
            if collision_status['status'].value >= 2:  # WARNING or worse
                print(f"⚠️  Step {step}: Collision buffer violation detected")
                metrics['contingency_events'].append({
                    'step': step,
                    'type': 'COLLISION_BUFFER_VIOLATION',
                    'min_distance': min_dist,
                    'action': collision_status['action']
                })
                
                if collision_status['status'].value == 3:  # CRITICAL
                    print(f"🛑 CRITICAL collision risk - emergency stop")
                    self.simulation_safe = False
                    self.formation_controller.emergency_stop()
            
            # ==== CONTINGENCY CHECK 3: Convergence Timeout ====
            converged = self.formation_controller.has_converged()
            timeout_status = self.contingency.check_convergence_timeout(step, converged)
            
            if timeout_status['timeout_reached']:
                print(f"⏱️  Step {step}: Convergence timeout - resetting formation")
                metrics['contingency_events'].append({
                    'step': step,
                    'type': 'CONVERGENCE_TIMEOUT',
                    'action': timeout_status['action']
                })
                self.formation_controller.reset_to_home()
            
            # ==== CONTINGENCY CHECK 4: Position Errors (every 100 steps) ====
            if step % 100 == 0 and step > 0:
                target_positions = self.formation_controller.get_target_positions()
                pos_errors = np.linalg.norm(positions - target_positions, axis=1)
                
                error_status = self.contingency.check_position_errors(pos_errors, formation_name)
                
                if not error_status['acceptable']:
                    print(f"⚠️  Step {step}: Position errors exceed threshold")
                    metrics['contingency_events'].append({
                        'step': step,
                        'type': 'HIGH_POSITION_ERROR',
                        'max_error': error_status['max_error'],
                        'action': error_status['action']
                    })
                    
                    # Fine position correction
                    if 'affected_agents' in error_status:
                        self.formation_controller.fine_position_correction(
                            error_status['affected_agents']
                        )
            
            # Record metrics for dashboard
            metrics['step'].append(step)
            metrics['agent_energies'].append(energies.copy())
            metrics['min_distances'].append(min_dist)
            
            if step % 100 == 0:
                print(f"Step {step}/{steps} - Energy: {energies.mean():.2f}J, "
                      f"Min dist: {min_dist:.3f}m, Converged: {converged}")
        
        return metrics, self.contingency.generate_contingency_report()
    
    def _compute_minimum_distance(self, positions):
        """Compute minimum pairwise distance between agents"""
        if len(positions) < 2:
            return np.inf
        
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        return np.min(distances) if distances else np.inf


class ContingencyTestSuite:
    """Systematic testing of contingency conditions"""
    
    def __init__(self, formation_controller):
        self.controller = formation_controller
        self.test_results = {}
    
    def test_energy_buffer_enforcement(self):
        """Verify energy reserves are maintained"""
        reserves = ContingencyReserves(energy_reserve_percent=0.15)
        manager = SwarmContingencyManager(20, reserves)
        
        # Test case: agent at 85% capacity
        test_energies = np.full(20, 3.825)  # 85% of 4.5J limit
        test_energies[0] = 4.275  # 95% for agent 0
        
        warning = manager.update_agent_energy(test_energies, energy_limit=4.5)
        
        test_passed = (
            warning is not None and
            0 in warning['affected_agents'] and
            warning['action'] == 'REDUCE_MOVEMENT_SPEED'
        )
        
        self.test_results['energy_buffer'] = {
            'passed': test_passed,
            'details': warning
        }
        return test_passed
    
    def test_collision_buffer_detection(self):
        """Verify collision buffers trigger appropriately"""
        reserves = ContingencyReserves(collision_buffer_margin=0.05)
        manager = SwarmContingencyManager(20, reserves)
        
        # Test distances: some in buffer zone
        min_distances = np.array([0.294, 0.275, 0.252, 0.310])  # Last two in warning zone
        
        status = manager.check_collision_buffer(min_distances, "TEST")
        
        test_passed = (
            status['status'].value >= 1 and  # At least WARNING
            status['action'] == 'REDUCE_CONVERGENCE_SPEED'
        )
        
        self.test_results['collision_buffer'] = {
            'passed': test_passed,
            'details': status
        }
        return test_passed
    
    def test_convergence_timeout_trigger(self):
        """Verify timeout triggers at configured step limit"""
        reserves = ContingencyReserves(convergence_timeout=1000)
        manager = SwarmContingencyManager(20, reserves)
        
        # Before timeout
        status_early = manager.check_convergence_timeout(999, converged=False)
        
        # After timeout
        status_late = manager.check_convergence_timeout(1001, converged=False)
        
        test_passed = (
            not status_early['timeout_reached'] and
            status_late['timeout_reached'] and
            status_late['action'] == 'RESET_TO_HOME_POSITION'
        )
        
        self.test_results['convergence_timeout'] = {
            'passed': test_passed,
            'details': {
                'early': status_early,
                'late': status_late
            }
        }
        return test_passed
    
    def test_graceful_agent_failure(self):
        """Verify system handles agent failures gracefully"""
        reserves = ContingencyReserves()
        manager = SwarmContingencyManager(20, reserves)
        
        # Simulate single agent failure
        failure_info = manager.simulate_agent_failure(5, step=500)
        
        test_passed = (
            failure_info['operational_agents'] == 19 and
            failure_info['contingency_action'] == 'CONTINUE_WITH_DEGRADED_PERFORMANCE' and
            failure_info['contingency_level'].name == 'CAUTION'
        )
        
        # Simulate multiple failures (>5%)
        for i in range(1, 3):
            manager.simulate_agent_failure(5 + i, step=500 + i)
        
        failure_info_multi = manager.failure_history[-1]
        
        test_passed_multi = (
            failure_info_multi['operational_agents'] == 17 and
            failure_info_multi['contingency_level'].name == 'WARNING'
        )
        
        self.test_results['agent_failure'] = {
            'passed': test_passed and test_passed_multi,
            'single_failure': failure_info,
            'multi_failure': failure_info_multi
        }
        return test_passed and test_passed_multi
    
    def run_all_tests(self):
        """Execute complete test suite"""
        print("=" * 60)
        print("CONTINGENCY RESERVE SYSTEM - TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("Energy Buffer Enforcement", self.test_energy_buffer_enforcement),
            ("Collision Buffer Detection", self.test_collision_buffer_detection),
            ("Convergence Timeout Trigger", self.test_convergence_timeout_trigger),
            ("Graceful Agent Failure", self.test_graceful_agent_failure),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                passed = test_func()
                status = "✅ PASSED" if passed else "❌ FAILED"
                results.append((test_name, passed))
                print(f"{status}: {test_name}")
            except Exception as e:
                print(f"❌ ERROR: {test_name} - {str(e)}")
                results.append((test_name, False))
        
        print("=" * 60)
        passed_count = sum(1 for _, passed in results if passed)
        print(f"RESULTS: {passed_count}/{len(results)} tests passed")
        print("=" * 60)
        
        return results
    
    def generate_test_report(self):
        """Generate detailed test report"""
        return self.test_results


# Example: Integration with your existing formation code
if __name__ == "__main__":
    # This would integrate with your PyBullet controller
    # Example pseudo-code structure:
    
    """
    # In your main simulation loop:
    from contingency_reserve import ContingencyReserves, SwarmContingencyManager
    
    # Initialize once
    reserves = ContingencyReserves(
        energy_reserve_percent=0.15,
        collision_buffer_margin=0.05,
        max_position_error_threshold=0.15,
        convergence_timeout=12000
    )
    contingency = SwarmContingencyManager(20, reserves)
    
    # During each simulation step:
    for step in range(10000):
        # Get state
        energies = formation_controller.get_agent_energies()
        
        # Check contingency
        energy_warning = contingency.update_agent_energy(energies, 4.5)
        if energy_warning:
            formation_controller.reduce_convergence_speed(0.75)
        
        # Continue with normal control...
    """
    
    # Run test suite
    suite = ContingencyTestSuite(None)
    suite.run_all_tests()
    
    print("\n📊 Test Details:")
    for test_name, result in suite.test_results.items():
        print(f"\n{test_name}:")
        print(f"  Passed: {result['passed']}")
