"""
End-to-End Parameter Tuning Workflow Example
Complete orchestration for KITTI-based multi-object tracking system
"""

import logging
import json
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./tuning_workflow.log')
    ]
)
logger = logging.getLogger(__name__)


class ParameterTuningWorkflow:
    """
    Complete workflow for parameter tuning
    
    Steps:
    1. Prepare data (KITTI sequences)
    2. Establish baselines (single sensor)
    3. Run grid search (coarse tuning)
    4. Run Bayesian optimization (fine tuning)
    5. Evaluate on test set
    6. Generate report
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize workflow
        
        Args:
            config_path: Path to config JSON with tuning parameters
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'config': self.config,
            'baselines': {},
            'grid_search': {},
            'bayesian_optimization': {},
            'test_results': {},
            'best_parameters': None,
        }
    
    @staticmethod
    def _load_config(config_path: str = None) -> Dict:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'output_dir': './tuning_results',
            'kitti_data_dir': './KITTI/tracking/data_tracking_image_2',
            'tuning_sequences': ['0001', '0002', '0003', '0004', '0005',
                               '0006', '0007', '0008', '0009', '0010'],
            'test_sequences': ['0011', '0012', '0013', '0014', '0015'],
            'grid_search': {
                'q_pos': [0.1, 0.3, 0.5, 1.0],
                'q_vel': [0.01, 0.05, 0.1],
                'r_camera': [0.2, 0.5, 1.0],
                'r_lidar': [0.2, 0.5, 1.0],
                'gate_threshold': [6.3, 7.8, 9.0, 11.3],
                'init_frames': [2, 3, 5],
                'max_age': [30, 40, 50],
                'age_threshold': [2, 3, 4],
            },
            'bayesian_optimization': {
                'iterations': 30,
                'random_seed': 42,
            },
        }
    
    def run_complete_workflow(self, tracker_class, verbose: bool = True):
        """
        Execute complete tuning workflow
        
        Args:
            tracker_class: Your KalmanTracker class
            verbose: Print progress information
        
        Returns:
            best_parameters: Optimized KalmanParameters object
            metrics: Dictionary of performance metrics
        """
        
        logger.info("=" * 70)
        logger.info("STARTING COMPLETE PARAMETER TUNING WORKFLOW")
        logger.info("=" * 70)
        
        # Step 1: Prepare data
        logger.info("\n[STEP 1/6] Preparing data...")
        tuning_data = self._prepare_data(self.config['tuning_sequences'])
        test_data = self._prepare_data(self.config['test_sequences'])
        logger.info(f"  Tuning sequences: {len(tuning_data)}")
        logger.info(f"  Test sequences: {len(test_data)}")
        
        # Step 2: Establish baselines
        logger.info("\n[STEP 2/6] Running single-sensor baselines...")
        baselines = self._evaluate_baselines(tracker_class, tuning_data)
        self.results['baselines'] = baselines
        logger.info(f"  Camera-only MOTA: {-baselines['camera']:.2%}")
        logger.info(f"  LiDAR-only MOTA: {-baselines['lidar']:.2%}")
        
        # Step 3: Grid search
        logger.info("\n[STEP 3/6] Running grid search...")
        grid_best_params, grid_best_metric = self._run_grid_search(
            tracker_class, tuning_data
        )
        self.results['grid_search'] = {
            'best_params': grid_best_params.to_dict() if hasattr(grid_best_params, 'to_dict') else grid_best_params,
            'best_metric': grid_best_metric,
        }
        logger.info(f"  Best grid search metric: {grid_best_metric:.4f}")
        logger.info("\n[STEP 4/6] Skipping Bayesian optimization (using grid search results)")
        
        # Use grid search results as final parameters
        best_params_final = grid_best_params
        best_metric_final = grid_best_metric
        
        # Step 5: Test set evaluation
        logger.info("\n[STEP 5/6] Evaluating on test set...")
        test_metrics = self._evaluate_test_set(tracker_class, test_data, best_params_final)
        self.results['test_results'] = test_metrics
        logger.info(f"  Test set MOTA: {test_metrics['fusion_mota']:.2%}")
        logger.info(f"  Fusion gain vs camera: {test_metrics['fusion_gain_vs_camera']:.2%}")
        logger.info(f"  Fusion gain vs LiDAR: {test_metrics['fusion_gain_vs_lidar']:.2%}")
        
        # Step 6: Generate report
        logger.info("\n[STEP 6/6] Generating report...")
        self._generate_report(bayesian_best_params)
        
        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW COMPLETE")
        logger.info("=" * 70)
        
        self.results['best_parameters'] = best_params_final.to_dict() if hasattr(best_params_final, 'to_dict') else best_params_final
        return best_params_final, self.results
    
    def _prepare_data(self, sequences: List[str]) -> Dict:
        """Load and prepare KITTI data for sequences"""
        logger.debug(f"Preparing data for sequences: {sequences}")
        # TODO: Connect to your KITTI data loader
        return {}
    
    def _evaluate_baselines(self, tracker_class, data: Dict) -> Dict:
        """Evaluate single-sensor baselines"""
        from parameter_tuning import KalmanParameters
        from tuning_integration import TrackingEvaluationWrapper
        
        evaluator = TrackingEvaluationWrapper(tracker_class, self.config['tuning_sequences'])
        
        # Camera-only parameters
        camera_params = KalmanParameters(
            q_pos=0.5, q_vel=0.05,
            r_camera=0.3, r_lidar=float('inf'),
            gate_threshold=7.8,
            init_frames=3, max_age=40, age_threshold=3
        )
        camera_metric = evaluator.evaluate_parameters(camera_params, sensor_type='camera')
        
        # LiDAR-only parameters
        lidar_params = KalmanParameters(
            q_pos=0.5, q_vel=0.05,
            r_camera=float('inf'), r_lidar=0.3,
            gate_threshold=7.8,
            init_frames=3, max_age=40, age_threshold=3
        )
        lidar_metric = evaluator.evaluate_parameters(lidar_params, sensor_type='lidar')
        
        return {
            'camera': camera_metric,
            'lidar': lidar_metric,
            'camera_params': camera_params.to_dict(),
            'lidar_params': lidar_params.to_dict(),
        }
    
    def _run_grid_search(self, tracker_class, data: Dict):
        """Execute grid search phase"""
        from parameter_tuning import ParameterOptimizer, KalmanParameters
        from tuning_integration import TrackingEvaluationWrapper
        
        evaluator = TrackingEvaluationWrapper(tracker_class, self.config['tuning_sequences'])
        optimizer = ParameterOptimizer(
            metric_fn=evaluator.evaluate_parameters,
            validation_data=data
        )
        
        best_params, best_metric = optimizer.grid_search_noise(
            self.config['grid_search']
        )
        
        base_params = KalmanParameters.from_dict(best_params)
        best_threshold, gate_metric = optimizer.grid_search_gating(base_params)
        base_params.gate_threshold = best_threshold
        
        best_lifecycle_params, lifecycle_metric = optimizer.grid_search_lifecycle(base_params)
        
        return KalmanParameters.from_dict(best_lifecycle_params), lifecycle_metric
    
    def _run_bayesian_optimization(self, tracker_class, data: Dict,
                                   initial_params) -> tuple:
        """Execute Bayesian optimization phase"""
        from parameter_tuning import ParameterOptimizer
        from tuning_integration import TrackingEvaluationWrapper
        
        evaluator = TrackingEvaluationWrapper(tracker_class, self.config['tuning_sequences'])
        optimizer = ParameterOptimizer(
            metric_fn=evaluator.evaluate_parameters,
            validation_data=data
        )
        
        bounds = {
            'q_pos': (max(0.01, initial_params.q_pos * 0.5),
                     initial_params.q_pos * 2.0),
            'q_vel': (max(0.001, initial_params.q_vel * 0.5),
                     initial_params.q_vel * 2.0),
            'r_camera': (max(0.05, initial_params.r_camera * 0.5),
                        initial_params.r_camera * 2.0),
            'r_lidar': (max(0.05, initial_params.r_lidar * 0.5),
                       initial_params.r_lidar * 2.0),
            'gate_threshold': (4.0, 15.0),
            'init_frames': (2, 8),
            'max_age': (20, 60),
            'age_threshold': (2, 8),
        }
        
        iterations = self.config['bayesian_optimization']['iterations']
        best_params, best_metric = optimizer.bayesian_optimization(bounds, iterations)
        
        return best_params, best_metric
    
    def _evaluate_test_set(self, tracker_class, data: Dict,
                          best_params) -> Dict:
        """Evaluate best parameters on test set"""
        from parameter_tuning import KalmanParameters
        from tuning_integration import TrackingEvaluationWrapper
        
        evaluator = TrackingEvaluationWrapper(tracker_class, self.config['test_sequences'])
        
        # Fusion evaluation
        fusion_metric = evaluator.evaluate_parameters(best_params, sensor_type='fusion')
        fusion_mota = -fusion_metric
        
        # Camera baseline
        camera_params = KalmanParameters(**best_params.to_dict())
        camera_params.r_lidar = float('inf')
        camera_metric = evaluator.evaluate_parameters(camera_params, sensor_type='camera')
        camera_mota = -camera_metric
        
        # LiDAR baseline
        lidar_params = KalmanParameters(**best_params.to_dict())
        lidar_params.r_camera = float('inf')
        lidar_metric = evaluator.evaluate_parameters(lidar_params, sensor_type='lidar')
        lidar_mota = -lidar_metric
        
        return {
            'fusion_mota': fusion_mota,
            'camera_mota': camera_mota,
            'lidar_mota': lidar_mota,
            'fusion_gain_vs_camera': fusion_mota - camera_mota,
            'fusion_gain_vs_lidar': fusion_mota - lidar_mota,
            'best_baseline': max(camera_mota, lidar_mota),
        }
    
    def _generate_report(self, best_params):
        """Generate comprehensive tuning report"""
        report_path = self.output_dir / 'TUNING_REPORT.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PARAMETER TUNING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Tuning Sequences: {len(self.config['tuning_sequences'])}\n")
            f.write(f"Test Sequences: {len(self.config['test_sequences'])}\n\n")
            
            # Baselines
            f.write("BASELINE RESULTS (Single Sensor)\n")
            f.write("-" * 70 + "\n")
            baselines = self.results['baselines']
            f.write(f"Camera-only MOTA: {-baselines['camera']:.2%}\n")
            f.write(f"LiDAR-only MOTA: {-baselines['lidar']:.2%}\n\n")
            
            # Grid Search Results
            f.write("GRID SEARCH RESULTS\n")
            f.write("-" * 70 + "\n")
            grid_results = self.results['grid_search']
            f.write(f"Best Metric: {grid_results['best_metric']:.4f}\n")
            f.write(f"Parameters:\n")
            for key, value in grid_results['best_params'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Bayesian Optimization Results
            f.write("BAYESIAN OPTIMIZATION RESULTS\n")
            f.write("-" * 70 + "\n")
            bayesian_results = self.results['bayesian_optimization']
            f.write(f"Best Metric: {bayesian_results['best_metric']:.4f}\n")
            f.write(f"Improvement over grid: "
                   f"{(grid_results['best_metric'] - bayesian_results['best_metric']):.4f}\n")
            f.write(f"Parameters:\n")
            for key, value in bayesian_results['best_params'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Test Set Results
            f.write("TEST SET PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            test_results = self.results['test_results']
            f.write(f"Fusion MOTA: {test_results['fusion_mota']:.2%}\n")
            f.write(f"Camera-only MOTA: {test_results['camera_mota']:.2%}\n")
            f.write(f"LiDAR-only MOTA: {test_results['lidar_mota']:.2%}\n")
            f.write(f"Fusion Gain vs Camera: {test_results['fusion_gain_vs_camera']:+.2%}\n")
            f.write(f"Fusion Gain vs LiDAR: {test_results['fusion_gain_vs_lidar']:+.2%}\n\n")
            
            # Final Parameters
            f.write("FINAL OPTIMIZED PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write("Process Noise (Q):\n")
            f.write(f"  q_pos: {best_params.q_pos:.4f}\n")
            f.write(f"  q_vel: {best_params.q_vel:.4f}\n")
            f.write("Measurement Noise (R):\n")
            f.write(f"  r_camera: {best_params.r_camera:.4f}\n")
            f.write(f"  r_lidar: {best_params.r_lidar:.4f}\n")
            f.write("Data Association:\n")
            f.write(f"  gate_threshold: {best_params.gate_threshold:.4f}\n")
            f.write("Track Lifecycle:\n")
            f.write(f"  init_frames: {best_params.init_frames}\n")
            f.write(f"  max_age: {best_params.max_age}\n")
            f.write(f"  age_threshold: {best_params.age_threshold}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")
            f.write("1. Deploy the parameters above to your production tracker\n")
            f.write("2. Evaluate on additional held-out sequences to verify generalization\n")
            f.write("3. Monitor performance metrics during deployment\n")
            f.write("4. Re-tune if dataset distribution changes significantly\n")
            f.write("5. Consider seasonal/weather-specific tuning if needed\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def save_results(self):
        """Save all results to JSON"""
        results_path = self.output_dir / 'tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")


def main():
    """
    Main entry point for parameter tuning workflow
    
    Usage:
        from workflow import main
        from your_tracker_module import KalmanTracker
        
        # Set your tracker class
        main(KalmanTracker)
    """
    
    # TODO: Import your tracker class
    # from your_module import KalmanTracker
    
    # Create workflow
    workflow = ParameterTuningWorkflow(config_path='./tuning_config.json')
    
    # Run complete workflow
    # best_params, results = workflow.run_complete_workflow(KalmanTracker)
    
    # Save results
    # workflow.save_results()


if __name__ == '__main__':
    main()


# QUICK START GUIDE
QUICK_START = """
PARAMETER TUNING QUICK START
════════════════════════════════════════════════════════════════════

1. PREPARE YOUR ENVIRONMENT
   ──────────────────────────
   pip install numpy scipy pandas matplotlib
   
   Directory structure:
   ├── your_tracker.py           (your KalmanTracker class)
   ├── parameter_tuning.py       (tuning framework)
   ├── tuning_integration.py     (KITTI integration)
   ├── tuning_strategy_guide.py  (reference guide)
   ├── workflow.py               (this file)
   ├── tuning_config.json        (configuration)
   └── KITTI/                    (your KITTI data)

2. CONFIGURE TUNING
   ──────────────────
   Edit tuning_config.json:
   {
     "output_dir": "./tuning_results",
     "kitti_data_dir": "./KITTI/tracking/data_tracking_image_2",
     "tuning_sequences": ["0001", "0002", "0003", ...],
     "test_sequences": ["0011", "0012", "0013", ...],
     "grid_search": {...},
     "bayesian_optimization": {...}
   }

3. RUN TUNING
   ──────────
   from workflow import ParameterTuningWorkflow
   from your_tracker import KalmanTracker
   
   workflow = ParameterTuningWorkflow('tuning_config.json')
   best_params, results = workflow.run_complete_workflow(KalmanTracker)
   workflow.save_results()

4. INTERPRET RESULTS
   ──────────────────
   Check ./tuning_results/ for:
   • TUNING_REPORT.txt - Human-readable report
   • tuning_results.json - Detailed metrics
   • best_parameters.json - Optimal parameters
   • Visualizations (PNG files)

5. DEPLOY
   ───────
   Copy best_parameters.json to production:
   
   with open('best_parameters.json') as f:
       params = json.load(f)
   
   tracker = KalmanTracker(
       q_pos=params['q_pos'],
       q_vel=params['q_vel'],
       r_camera=params['r_camera'],
       r_lidar=params['r_lidar'],
       gate_threshold=params['gate_threshold'],
       init_frames=params['init_frames'],
       max_age=params['max_age'],
       age_threshold=params['age_threshold']
   )

EXPECTED RUNTIME
────────────────
• Grid search: 2-4 hours (depends on hardware)
• Bayesian optimization: 1-2 hours
• Test evaluation: 30 minutes
• Total: ~4-7 hours

TROUBLESHOOTING
───────────────
See tuning_strategy_guide.py for:
• Parameter interpretation
• Common pitfalls
• Debugging strategies
• Scenario-specific configurations

SUCCESS CRITERIA
────────────────
✓ Fusion MOTA > best single sensor MOTA
✓ Test set MOTA within ~2% of tuning set
✓ Fusion gains > 2% absolute MOTA
✓ All parameters have reasonable values
"""

print(QUICK_START)
