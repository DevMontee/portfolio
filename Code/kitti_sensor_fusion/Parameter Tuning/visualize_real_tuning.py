"""
TUNING RESULTS VISUALIZATION

Creates publication-quality plots from real KITTI tuning results.
Visualizes:
- Process/measurement noise tuning
- Gating threshold optimization
- Track lifecycle parameters
- Parameter sensitivity analysis
- 3D performance landscapes
- Parameter correlations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Setup plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TuningResultsVisualizer:
    """Visualize real tuning results from KITTI evaluation."""
    
    def __init__(self, results_file: str, output_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            results_file: Path to tuning_results.json
            output_dir: Where to save plots (default: same as results file)
        """
        self.results_file = Path(results_file)
        
        if output_dir is None:
            output_dir = self.results_file.parent
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self.df = self._convert_to_dataframe()
        
        print(f"✓ Loaded {len(self.results)} tuning results")
        print(f"✓ Output directory: {self.output_dir}\n")
    
    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        
        for result in self.results:
            row = result['params'].copy()
            row['mota'] = result['score']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_noise_parameters(self):
        """Plot process and measurement noise optimization."""
        print("Generating: Process & Measurement Noise Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Noise Parameter Tuning Analysis', fontsize=16, fontweight='bold')
        
        # 1. Q_pos vs Q_vel (process noise)
        ax = axes[0, 0]
        pivot_q = self.df.pivot_table(
            values='mota',
            index='q_vel',
            columns='q_pos',
            aggfunc='mean'
        )
        sns.heatmap(pivot_q, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': 'MOTA'})
        ax.set_title('Process Noise (Q): Position vs Velocity')
        ax.set_xlabel('Q Position (q_pos)')
        ax.set_ylabel('Q Velocity (q_vel)')
        
        # 2. R_camera vs R_lidar (measurement noise)
        ax = axes[0, 1]
        pivot_r = self.df.pivot_table(
            values='mota',
            index='r_lidar',
            columns='r_camera',
            aggfunc='mean'
        )
        sns.heatmap(pivot_r, annot=True, fmt='.3f', cmap='plasma', ax=ax, cbar_kws={'label': 'MOTA'})
        ax.set_title('Measurement Noise (R): Camera vs LiDAR')
        ax.set_xlabel('R Camera (r_camera)')
        ax.set_ylabel('R LiDAR (r_lidar)')
        
        # 3. Best Q parameters
        ax = axes[1, 0]
        q_best = self.df.groupby(['q_pos', 'q_vel'])['mota'].mean().reset_index()
        q_best_sorted = q_best.sort_values('mota', ascending=False).head(5)
        labels = [f"q_pos={r['q_pos']}\nq_vel={r['q_vel']}" for _, r in q_best_sorted.iterrows()]
        ax.barh(range(len(q_best_sorted)), q_best_sorted['mota'].values, color='teal')
        ax.set_yticks(range(len(q_best_sorted)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Average MOTA')
        ax.set_title('Top 5 Process Noise (Q) Combinations')
        ax.invert_yaxis()
        
        # 4. Best R parameters
        ax = axes[1, 1]
        r_best = self.df.groupby(['r_camera', 'r_lidar'])['mota'].mean().reset_index()
        r_best_sorted = r_best.sort_values('mota', ascending=False).head(5)
        labels = [f"r_cam={r['r_camera']}\nr_lid={r['r_lidar']}" for _, r in r_best_sorted.iterrows()]
        ax.barh(range(len(r_best_sorted)), r_best_sorted['mota'].values, color='coral')
        ax.set_yticks(range(len(r_best_sorted)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Average MOTA')
        ax.set_title('Top 5 Measurement Noise (R) Combinations')
        ax.invert_yaxis()
        
        plt.tight_layout()
        output_file = self.output_dir / '01_noise_parameters.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def plot_gating_threshold(self):
        """Plot gating threshold optimization."""
        print("Generating: Gating Threshold Analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Data Association: Gating Threshold Optimization', fontsize=16, fontweight='bold')
        
        # 1. Gate threshold performance curve
        ax = axes[0]
        gate_perf = self.df.groupby('gate_threshold')['mota'].agg(['mean', 'std']).reset_index()
        ax.plot(gate_perf['gate_threshold'], gate_perf['mean'], 'o-', linewidth=2.5, 
                markersize=10, color='steelblue', label='Mean MOTA')
        ax.fill_between(gate_perf['gate_threshold'], 
                        gate_perf['mean'] - gate_perf['std'],
                        gate_perf['mean'] + gate_perf['std'],
                        alpha=0.3, color='steelblue', label='±1 Std Dev')
        
        # Mark best
        best_gate = gate_perf.loc[gate_perf['mean'].idxmax()]
        ax.plot(best_gate['gate_threshold'], best_gate['mean'], 'r*', markersize=20, label='Optimal')
        
        ax.set_xlabel('Gating Threshold (χ² value)', fontsize=11)
        ax.set_ylabel('MOTA', fontsize=11)
        ax.set_title('Performance vs Gating Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gate threshold impact categories
        ax = axes[1]
        gate_categories = {
            'Too Tight\n(6.3)': self.df[self.df['gate_threshold'] == 6.3]['mota'].mean(),
            'Conservative\n(7.8)': self.df[self.df['gate_threshold'] == 7.8]['mota'].mean(),
            'Moderate\n(9.0)': self.df[self.df['gate_threshold'] == 9.0]['mota'].mean(),
            'Relaxed\n(11.3)': self.df[self.df['gate_threshold'] == 11.3]['mota'].mean(),
        }
        colors = ['red', 'orange', 'green', 'blue']
        bars = ax.bar(range(len(gate_categories)), list(gate_categories.values()), color=colors, alpha=0.7)
        ax.set_xticks(range(len(gate_categories)))
        ax.set_xticklabels(list(gate_categories.keys()))
        ax.set_ylabel('Average MOTA', fontsize=11)
        ax.set_title('Gate Threshold Categories')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / '02_gating_threshold.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def plot_lifecycle_parameters(self):
        """Plot track lifecycle parameter tuning."""
        print("Generating: Track Lifecycle Parameters...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Track Lifecycle Parameter Tuning', fontsize=16, fontweight='bold')
        
        # 1. init_frames impact
        ax = axes[0, 0]
        init_perf = self.df.groupby('init_frames')['mota'].agg(['mean', 'std']).reset_index()
        bars = ax.bar(init_perf['init_frames'].astype(str), init_perf['mean'], 
                      color='teal', alpha=0.7, yerr=init_perf['std'], capsize=5)
        ax.set_xlabel('Frames to Confirm Track (init_frames)', fontsize=11)
        ax.set_ylabel('MOTA', fontsize=11)
        ax.set_title('Track Initialization Impact')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. max_age impact
        ax = axes[0, 1]
        max_age_perf = self.df.groupby('max_age')['mota'].agg(['mean', 'std']).reset_index()
        bars = ax.bar(max_age_perf['max_age'].astype(str), max_age_perf['mean'], 
                      color='coral', alpha=0.7, yerr=max_age_perf['std'], capsize=5)
        ax.set_xlabel('Max Frames Without Detection (max_age)', fontsize=11)
        ax.set_ylabel('MOTA', fontsize=11)
        ax.set_title('Track Lifespan Impact')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. age_threshold impact
        ax = axes[1, 0]
        age_thresh_perf = self.df.groupby('age_threshold')['mota'].agg(['mean', 'std']).reset_index()
        bars = ax.bar(age_thresh_perf['age_threshold'].astype(str), age_thresh_perf['mean'], 
                      color='skyblue', alpha=0.7, yerr=age_thresh_perf['std'], capsize=5)
        ax.set_xlabel('Frames Before Activation (age_threshold)', fontsize=11)
        ax.set_ylabel('MOTA', fontsize=11)
        ax.set_title('Track Activation Impact')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Lifecycle interaction heatmap
        ax = axes[1, 1]
        lifecycle = self.df.pivot_table(
            values='mota',
            index='max_age',
            columns='init_frames',
            aggfunc='mean'
        )
        sns.heatmap(lifecycle, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                   cbar_kws={'label': 'MOTA'}, vmin=lifecycle.min().min(), vmax=lifecycle.max().max())
        ax.set_title('init_frames vs max_age Interaction')
        ax.set_xlabel('Init Frames')
        ax.set_ylabel('Max Age')
        
        plt.tight_layout()
        output_file = self.output_dir / '03_lifecycle_parameters.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def plot_sensitivity_analysis(self):
        """Plot parameter sensitivity."""
        print("Generating: Parameter Sensitivity Analysis...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate sensitivity for each parameter
        sensitivity = {}
        
        for param in self.df.columns:
            if param == 'mota':
                continue
            
            # Get unique values
            unique_vals = self.df[param].unique()
            
            if len(unique_vals) < 2:
                continue
            
            # Calculate variance explained
            perf_by_param = self.df.groupby(param)['mota'].std()
            sensitivity[param] = perf_by_param.std()
        
        # Sort by sensitivity
        sorted_sens = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        params = [s[0] for s in sorted_sens]
        sens_values = [s[1] for s in sorted_sens]
        
        # Plot
        colors = ['darkred' if s > 0.05 else 'orange' if s > 0.03 else 'green' 
                 for s in sens_values]
        bars = ax.barh(params, sens_values, color=colors, alpha=0.7)
        
        ax.set_xlabel('Parameter Sensitivity (Std Dev of Performance)', fontsize=12)
        ax.set_title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, sens_values):
            ax.text(val, bar.get_y() + bar.get_height()/2., 
                   f'{val:.4f}', ha='left', va='center', fontsize=10)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', alpha=0.7, label='High Sensitivity (>0.05)'),
            Patch(facecolor='orange', alpha=0.7, label='Medium Sensitivity (0.03-0.05)'),
            Patch(facecolor='green', alpha=0.7, label='Low Sensitivity (<0.03)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_file = self.output_dir / '04_parameter_sensitivity.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def plot_performance_distribution(self):
        """Plot overall performance distribution."""
        print("Generating: Performance Distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Overall Performance Distribution', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        ax = axes[0]
        ax.hist(self.df['mota'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(self.df['mota'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["mota"].mean():.4f}')
        ax.axvline(self.df['mota'].max(), color='green', linestyle='--', linewidth=2, label=f'Best: {self.df["mota"].max():.4f}')
        ax.set_xlabel('MOTA', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('MOTA Distribution')
        ax.legend()
        
        # 2. Best vs Worst
        ax = axes[1]
        best_result = self.df.loc[self.df['mota'].idxmax()]
        worst_result = self.df.loc[self.df['mota'].idxmin()]
        
        comparison_data = {
            'Best': best_result['mota'],
            'Mean': self.df['mota'].mean(),
            'Median': self.df['mota'].median(),
            'Worst': worst_result['mota']
        }
        
        bars = ax.bar(comparison_data.keys(), comparison_data.values(), 
                      color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        ax.set_ylabel('MOTA', fontsize=11)
        ax.set_title('Performance Statistics')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / '05_performance_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ {output_file.name}")
        plt.close()
    
    def plot_summary_report(self):
        """Generate text summary report."""
        print("Generating: Summary Report...")
        
        report_file = self.output_dir / 'VISUALIZATION_SUMMARY.txt'
        
        best_result = self.df.loc[self.df['mota'].idxmax()]
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TUNING RESULTS VISUALIZATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Total combinations tested: {len(self.df)}\n")
            f.write(f"Best MOTA: {self.df['mota'].max():.4f}\n")
            f.write(f"Mean MOTA: {self.df['mota'].mean():.4f}\n")
            f.write(f"Std Dev: {self.df['mota'].std():.4f}\n")
            f.write(f"Worst MOTA: {self.df['mota'].min():.4f}\n\n")
            
            f.write("BEST PARAMETERS FOUND:\n")
            f.write("-"*70 + "\n")
            for key, val in best_result.items():
                if key != 'mota':
                    f.write(f"{key:20s}: {val}\n")
            f.write(f"{'MOTA':20s}: {best_result['mota']:.4f}\n\n")
            
            f.write("PARAMETER RANGES EXPLORED:\n")
            f.write("-"*70 + "\n")
            for col in sorted(self.df.columns):
                if col != 'mota':
                    unique_vals = sorted(self.df[col].unique())
                    f.write(f"{col:20s}: {unique_vals}\n")
            f.write("\n")
            
            f.write("PLOTS GENERATED:\n")
            f.write("-"*70 + "\n")
            f.write("1. 01_noise_parameters.png\n")
            f.write("   - Process noise (Q) tuning\n")
            f.write("   - Measurement noise (R) tuning\n\n")
            f.write("2. 02_gating_threshold.png\n")
            f.write("   - Gating threshold optimization\n")
            f.write("   - Impact categories (Tight/Conservative/Moderate/Relaxed)\n\n")
            f.write("3. 03_lifecycle_parameters.png\n")
            f.write("   - init_frames impact\n")
            f.write("   - max_age impact\n")
            f.write("   - age_threshold impact\n\n")
            f.write("4. 04_parameter_sensitivity.png\n")
            f.write("   - Ranked parameter importance\n\n")
            f.write("5. 05_performance_distribution.png\n")
            f.write("   - Overall performance statistics\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("-"*70 + "\n")
            f.write("1. Use best_parameters.json in your tracking system\n")
            f.write("2. Evaluate on test sequences (0010-0014)\n")
            f.write("3. Compare with single-sensor baselines\n")
            f.write("4. Submit to PhD applications with real KITTI results!\n")
        
        print(f"  ✓ {report_file.name}")
    
    def generate_all(self):
        """Generate all visualizations."""
        print("="*70)
        print("GENERATING TUNING VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.plot_noise_parameters()
        self.plot_gating_threshold()
        self.plot_lifecycle_parameters()
        self.plot_sensitivity_analysis()
        self.plot_performance_distribution()
        self.plot_summary_report()
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}\n")


# Main execution
if __name__ == '__main__':
    import sys
    
    # Find results file
    results_file = Path('tuning_results') / 'tuning_results.json'
    
    if not results_file.exists():
        # Try alternative path
        results_file = Path('.') / 'tuning_results.json'
    
    if not results_file.exists():
        print("ERROR: tuning_results.json not found")
        print(f"Looked in: {Path('tuning_results') / 'tuning_results.json'}")
        sys.exit(1)
    
    visualizer = TuningResultsVisualizer(str(results_file))
    visualizer.generate_all()
