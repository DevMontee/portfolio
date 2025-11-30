"""
Occlusion Robustness Analysis Visualization
Generates publication-quality figures from occlusion test results
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path


def load_results(json_path):
    """Load occlusion test results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_performance_comparison_figure(results, output_dir="./outputs"):
    """Create comprehensive performance comparison across occlusion levels"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Extract data
    occlusion_levels = [0.0] + sorted([float(k) for k in results['occlusion_tests'].keys()])
    mota_scores = [results['baseline']['mota']]
    motp_scores = [results['baseline']['motp']]
    id_switches = [results['baseline']['id_switches']]
    
    for level in sorted(results['occlusion_tests'].keys(), key=float):
        test = results['occlusion_tests'][level]
        mota_scores.append(test['mota'])
        motp_scores.append(test['motp'])
        id_switches.append(test['id_switches'])
    
    # MOTA (Multiple Object Tracking Accuracy)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(occlusion_levels, mota_scores, 'o-', linewidth=2.5, markersize=8, 
             color='#2E86AB', label='MOTA')
    ax1.fill_between(occlusion_levels, mota_scores, alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Occlusion Level', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MOTA Score', fontsize=11, fontweight='bold')
    ax1.set_title('Tracking Accuracy vs Occlusion', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1])
    
    # Add value labels
    for x, y in zip(occlusion_levels, mota_scores):
        ax1.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MOTP (Multiple Object Tracking Precision)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(occlusion_levels, motp_scores, 's-', linewidth=2.5, markersize=8, 
             color='#A23B72', label='MOTP')
    ax2.fill_between(occlusion_levels, motp_scores, alpha=0.2, color='#A23B72')
    ax2.set_xlabel('Occlusion Level', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MOTP Score', fontsize=11, fontweight='bold')
    ax2.set_title('Localization Precision vs Occlusion', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for x, y in zip(occlusion_levels, motp_scores):
        ax2.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ID Switches (lower is better)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(occlusion_levels, id_switches, '^-', linewidth=2.5, markersize=8, 
             color='#F18F01', label='ID Switches')
    ax3.fill_between(occlusion_levels, id_switches, alpha=0.2, color='#F18F01')
    ax3.set_xlabel('Occlusion Level', fontsize=11, fontweight='bold')
    ax3.set_ylabel('ID Switches Count', fontsize=11, fontweight='bold')
    ax3.set_title('Tracking Identity Consistency vs Occlusion', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for x, y in zip(occlusion_levels, id_switches):
        ax3.text(x, y + 1, f'{int(y)}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Occlusion Robustness Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 01_performance_comparison.png")
    plt.close()


def create_degradation_analysis_figure(results, output_dir="./outputs"):
    """Analyze performance degradation with occlusion"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    baseline_mota = results['baseline']['mota']
    baseline_motp = results['baseline']['motp']
    baseline_ids = results['baseline']['id_switches']
    
    occlusion_levels = [float(k) for k in sorted(results['occlusion_tests'].keys())]
    mota_degradation = []
    motp_degradation = []
    ids_improvement = []
    
    for level in occlusion_levels:
        test = results['occlusion_tests'][str(level)]
        mota_degradation.append((baseline_mota - test['mota']) / baseline_mota * 100)
        motp_degradation.append((test['motp'] - baseline_motp) / baseline_motp * 100)
        ids_improvement.append((baseline_ids - test['id_switches']) / baseline_ids * 100)
    
    # Degradation metrics
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(occlusion_levels))
    width = 0.25
    
    ax1.bar(x_pos - width, mota_degradation, width, label='MOTA Loss %', color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos, motp_degradation, width, label='MOTP Increase %', color='#A23B72', alpha=0.8)
    ax1.bar(x_pos + width, ids_improvement, width, label='ID Switch Reduction %', color='#F18F01', alpha=0.8)
    
    ax1.set_xlabel('Occlusion Level', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Change from Baseline (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Degradation Analysis', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(occlusion_levels)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=0.8)
    
    # Occlusion severity vs detections hidden
    ax2 = fig.add_subplot(gs[0, 1])
    detections_hidden = [results['occlusion_tests'][str(level)].get('detections_hidden', 0) 
                         for level in occlusion_levels]
    
    ax2.bar(occlusion_levels, detections_hidden, color='#C73E1D', alpha=0.8, width=0.08)
    ax2.set_xlabel('Occlusion Level', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Detections Hidden', fontsize=11, fontweight='bold')
    ax2.set_title('Occlusion Severity: Objects Hidden', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for x, y in zip(occlusion_levels, detections_hidden):
        ax2.text(x, y + 5, f'{int(y)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('Robustness Degradation Under Occlusion', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_degradation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 02_degradation_analysis.png")
    plt.close()


def create_robustness_score_figure(results, output_dir="./outputs"):
    """Calculate and visualize robustness scores"""
    Path(output_dir).mkdir(exist_ok=True)
    
    baseline_mota = results['baseline']['mota']
    baseline_motp = results['baseline']['motp']
    baseline_ids = results['baseline']['id_switches']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    occlusion_levels = [0.0] + sorted([float(k) for k in results['occlusion_tests'].keys()])
    robustness_scores = [100.0]  # Baseline is 100%
    
    for level in sorted(results['occlusion_tests'].keys(), key=float):
        test = results['occlusion_tests'][level]
        
        # Composite robustness score (normalized)
        mota_component = test['mota'] / baseline_mota * 0.5  # 50% weight
        ids_component = test['id_switches'] / baseline_ids * 0.5  # 50% weight (lower is better)
        
        robustness = (mota_component + (1 - ids_component + 1) / 2) / 2 * 100
        robustness_scores.append(robustness)
    
    # Create color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(occlusion_levels)))
    
    bars = ax.bar(range(len(occlusion_levels)), robustness_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Occlusion Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Robustness Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Tracker Robustness Score Under Occlusion', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(occlusion_levels)))
    ax.set_xticklabels([f'{x:.2f}' for x in occlusion_levels])
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels and benchmark lines
    for i, (bar, score) in enumerate(zip(bars, robustness_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add robustness zones
    ax.axhspan(80, 110, alpha=0.1, color='green', label='Excellent (>80%)')
    ax.axhspan(60, 80, alpha=0.1, color='yellow', label='Good (60-80%)')
    ax.axhspan(0, 60, alpha=0.1, color='red', label='Degraded (<60%)')
    
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_robustness_score.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 03_robustness_score.png")
    plt.close()


def create_comparison_table(results, output_dir="./outputs"):
    """Generate detailed comparison table as figure"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    rows = ['MOTA', 'MOTP', 'ID Switches', 'Detections Hidden']
    baseline = results['baseline']
    
    table_data = [
        ['Baseline', f"{baseline['mota']:.4f}", f"{baseline['motp']:.4f}", f"{baseline['id_switches']}", '0']
    ]
    
    for level in sorted(results['occlusion_tests'].keys(), key=float):
        test = results['occlusion_tests'][level]
        table_data.append([
            f"Occl. {level}",
            f"{test['mota']:.4f}",
            f"{test['motp']:.4f}",
            f"{test['id_switches']}",
            f"{test.get('detections_hidden', 0)}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, 
                     colLabels=['Scenario', 'MOTA ↑', 'MOTP ↓', 'ID Switches ↓', 'Hidden Dets.'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    for i in range(1, len(table_data) + 1):
        color = '#E8F4F8' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')
    
    # Add legend
    legend_text = "↑ = Higher is Better  |  ↓ = Lower is Better"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')
    
    fig.suptitle('Occlusion Test Results Summary', fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(f'{output_dir}/04_comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 04_comparison_table.png")
    plt.close()


def generate_summary_report(results, output_dir="./outputs"):
    """Generate text summary of findings"""
    Path(output_dir).mkdir(exist_ok=True)
    
    baseline = results['baseline']
    best_test = max(results['occlusion_tests'].items(), 
                    key=lambda x: x[1]['mota'])
    worst_test = min(results['occlusion_tests'].items(), 
                     key=lambda x: x[1]['mota'])
    
    report = f"""
OCCLUSION ROBUSTNESS ANALYSIS REPORT
{'='*60}

BASELINE PERFORMANCE (No Occlusion)
  • MOTA (Accuracy): {baseline['mota']:.4f}
  • MOTP (Precision): {baseline['motp']:.4f}
  • ID Switches: {baseline['id_switches']}

KEY FINDINGS
  • Best Performance: Occlusion level {best_test[0]} (MOTA: {best_test[1]['mota']:.4f})
  • Worst Performance: Occlusion level {worst_test[0]} (MOTA: {worst_test[1]['mota']:.4f})
  • Maximum MOTA Degradation: {(baseline['mota'] - worst_test[1]['mota']) / baseline['mota'] * 100:.2f}%

ROBUSTNESS INSIGHTS
  • The tracker shows reasonable robustness to low-level occlusion (0.1)
  • Performance degrades significantly at moderate occlusion levels (0.25+)
  • ID switching improves with higher occlusion (fewer tracked objects = fewer switches)
  • Localization precision (MOTP) deteriorates as occlusion increases

RECOMMENDATIONS
  1. Implement occlusion-aware gating thresholds for robust association
  2. Consider multi-hypothesis tracking for occluded objects
  3. Use temporal predictions to maintain track continuity
  4. Tune Kalman filter process noise for occluded scenarios

{'='*60}
"""
    
    with open(f'{output_dir}/ANALYSIS_REPORT.txt', 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: ANALYSIS_REPORT.txt")
    print(report)


def main(json_path, output_dir="./outputs"):
    """Generate all visualizations"""
    print("\n📊 Generating Occlusion Robustness Visualizations...\n")
    
    results = load_results(json_path)
    
    create_performance_comparison_figure(results, output_dir)
    create_degradation_analysis_figure(results, output_dir)
    create_robustness_score_figure(results, output_dir)
    create_comparison_table(results, output_dir)
    generate_summary_report(results, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  1. 01_performance_comparison.png - MOTA, MOTP, ID switches curves")
    print(f"  2. 02_degradation_analysis.png - Performance loss analysis")
    print(f"  3. 03_robustness_score.png - Composite robustness metrics")
    print(f"  4. 04_comparison_table.png - Summary table with all metrics")
    print(f"  5. ANALYSIS_REPORT.txt - Detailed findings and recommendations\n")


if __name__ == "__main__":
    # Update paths to your local directories
    json_file = r"D:\kitti_sensor_fusion\Occlusion Handling\OCCLUSION_TEST_RESULTS.json"
    output_directory = r"D:\kitti_sensor_fusion\Occlusion Handling\visualizations"
    
    main(json_file, output_directory)
