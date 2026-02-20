#!/usr/bin/env python3
"""
Heatmap Comparison Tool

Creates side-by-side and difference heatmaps to compare performance patterns
between Clean Restart and Green Run across all MMLU subtasks.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_heatmap_data(csv_path):
    """Load heatmap data from CSV and prepare for comparison."""
    df = pd.read_csv(csv_path, index_col=0)
    
    # Remove the AVG column for comparison (we'll add it back later)
    if 'AVG' in df.columns:
        avg_col = df['AVG'].copy()
        df = df.drop('AVG', axis=1)
    else:
        avg_col = df.mean(axis=1)
    
    # Clean subject names - extract just the subject name before (Base:...)
    clean_index = []
    for subject in df.index:
        # Extract subject name before " (Base:" if it exists
        if " (Base:" in subject:
            clean_name = subject.split(" (Base:")[0].strip().strip('"')
        else:
            clean_name = subject.strip().strip('"')
        clean_index.append(clean_name)
    
    df.index = clean_index
    avg_col.index = clean_index
    
    return df, avg_col

def create_comparison_heatmaps():
    """Create comparison visualizations between Clean Restart and Green Run."""
    
    print("🔍 CREATING HEATMAP COMPARISONS")
    print("="*60)
    
    # Load both heatmap datasets
    clean_restart_path = "evaluation_results/2025-08-17_03-57-04/2025-08-17_03-57-04_baseline_relative_baseline_relative_clean_restart_data.csv"
    green_run_path = "evaluation_results/2025-08-17_03-57-04/2025-08-17_03-57-04_baseline_relative_baseline_relative_green_run_data.csv"
    
    try:
        clean_data, clean_avg = load_heatmap_data(clean_restart_path)
        green_data, green_avg = load_heatmap_data(green_run_path)
        print(f"✅ Loaded Clean Restart data: {clean_data.shape}")
        print(f"✅ Loaded Green Run data: {green_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Align the datasets (same subjects, overlapping steps)
    common_subjects = clean_data.index.intersection(green_data.index)
    common_steps = clean_data.columns.intersection(green_data.columns)
    
    print(f"📊 Common subjects: {len(common_subjects)}")
    print(f"📊 Common steps: {len(common_steps)}")
    
    # Create aligned datasets
    clean_aligned = clean_data.loc[common_subjects, common_steps]
    green_aligned = green_data.loc[common_subjects, common_steps]
    
    # Sort subjects by average performance across both runs
    combined_avg = (clean_avg.loc[common_subjects] + green_avg.loc[common_subjects]) / 2
    sorted_subjects = combined_avg.sort_values(ascending=False).index
    
    clean_aligned = clean_aligned.loc[sorted_subjects]
    green_aligned = green_aligned.loc[sorted_subjects]
    
    # 1. SIDE-BY-SIDE COMPARISON
    create_side_by_side_heatmaps(clean_aligned, green_aligned, sorted_subjects)
    
    # 2. DIFFERENCE HEATMAP
    create_difference_heatmap(clean_aligned, green_aligned, sorted_subjects)
    
    # 3. PERFORMANCE ADVANTAGE ANALYSIS
    analyze_performance_advantages(clean_aligned, green_aligned, sorted_subjects)
    
    print("\n🎉 All comparison visualizations created!")

def create_side_by_side_heatmaps(clean_data, green_data, sorted_subjects):
    """Create side-by-side heatmaps for direct comparison."""
    
    print("\n📊 Creating side-by-side heatmaps...")
    
    # Create figure with subplots (16:9 aspect ratio)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(48, 27))
    
    # Clean subject names for display
    clean_subject_names = [subj.split('(')[0].strip() for subj in sorted_subjects]
    step_labels = [col.replace('k', 'k') for col in clean_data.columns]
    
    # Heatmap 1: Clean Restart
    im1 = ax1.imshow(clean_data.values, cmap='RdYlGn', aspect='auto', 
                     vmin=-50, vmax=50, interpolation='nearest')
    ax1.set_title('🔵 Clean Restart (LR 0.0001)\nBaseline-Relative Performance', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(step_labels)))
    ax1.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(range(len(clean_subject_names)))
    ax1.set_yticklabels(clean_subject_names, fontsize=9)
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MMLU Subtasks', fontsize=12, fontweight='bold')
    
    # Heatmap 2: Green Run
    im2 = ax2.imshow(green_data.values, cmap='RdYlGn', aspect='auto', 
                     vmin=-50, vmax=50, interpolation='nearest')
    ax2.set_title('🟢 Green Run (LR 0.001)\nBaseline-Relative Performance', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(step_labels)))
    ax2.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_yticks(range(len(clean_subject_names)))
    ax2.set_yticklabels(clean_subject_names, fontsize=9)
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MMLU Subtasks', fontsize=12, fontweight='bold')
    
    # Heatmap 3: Difference (Green - Clean)
    difference = green_data.values - clean_data.values
    im3 = ax3.imshow(difference, cmap='RdBu_r', aspect='auto', 
                     vmin=-20, vmax=20, interpolation='nearest')
    ax3.set_title('⚖️ Performance Difference\n(Green Run - Clean Restart)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(range(len(step_labels)))
    ax3.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=10)
    ax3.set_yticks(range(len(clean_subject_names)))
    ax3.set_yticklabels(clean_subject_names, fontsize=9)
    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MMLU Subtasks', fontsize=12, fontweight='bold')
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Improvement over Baseline (%)', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Improvement over Baseline (%)', fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Performance Difference (%)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('heatmap_side_by_side_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Side-by-side heatmaps saved: heatmap_side_by_side_comparison.png")
    plt.close()

def create_difference_heatmap(clean_data, green_data, sorted_subjects):
    """Create a focused difference heatmap with analysis."""
    
    print("\n🔍 Creating detailed difference analysis...")
    
    # Calculate differences
    difference = green_data.values - clean_data.values
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Clean subject names
    clean_subject_names = [subj.split('(')[0].strip() for subj in sorted_subjects]
    step_labels = [col.replace('k', 'k') for col in clean_data.columns]
    
    # Create heatmap
    im = plt.imshow(difference, cmap='RdBu_r', aspect='auto', 
                    vmin=-25, vmax=25, interpolation='nearest')
    
    plt.title('⚖️ Performance Advantage: Green Run vs Clean Restart\n' + 
              'Blue = Green Run Better, Red = Clean Restart Better', 
              fontsize=18, fontweight='bold', pad=20)
    
    plt.xticks(range(len(step_labels)), step_labels, rotation=45, ha='right', fontsize=11)
    plt.yticks(range(len(clean_subject_names)), clean_subject_names, fontsize=10)
    plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
    plt.ylabel('MMLU Subtasks (Best → Worst Average Performance)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Performance Difference (Green - Clean) %', fontweight='bold', fontsize=12)
    
    # Add statistics text
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    green_better = np.sum(difference > 0)
    clean_better = np.sum(difference < 0)
    total_cells = difference.size
    
    stats_text = f"""Statistics:
Mean Difference: {mean_diff:+.2f}%
Std Deviation: {std_diff:.2f}%
Green Run Better: {green_better}/{total_cells} ({green_better/total_cells*100:.1f}%)
Clean Restart Better: {clean_better}/{total_cells} ({clean_better/total_cells*100:.1f}%)"""
    
    plt.text(len(step_labels)*1.05, len(clean_subject_names)*0.8, stats_text, 
             fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('heatmap_performance_difference.png', dpi=300, bbox_inches='tight')
    print(f"📊 Difference heatmap saved: heatmap_performance_difference.png")
    plt.close()

def analyze_performance_advantages(clean_data, green_data, sorted_subjects):
    """Analyze where each run has systematic advantages."""
    
    print("\n📈 PERFORMANCE ADVANTAGE ANALYSIS")
    print("-" * 50)
    
    # Calculate differences
    difference = green_data.values - clean_data.values
    
    # Subject-wise analysis
    subject_advantages = np.mean(difference, axis=1)
    clean_subject_names = [subj.split('(')[0].strip() for subj in sorted_subjects]
    
    # Find subjects where each run excels
    green_advantage_mask = subject_advantages > 2.0  # >2% advantage
    clean_advantage_mask = subject_advantages < -2.0  # >2% advantage
    
    print("🟢 **Subjects where Green Run excels (>2% advantage):**")
    green_excels = [(name, adv) for name, adv in zip(clean_subject_names, subject_advantages) if adv > 2.0]
    green_excels.sort(key=lambda x: x[1], reverse=True)
    for subject, advantage in green_excels[:10]:
        print(f"  - {subject:<30}: +{advantage:.2f}%")
    
    print(f"\n🔵 **Subjects where Clean Restart excels (>2% advantage):**")
    clean_excels = [(name, -adv) for name, adv in zip(clean_subject_names, subject_advantages) if adv < -2.0]
    clean_excels.sort(key=lambda x: x[1], reverse=True)
    for subject, advantage in clean_excels[:10]:
        print(f"  - {subject:<30}: +{advantage:.2f}%")
    
    # Time-based analysis
    step_advantages = np.mean(difference, axis=0)
    step_labels = list(clean_data.columns)
    
    print(f"\n⏰ **Time-based Performance Patterns:**")
    
    # Early, middle, late performance
    n_steps = len(step_labels)
    early_avg = np.mean(step_advantages[:n_steps//3])
    middle_avg = np.mean(step_advantages[n_steps//3:2*n_steps//3])
    late_avg = np.mean(step_advantages[2*n_steps//3:])
    
    print(f"  - Early training advantage: {early_avg:+.2f}% ({'Green Run' if early_avg > 0 else 'Clean Restart'})")
    print(f"  - Middle training advantage: {middle_avg:+.2f}% ({'Green Run' if middle_avg > 0 else 'Clean Restart'})")
    print(f"  - Late training advantage: {late_avg:+.2f}% ({'Green Run' if late_avg > 0 else 'Clean Restart'})")
    
    # Overall statistics
    print(f"\n📊 **Overall Performance Comparison:**")
    print(f"  - Average advantage: {np.mean(difference):+.2f}% ({'Green Run' if np.mean(difference) > 0 else 'Clean Restart'})")
    print(f"  - Std deviation: {np.std(difference):.2f}%")
    print(f"  - Max Green advantage: +{np.max(difference):.2f}%")
    print(f"  - Max Clean advantage: +{-np.min(difference):.2f}%")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'Subject': clean_subject_names,
        'Average_Advantage_Green_vs_Clean': subject_advantages,
        'Green_Better': subject_advantages > 0,
        'Advantage_Magnitude': np.abs(subject_advantages)
    })
    
    analysis_df.to_csv('performance_advantage_analysis.csv', index=False)
    print(f"\n📋 Detailed analysis saved: performance_advantage_analysis.csv")

if __name__ == "__main__":
    create_comparison_heatmaps()