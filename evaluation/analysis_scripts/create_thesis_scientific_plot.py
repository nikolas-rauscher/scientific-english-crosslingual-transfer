#!/usr/bin/env python3
"""
Scientific MMLU Performance Plot for Thesis

Creates a comprehensive three-level visualization:
1. Overall MMLU score
2. Category scores (Humanities, STEM, Social Sciences, Other)
3. All individual subtasks grouped by category

Usage:
    python create_thesis_scientific_plot.py <results_dir> [output_name]
    python create_thesis_scientific_plot.py evaluation_results/pretraining-logs-lr-001-OPTIMIZED-clean-restart-v2
"""

import pandas as pd
import sys
import json
from pathlib import Path
import numpy as np
import os

# Fix matplotlib display issues on cluster
os.environ['DISPLAY'] = ''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Category definitions for MMLU
MMLU_CATEGORIES = {
    'STEM': [
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
        'high_school_statistics', 'machine_learning'
    ],
    'Humanities': [
        'formal_logic', 'high_school_european_history', 'high_school_us_history',
        'high_school_world_history', 'international_law', 'jurisprudence',
        'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
        'prehistory', 'professional_law', 'world_religions'
    ],
    'Social Sciences': [
        'econometrics', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology',
        'human_sexuality', 'professional_psychology', 'public_relations',
        'security_studies', 'sociology', 'us_foreign_policy'
    ],
    'Other': [
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
        'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine',
        'virology'
    ]
}


def clean_subject_name(subject):
    """Convert subject name to readable format."""
    return subject.replace('_', ' ').title()


def find_best_checkpoint(results_dir):
    """Find the best checkpoint from comparison table."""
    comparison_csv = None
    for csv_file in Path(results_dir).glob('*_comparison_table.csv'):
        comparison_csv = csv_file
        break

    if not comparison_csv:
        raise FileNotFoundError(f"No comparison table found in {results_dir}")

    df = pd.read_csv(comparison_csv)

    # Find checkpoint with highest MMLU overall score
    if 'mmlu_overall' not in df.columns:
        raise ValueError("No mmlu_overall column found in comparison table")

    best_idx = df['mmlu_overall'].idxmax()
    best_checkpoint = df.loc[best_idx]

    print(f"Best checkpoint: Step {best_checkpoint['step']}, MMLU: {best_checkpoint['mmlu_overall']:.4f}")
    return best_checkpoint


def load_subtask_data(results_dir):
    """Load individual subtask data from CSV files."""
    results_path = Path(results_dir)

    # Look for subtasks progression CSV data
    csv_file = None
    for candidate in results_path.glob('*_subtasks_progression_*_data.csv'):
        csv_file = candidate
        break

    if not csv_file:
        raise FileNotFoundError(f"No subtasks CSV data file found in {results_dir}")

    # Load CSV data
    df = pd.read_csv(csv_file, index_col=0)

    # Use the AVG column for best average performance across training
    if 'AVG' not in df.columns:
        raise ValueError("No AVG column found in subtasks CSV")

    # Extract subtask scores from AVG column
    subtask_scores = {}

    for task_name in df.index:
        # Skip special rows
        if task_name in ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)']:
            continue

        avg_score = df.loc[task_name, 'AVG']

        if pd.notna(avg_score):
            # Clean up task name - remove average annotation if present
            clean_name = task_name.split(' (Avg:')[0].strip()
            # Convert to lowercase with underscores for matching
            clean_name = clean_name.lower().replace(' ', '_')

            subtask_scores[clean_name] = avg_score

    return subtask_scores


def create_scientific_plot(results_dir, output_name=None):
    """Create comprehensive three-level scientific plot for thesis."""

    # Load best checkpoint data
    best_checkpoint = find_best_checkpoint(results_dir)

    # Load subtask data
    subtask_scores = load_subtask_data(results_dir)

    # Create figure with three subplots vertically
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 1.5, 4]})
    fig.suptitle('MMLU Performance Analysis: Three-Level Breakdown',
                 fontsize=16, fontweight='bold', y=0.995)

    # Color scheme
    color_overall = '#2E7D32'  # Green
    category_colors = {
        'Humanities': '#1565C0',      # Blue
        'STEM': '#2E7D32',            # Green
        'Social Sciences': '#E65100', # Orange
        'Other': '#6A1B9A'            # Purple
    }

    # Plot 1: Overall MMLU Score
    ax1 = axes[0]
    overall_score = best_checkpoint['mmlu_overall']
    ax1.barh([0], [overall_score], color=color_overall, height=0.6, edgecolor='black', linewidth=1.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Level 1: Overall MMLU Performance', fontsize=13, fontweight='bold', pad=10)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['MMLU Overall'], fontsize=11)
    ax1.text(overall_score + 0.02, 0, f'{overall_score:.4f}',
             va='center', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=0.25, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Random Baseline (0.25)')
    ax1.legend(loc='upper right', fontsize=9)

    # Plot 2: Category Scores
    ax2 = axes[1]
    categories = ['Humanities', 'STEM', 'Social Sciences', 'Other']
    category_scores = []
    category_labels = []
    colors_list = []

    for cat in categories:
        col_name = f'mmlu_{cat.lower().replace(" ", "_")}'
        if col_name in best_checkpoint.index:
            score = best_checkpoint[col_name]
            if pd.notna(score):
                category_scores.append(score)
                category_labels.append(cat)
                colors_list.append(category_colors[cat])

    y_pos = np.arange(len(category_labels))
    ax2.barh(y_pos, category_scores, color=colors_list, height=0.6,
             edgecolor='black', linewidth=1.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Level 2: MMLU Category Performance', fontsize=13, fontweight='bold', pad=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(category_labels, fontsize=11)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=0.25, color='red', linestyle=':', alpha=0.5, linewidth=1)

    for i, (score, label) in enumerate(zip(category_scores, category_labels)):
        ax2.text(score + 0.02, i, f'{score:.4f}',
                va='center', fontsize=10, fontweight='bold')

    # Plot 3: All Individual Subtasks grouped by category
    ax3 = axes[2]

    # Organize subtasks by category
    grouped_subtasks = []
    grouped_scores = []
    grouped_colors = []

    for category, subjects in MMLU_CATEGORIES.items():
        # Add category separator
        grouped_subtasks.append(f'--- {category} ---')
        grouped_scores.append(0)
        grouped_colors.append('white')

        # Add subjects in this category
        category_subjects = []
        for subject in subjects:
            if subject in subtask_scores:
                category_subjects.append((subject, subtask_scores[subject]))

        # Sort subjects by score within category
        category_subjects.sort(key=lambda x: x[1], reverse=True)

        for subject, score in category_subjects:
            grouped_subtasks.append(f'  {clean_subject_name(subject)}')
            grouped_scores.append(score)
            grouped_colors.append(category_colors[category])

    # Create horizontal bar plot
    y_pos = np.arange(len(grouped_subtasks))
    bars = ax3.barh(y_pos, grouped_scores, color=grouped_colors, height=0.7,
                    edgecolor='black', linewidth=0.5, alpha=0.8)

    ax3.set_xlim(0, 1)
    ax3.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Level 3: Individual MMLU Subtask Performance (Grouped by Category)',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(grouped_subtasks, fontsize=8)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.axvline(x=0.25, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax3.invert_yaxis()  # Best scores at top

    # Add score labels for subtasks
    for i, (score, label) in enumerate(zip(grouped_scores, grouped_subtasks)):
        if score > 0 and not label.startswith('---'):
            ax3.text(score + 0.01, i, f'{score:.3f}',
                    va='center', fontsize=7)

    # Style category separators
    for i, label in enumerate(grouped_subtasks):
        if label.startswith('---'):
            ax3.axhline(y=i, color='black', linewidth=2, linestyle='-', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save plot
    if output_name is None:
        output_name = f"{Path(results_dir).name}_thesis_scientific_plot.png"

    output_path = Path(results_dir) / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved scientific plot to: {output_path}")

    # Also save as PDF for publication quality
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF version to: {pdf_path}")

    plt.close()

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_thesis_scientific_plot.py <results_dir> [output_name]")
        print("Example: python create_thesis_scientific_plot.py evaluation_results/pretraining-logs-lr-001-OPTIMIZED-clean-restart-v2")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(results_dir).exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    try:
        output_path = create_scientific_plot(results_dir, output_name)
        print(f"\nSuccess! Scientific plot created: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()