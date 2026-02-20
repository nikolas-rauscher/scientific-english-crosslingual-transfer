#!/usr/bin/env python3
"""
Create horizontal bar chart comparing best checkpoints from Clean Restart, Green Run, and T5-Base baseline.
Shows all individual MMLU tasks on Y-axis with performance scores on X-axis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For cluster compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple

# Ensure no display issues on cluster
os.environ['DISPLAY'] = ''

class MMComparisonBarChart:
    def __init__(self):
        # Use modern, vibrant colors that distinguish baseline from trained models
        self.run_colors = {
            'T5-Base (Baseline)': '#FF6B6B',      # Coral red for baseline (untrained)
            'Clean Restart (Best)': '#4ECDC4',    # Turquoise for clean restart (trained)
            'Green Run (Best)': '#95E77E',        # Light green for green run (trained)
        }
        
    def load_evaluation_data(self, run_dir: str) -> List[Dict]:
        """Load all evaluation JSON files from the run directory."""
        print(f"🔍 Loading evaluation data from: {run_dir}")
        
        results = []
        run_path = Path(run_dir)
        
        if not run_path.exists():
            print(f"❌ Directory not found: {run_dir}")
            return results
        
        # Search for JSON files in evaluation subdirectories
        json_files = list(run_path.rglob("*.json"))
        
        if not json_files:
            print(f"⚠️  No JSON files found in {run_dir}")
            return results
        
        print(f"📄 Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract evaluation results
                if 'results' in data:
                    result_entry = self.parse_evaluation_result(data['results'], str(json_file))
                    if result_entry:
                        results.append(result_entry)
                        
            except Exception as e:
                print(f"⚠️  Error reading {json_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(results)} evaluation results")
        return results
    
    def parse_evaluation_result(self, results_data: Dict, file_path: str) -> Dict:
        """Parse individual evaluation result and extract MMLU scores."""
        result = {
            'file_path': file_path,
            'step': self.extract_step_from_path(file_path),
            'run_type': self.extract_run_type_from_path(file_path),
            'is_baseline': 'baseline' in file_path.lower() or 't5_base' in file_path.lower()
        }
        
        # Extract MMLU overall score - support both formats
        if 'global_mmlu_full_en' in results_data:
            # Global MMLU format
            mmlu_data = results_data['global_mmlu_full_en']
            if 'acc,none' in mmlu_data:
                result['mmlu_overall'] = mmlu_data['acc,none']
                
            # Extract individual subject scores for global MMLU
            for key, value in results_data.items():
                if key.startswith('global_mmlu_full_en_') and isinstance(value, dict):
                    # Skip category keys
                    if key not in ['global_mmlu_full_en', 'global_mmlu_full_en_humanities',
                                  'global_mmlu_full_en_social_sciences', 'global_mmlu_full_en_stem',
                                  'global_mmlu_full_en_other']:
                        subject = key.replace('global_mmlu_full_en_', '')
                        if 'acc,none' in value:
                            result[f'mmlu_subject_{subject}'] = value['acc,none']
                            
        elif 'mmlu_flan_n_shot_loglikelihood' in results_data:
            # Original FLAN format (fallback)
            mmlu_data = results_data['mmlu_flan_n_shot_loglikelihood']
            if 'acc,none' in mmlu_data:
                result['mmlu_overall'] = mmlu_data['acc,none']
                
            # Extract individual MMLU subject scores for FLAN format
            for key, value in results_data.items():
                if key.startswith('mmlu_flan_n_shot_loglikelihood_'):
                    subject = key.replace('mmlu_flan_n_shot_loglikelihood_', '')
                    if 'acc,none' in value:
                        result[f'mmlu_subject_{subject}'] = value['acc,none']
        
        return result
    
    def extract_step_from_path(self, file_path: str) -> int:
        """Extract training step from file path."""
        import re
        
        # Look for step patterns in the path
        step_patterns = [
            r'step-(\d+)',
            r'(\d+)k',
            r'(\d+)000',
        ]
        
        for pattern in step_patterns:
            match = re.search(pattern, file_path)
            if match:
                step_str = match.group(1)
                if 'k' in file_path:
                    return int(step_str) * 1000
                return int(step_str)
        
        return 0
    
    def extract_run_type_from_path(self, file_path: str) -> str:
        """Extract run type from file path."""
        file_path_lower = file_path.lower()
        
        if 'baseline' in file_path_lower or 't5_base' in file_path_lower:
            return 'baseline'
        elif 'clean_restart' in file_path_lower:
            return 'clean_restart'
        elif 'green' in file_path_lower:
            return 'green_run'
        elif 'flan' in file_path_lower:
            return 'flan_t5'
        else:
            return 'unknown'
    
    def find_best_checkpoints(self, results: List[Dict]) -> Dict[str, Dict]:
        """Find the best checkpoint for each run type based on MMLU overall score."""
        print("🎯 Finding best checkpoints for each run type...")
        
        run_types = ['baseline', 'clean_restart', 'green_run']
        best_checkpoints = {}
        
        for run_type in run_types:
            run_results = [r for r in results if r['run_type'] == run_type and 'mmlu_overall' in r]
            
            if not run_results:
                print(f"⚠️  No results found for {run_type}")
                continue
            
            # Find best checkpoint based on MMLU overall score
            best_result = max(run_results, key=lambda x: x['mmlu_overall'])
            best_checkpoints[run_type] = best_result
            
            print(f"✅ {run_type}: Step {best_result['step']}, MMLU: {best_result['mmlu_overall']:.4f}")
        
        return best_checkpoints
    
    def extract_mmlu_subjects(self, best_checkpoints: Dict[str, Dict]) -> List[str]:
        """Extract all available MMLU subjects from the best checkpoints."""
        all_subjects = set()
        
        for checkpoint_data in best_checkpoints.values():
            for key in checkpoint_data.keys():
                if key.startswith('mmlu_subject_'):
                    subject = key.replace('mmlu_subject_', '')
                    all_subjects.add(subject)
        
        # Sort subjects alphabetically for consistent ordering
        subjects_list = sorted(list(all_subjects))
        print(f"📝 Found {len(subjects_list)} MMLU subjects")
        
        return subjects_list
    
    def create_comparison_data(self, best_checkpoints: Dict[str, Dict], subjects: List[str]) -> pd.DataFrame:
        """Create DataFrame with comparison data for all runs and subjects."""
        comparison_data = []
        
        # Map run types to display names
        run_display_names = {
            'baseline': 'T5-Base (Baseline)',
            'clean_restart': 'Clean Restart (Best)',
            'green_run': 'Green Run (Best)'
        }
        
        for subject in subjects:
            subject_key = f'mmlu_subject_{subject}'
            
            for run_type, display_name in run_display_names.items():
                if run_type in best_checkpoints:
                    checkpoint = best_checkpoints[run_type]
                    score = checkpoint.get(subject_key, np.nan)
                    
                    if pd.notna(score):
                        comparison_data.append({
                            'Subject': subject.replace('_', ' ').title(),
                            'Run': display_name,
                            'Score': score,
                            'Step': checkpoint.get('step', 0)
                        })
        
        df = pd.DataFrame(comparison_data)
        print(f"📊 Created comparison data: {len(df)} rows")
        
        return df
    
    def create_horizontal_bar_chart(self, df: pd.DataFrame, save_path: str):
        """Create horizontal bar chart comparing all runs across MMLU subjects."""
        print("🎨 Creating horizontal bar chart...")
        
        # Calculate average scores for sorting
        subject_averages = df.groupby('Subject')['Score'].mean().sort_values(ascending=True)
        sorted_subjects = subject_averages.index.tolist()
        
        # Set up the bar positions with more spacing between tasks
        group_spacing = 1.8  # Space between task groups
        bar_height = 0.45  # Thicker bars
        bar_gap = 0.05  # Small gap between bars in same group
        
        # Create the plot with adjusted height for increased spacing
        # Calculate exact height needed based on spacing
        total_height = len(sorted_subjects) * group_spacing * 0.3  # More compact
        fig, ax = plt.subplots(figsize=(16, max(total_height, 20)))
        
        y_positions = np.arange(len(sorted_subjects)) * group_spacing
        
        # Create bars for each run type
        run_types = ['T5-Base (Baseline)', 'Clean Restart (Best)', 'Green Run (Best)']
        
        for i, run_type in enumerate(run_types):
            run_data = df[df['Run'] == run_type]
            
            scores = []
            for subject in sorted_subjects:
                subject_data = run_data[run_data['Subject'] == subject]
                if not subject_data.empty:
                    scores.append(subject_data['Score'].iloc[0])
                else:
                    scores.append(0)  # Missing data
            
            # Plot bars with gap between them
            bar_offset = i * (bar_height + bar_gap)
            
            # Different alpha for baseline vs trained models
            alpha_value = 0.7 if 'Baseline' in run_type else 0.9
            
            bars = ax.barh(y_positions + bar_offset, scores, 
                          height=bar_height, 
                          label=run_type,
                          color=self.run_colors[run_type],
                          alpha=alpha_value,
                          edgecolor='black',
                          linewidth=0.5)
            
            # Add score labels on bars
            for j, (bar, score) in enumerate(zip(bars, scores)):
                if score > 0:
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                           f'{score:.3f}', 
                           ha='left', va='center', fontsize=7, fontweight='bold')
        
        # Customize the plot - center y-ticks in the middle of all three bars
        # Calculate the center position of the three bars
        total_bars_height = 3 * bar_height + 2 * bar_gap
        center_offset = total_bars_height / 2 - bar_height / 2
        ax.set_yticks(y_positions + center_offset)
        ax.set_yticklabels(sorted_subjects, fontsize=10)
        ax.set_xlabel('MMLU Accuracy Score', fontweight='bold', fontsize=14)
        ax.set_ylabel('MMLU Tasks (Sorted by Average Performance)', fontweight='bold', fontsize=14)
        
        # Set title
        plt.title('MMLU Performance Comparison: Best Checkpoints vs Baseline\n' + 
                 'Clean Restart vs Green Run vs T5-Base', 
                 fontweight='bold', fontsize=16, pad=20)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        
        # Set grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set x-axis limits (max 0.5 since we don't go above that)
        ax.set_xlim(0, 0.5)
        
        # Add vertical lines for reference
        ax.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add reference line labels - adjust position for new spacing
        max_y_pos = (len(sorted_subjects) - 1) * group_spacing + 2 * (bar_height + bar_gap) + bar_height
        ax.text(0.25, max_y_pos, 'Random\nChance', ha='center', va='bottom', 
                fontsize=8, color='gray', rotation=90)
        ax.text(0.5, max_y_pos, '50%\nAccuracy', ha='center', va='bottom', 
                fontsize=8, color='gray', rotation=90)
        
        # Adjust margins to reduce whitespace
        ax.margins(y=0.01)  # Reduce vertical margins to 1%
        
        # Set y-axis limits to fit data tightly
        ax.set_ylim(-0.5, (len(sorted_subjects) - 1) * group_spacing + total_bars_height + 0.5)
        
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        # Save the plot
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                       edgecolor='none', format='png', transparent=False)
            print(f"📊 Horizontal bar chart saved to: {save_path}")
            
            # Also save summary statistics
            summary_path = save_path.replace('.png', '_summary.csv')
            summary_stats = df.groupby('Run')['Score'].agg(['mean', 'std', 'min', 'max', 'count'])
            summary_stats.to_csv(summary_path)
            print(f"📋 Summary statistics saved to: {summary_path}")
            
        except Exception as e:
            print(f"⚠️  Error saving plot: {e}")
        finally:
            plt.close()
    
    def analyze_run_directory(self, run_dir: str, output_dir: str = None):
        """Main analysis function to create the comparison bar chart."""
        print("🚀 Starting MMLU comparison analysis...")
        
        # Create output directory - use run name from run_dir
        if output_dir is None:
            run_name = Path(run_dir).name  # e.g., "2025-08-17_03-57-04"
            output_dir = f"evaluation/evaluation_results/{run_name}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation data
        results = self.load_evaluation_data(run_dir)
        if not results:
            print("❌ No evaluation results found!")
            return
        
        # Find best checkpoints
        best_checkpoints = self.find_best_checkpoints(results)
        if len(best_checkpoints) < 2:
            print("❌ Need at least 2 run types for comparison!")
            return
        
        # Extract MMLU subjects
        subjects = self.extract_mmlu_subjects(best_checkpoints)
        if not subjects:
            print("❌ No MMLU subjects found!")
            return
        
        # Create comparison data
        df = self.create_comparison_data(best_checkpoints, subjects)
        if df.empty:
            print("❌ No comparison data created!")
            return
        
        # Create the comparison chart
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        chart_path = output_path / f"mmlu_comparison_bar_chart_{timestamp}.png"
        
        self.create_horizontal_bar_chart(df, str(chart_path))
        
        print(f"✅ Analysis complete! Results saved to: {output_path}")


def main():
    """Main function to run the comparison analysis."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_comparison_bar_chart.py <run_directory>")
        print("Example: python create_comparison_bar_chart.py logs/eval_pipeline/runs/2025-08-17_03-57-04")
        sys.exit(1)
    
    run_directory = sys.argv[1]
    
    # Create analyzer and run analysis
    analyzer = MMComparisonBarChart()
    analyzer.analyze_run_directory(run_directory)


if __name__ == "__main__":
    main()