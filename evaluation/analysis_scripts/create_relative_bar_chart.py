#!/usr/bin/env python3
"""
Create horizontal bar chart showing relative improvements vs baseline.
Based on create_comparison_bar_chart.py but shows percentage improvements instead of absolute scores.
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

class RelativeImprovementBarChart:
    def __init__(self):
        # Use modern, vibrant colors for improvement visualization
        self.run_colors = {
            'Clean Restart Improvement': '#4ECDC4',    # Turquoise for clean restart  
            'Green Run Improvement': '#95E77E',        # Light green for green run
        }
        
    def load_evaluation_data(self, run_dir: str) -> List[Dict]:
        """Load all evaluation JSON files from the run directory."""
        print(f"Loading evaluation data from: {run_dir}")
        
        results = []
        run_path = Path(run_dir)
        
        if not run_path.exists():
            print(f"Directory not found: {run_dir}")
            return results
        
        # Search for JSON files in evaluation subdirectories
        json_files = list(run_path.rglob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {run_dir}")
            return results
        
        print(f"Found {len(json_files)} JSON files")
        
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
                print(f"Error reading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(results)} evaluation results")
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
            r'step-(\\d+)',
            r'(\\d+)k',
            r'(\\d+)000',
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
        print("Finding best checkpoints for each run type...")
        
        run_types = ['baseline', 'clean_restart', 'green_run']
        best_checkpoints = {}
        
        for run_type in run_types:
            run_results = [r for r in results if r['run_type'] == run_type and 'mmlu_overall' in r]
            
            if not run_results:
                print(f"No results found for {run_type}")
                continue
            
            # Find best checkpoint based on MMLU overall score
            best_result = max(run_results, key=lambda x: x['mmlu_overall'])
            best_checkpoints[run_type] = best_result
            
            print(f"{run_type}: Step {best_result['step']}, MMLU: {best_result['mmlu_overall']:.4f}")
        
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
        print(f"Found {len(subjects_list)} MMLU subjects")
        
        return subjects_list
    
    def calculate_relative_improvements(self, best_checkpoints: Dict[str, Dict], subjects: List[str]) -> pd.DataFrame:
        """Calculate percentage improvements relative to baseline for all subjects."""
        if 'baseline' not in best_checkpoints:
            print("No baseline found for relative comparison!")
            return pd.DataFrame()
        
        baseline = best_checkpoints['baseline']
        improvement_data = []
        
        # Map run types to display names
        run_display_names = {
            'clean_restart': 'Clean Restart Improvement',
            'green_run': 'Green Run Improvement'
        }
        
        for subject in subjects:
            subject_key = f'mmlu_subject_{subject}'
            baseline_score = baseline.get(subject_key, 0)
            
            if baseline_score == 0:
                continue  # Skip subjects without baseline scores
            
            for run_type, display_name in run_display_names.items():
                if run_type in best_checkpoints:
                    checkpoint = best_checkpoints[run_type]
                    trained_score = checkpoint.get(subject_key, 0)
                    
                    if trained_score > 0:
                        # Calculate percentage improvement
                        improvement_pct = ((trained_score - baseline_score) / baseline_score) * 100
                        
                        improvement_data.append({
                            'Subject': subject.replace('_', ' ').title(),
                            'Run': display_name,
                            'Improvement_Pct': improvement_pct,
                            'Baseline_Score': baseline_score,
                            'Trained_Score': trained_score,
                            'Step': checkpoint.get('step', 0)
                        })
        
        df = pd.DataFrame(improvement_data)
        print(f"Created relative improvement data: {len(df)} rows")
        
        return df
    
    def create_relative_bar_chart(self, df: pd.DataFrame, save_path: str):
        """Create horizontal bar chart showing relative improvements vs baseline."""
        print("Creating relative improvement bar chart...")
        
        if df.empty:
            print("No data available for chart creation!")
            return
        
        # Calculate average improvements for sorting
        subject_averages = df.groupby('Subject')['Improvement_Pct'].mean().sort_values(ascending=True)
        sorted_subjects = subject_averages.index.tolist()
        
        # Set up the bar positions with spacing
        group_spacing = 1.8  # Space between task groups
        bar_height = 0.45  # Bar thickness
        bar_gap = 0.05  # Small gap between bars in same group
        
        # Create the plot with adjusted height
        total_height = len(sorted_subjects) * group_spacing * 0.3
        fig, ax = plt.subplots(figsize=(16, max(total_height, 20)))
        
        y_positions = np.arange(len(sorted_subjects)) * group_spacing
        
        # Create bars for each run type
        run_types = ['Clean Restart Improvement', 'Green Run Improvement']
        
        for i, run_type in enumerate(run_types):
            run_data = df[df['Run'] == run_type]
            
            improvements = []
            for subject in sorted_subjects:
                subject_data = run_data[run_data['Subject'] == subject]
                if not subject_data.empty:
                    improvements.append(subject_data['Improvement_Pct'].iloc[0])
                else:
                    improvements.append(0)  # Missing data
            
            # Plot bars with gap between them
            bar_offset = i * (bar_height + bar_gap)
            
            bars = ax.barh(y_positions + bar_offset, improvements, 
                          height=bar_height, 
                          label=run_type,
                          color=self.run_colors[run_type],
                          alpha=0.9,
                          edgecolor='black',
                          linewidth=0.5)
            
            # Add improvement percentage labels on bars
            for j, (bar, improvement) in enumerate(zip(bars, improvements)):
                if improvement != 0:
                    # Position label based on positive/negative improvement
                    label_x = bar.get_width() + (2 if improvement > 0 else -2)
                    label_align = 'left' if improvement > 0 else 'right'
                    
                    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                           f'{improvement:+.1f}%', 
                           ha=label_align, va='center', fontsize=7, fontweight='bold')
        
        # Customize the plot
        total_bars_height = 2 * bar_height + bar_gap
        center_offset = total_bars_height / 2 - bar_height / 2
        ax.set_yticks(y_positions + center_offset)
        ax.set_yticklabels(sorted_subjects, fontsize=10)
        ax.set_xlabel('Improvement over Baseline (%)', fontweight='bold', fontsize=14)
        ax.set_ylabel('MMLU Tasks (Sorted by Average Improvement)', fontweight='bold', fontsize=14)
        
        # Set title
        plt.title('MMLU Relative Improvements vs T5-Base Baseline\\n' + 
                 'Clean Restart vs Green Run (Percentage Changes)', 
                 fontweight='bold', fontsize=16, pad=20)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        
        # Set grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set x-axis limits based on data range
        max_improvement = df['Improvement_Pct'].max()
        min_improvement = df['Improvement_Pct'].min()
        x_range = max(abs(max_improvement), abs(min_improvement))
        ax.set_xlim(-x_range * 1.2, x_range * 1.2)
        
        # Add vertical reference lines
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
        ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=-10, color='gray', linestyle=':', alpha=0.5)
        
        # Add reference line labels
        max_y_pos = (len(sorted_subjects) - 1) * group_spacing + total_bars_height + bar_height
        ax.text(0, max_y_pos, 'Baseline\\n(0%)', ha='center', va='bottom', 
                fontsize=8, color='black', rotation=0, fontweight='bold')
        ax.text(10, max_y_pos, '+10%\\nImprovement', ha='center', va='bottom', 
                fontsize=8, color='gray', rotation=90)
        ax.text(-10, max_y_pos, '-10%\\nDecline', ha='center', va='bottom', 
                fontsize=8, color='gray', rotation=90)
        
        # Add colored background regions
        ax.axvspan(0, x_range * 1.2, alpha=0.1, color='green', label='Improvement')
        ax.axvspan(-x_range * 1.2, 0, alpha=0.1, color='red', label='Decline')
        
        # Adjust margins
        ax.margins(y=0.01)
        ax.set_ylim(-0.5, (len(sorted_subjects) - 1) * group_spacing + total_bars_height + 0.5)
        
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        # Save the plot
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                       edgecolor='none', format='png', transparent=False)
            print(f"Relative improvement bar chart saved to: {save_path}")
            
            # Also save summary statistics
            summary_path = save_path.replace('.png', '_summary.csv')
            summary_stats = df.groupby('Run')['Improvement_Pct'].agg(['mean', 'std', 'min', 'max', 'count'])
            summary_stats.to_csv(summary_path)
            print(f"Summary statistics saved to: {summary_path}")
            
            # Save detailed improvement data
            detailed_path = save_path.replace('.png', '_detailed.csv')
            df.to_csv(detailed_path, index=False)
            print(f"Detailed improvement data saved to: {detailed_path}")
            
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()
    
    def analyze_run_directory(self, run_dir: str, output_dir: str = None):
        """Main analysis function to create the relative improvement bar chart."""
        print("Starting MMLU relative improvement analysis...")
        
        # Create output directory - use run name from run_dir
        if output_dir is None:
            run_name = Path(run_dir).name  # e.g., "2025-08-17_03-57-04"
            output_dir = f"evaluation/evaluation_results/{run_name}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation data
        results = self.load_evaluation_data(run_dir)
        if not results:
            print("No evaluation results found!")
            return
        
        # Find best checkpoints
        best_checkpoints = self.find_best_checkpoints(results)
        if 'baseline' not in best_checkpoints:
            print("No baseline found for relative comparison!")
            return
        if len(best_checkpoints) < 2:
            print("Need at least baseline + 1 trained model for comparison!")
            return
        
        # Extract MMLU subjects
        subjects = self.extract_mmlu_subjects(best_checkpoints)
        if not subjects:
            print("No MMLU subjects found!")
            return
        
        # Calculate relative improvements
        df = self.calculate_relative_improvements(best_checkpoints, subjects)
        if df.empty:
            print("No relative improvement data created!")
            return
        
        # Create the relative improvement chart
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        chart_path = output_path / f"mmlu_relative_improvement_bar_chart_{timestamp}.png"
        
        self.create_relative_bar_chart(df, str(chart_path))
        
        print(f"Analysis complete! Results saved to: {output_path}")


def main():
    """Main function to run the relative improvement analysis."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_relative_bar_chart.py <run_directory>")
        print("Example: python create_relative_bar_chart.py logs/eval_pipeline/runs/2025-08-17_03-57-04")
        sys.exit(1)
    
    run_directory = sys.argv[1]
    
    # Create analyzer and run analysis
    analyzer = RelativeImprovementBarChart()
    analyzer.analyze_run_directory(run_directory)


if __name__ == "__main__":
    main()