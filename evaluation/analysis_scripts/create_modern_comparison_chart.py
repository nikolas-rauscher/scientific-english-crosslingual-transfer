#!/usr/bin/env python3
"""
Create a modern, interactive horizontal bar chart using Plotly.
Compares best checkpoints from Clean Restart, Green Run, and T5-Base baseline.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, List

class ModernComparisonChart:
    def __init__(self):
        # Modern color palette
        self.colors = {
            'T5-Base (Baseline)': '#E74C3C',     # Modern red
            'Clean Restart (Best)': '#3498DB',   # Modern blue
            'Green Run (Best)': '#2ECC71',       # Modern green
        }
        
    def load_evaluation_data(self, run_dir: str) -> List[Dict]:
        """Load all evaluation JSON files from the run directory."""
        print(f"🔍 Loading evaluation data from: {run_dir}")
        
        results = []
        run_path = Path(run_dir)
        
        if not run_path.exists():
            print(f"❌ Directory not found: {run_dir}")
            return results
        
        # Search for JSON files
        json_files = list(run_path.rglob("*.json"))
        
        if not json_files:
            print(f"⚠️  No JSON files found in {run_dir}")
            return results
        
        print(f"📄 Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
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
        """Parse individual evaluation result."""
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
        
        step_patterns = [r'step-(\d+)', r'(\d+)k', r'(\d+)000']
        
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
        else:
            return 'unknown'
    
    def find_best_checkpoints(self, results: List[Dict]) -> Dict[str, Dict]:
        """Find the best checkpoint for each run type."""
        print("🎯 Finding best checkpoints...")
        
        run_types = ['baseline', 'clean_restart', 'green_run']
        best_checkpoints = {}
        
        for run_type in run_types:
            run_results = [r for r in results if r['run_type'] == run_type and 'mmlu_overall' in r]
            
            if not run_results:
                print(f"⚠️  No results found for {run_type}")
                continue
            
            best_result = max(run_results, key=lambda x: x['mmlu_overall'])
            best_checkpoints[run_type] = best_result
            
            print(f"✅ {run_type}: Step {best_result['step']}, MMLU: {best_result['mmlu_overall']:.4f}")
        
        return best_checkpoints
    
    def create_plotly_chart(self, best_checkpoints: Dict[str, Dict], save_path: str):
        """Create modern interactive chart with Plotly."""
        print("🎨 Creating modern interactive chart...")
        
        # Extract all subjects
        all_subjects = set()
        for checkpoint_data in best_checkpoints.values():
            for key in checkpoint_data.keys():
                if key.startswith('mmlu_subject_'):
                    subject = key.replace('mmlu_subject_', '')
                    all_subjects.add(subject)
        
        subjects = sorted(list(all_subjects))
        
        # Prepare data for plotting
        data = []
        run_display_names = {
            'baseline': 'T5-Base (Baseline)',
            'clean_restart': 'Clean Restart (Best)',
            'green_run': 'Green Run (Best)'
        }
        
        # Calculate average scores for sorting
        subject_averages = {}
        for subject in subjects:
            scores = []
            for run_type in ['baseline', 'clean_restart', 'green_run']:
                if run_type in best_checkpoints:
                    score = best_checkpoints[run_type].get(f'mmlu_subject_{subject}', 0)
                    scores.append(score)
            subject_averages[subject] = np.mean(scores) if scores else 0
        
        # Sort subjects by average performance
        sorted_subjects = sorted(subjects, key=lambda x: subject_averages[x])
        
        # Create traces for each run type
        fig = go.Figure()
        
        for run_type, display_name in run_display_names.items():
            if run_type not in best_checkpoints:
                continue
                
            checkpoint = best_checkpoints[run_type]
            scores = []
            labels = []
            
            for subject in sorted_subjects:
                subject_key = f'mmlu_subject_{subject}'
                score = checkpoint.get(subject_key, 0)
                scores.append(score)
                labels.append(subject.replace('_', ' ').title())
            
            # Add trace
            fig.add_trace(go.Bar(
                name=display_name,
                y=labels,
                x=scores,
                orientation='h',
                marker=dict(
                    color=self.colors[display_name],
                    line=dict(color='white', width=1)
                ),
                text=[f'{s:.3f}' for s in scores],
                textposition='outside',
                textfont=dict(size=9),
                hovertemplate='<b>%{y}</b><br>' +
                             f'{display_name}: %{{x:.3f}}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout for modern look
        fig.update_layout(
            title={
                'text': 'MMLU Performance Comparison: Best Checkpoints vs Baseline',
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='MMLU Accuracy Score',
                range=[0, 0.6],
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.2)',
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title='MMLU Tasks (Sorted by Average Performance)',
                automargin=True,
                tickfont=dict(size=10)
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(len(sorted_subjects) * 25, 800),
            width=1200,
            margin=dict(l=200, r=100, t=100, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            hovermode='y unified'
        )
        
        # Add reference lines
        fig.add_vline(x=0.25, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="Random Chance", annotation_position="top")
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="50% Accuracy", annotation_position="top")
        
        # Save as interactive HTML
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"📊 Interactive chart saved to: {html_path}")
        
        # Also save as static image (requires kaleido)
        try:
            fig.write_image(save_path, width=1200, height=max(len(sorted_subjects) * 25, 800))
            print(f"📊 Static image saved to: {save_path}")
        except:
            print("⚠️  Could not save static image (install kaleido: pip install kaleido)")
        
        # Save summary statistics
        summary_data = []
        for run_type, display_name in run_display_names.items():
            if run_type in best_checkpoints:
                checkpoint = best_checkpoints[run_type]
                scores = [checkpoint.get(f'mmlu_subject_{s}', 0) for s in subjects]
                summary_data.append({
                    'Run': display_name,
                    'Mean': np.mean(scores),
                    'Std': np.std(scores),
                    'Min': np.min(scores),
                    'Max': np.max(scores),
                    'Step': checkpoint.get('step', 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = save_path.replace('.png', '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"📋 Summary saved to: {summary_path}")
    
    def analyze(self, run_dir: str, output_dir: str = None):
        """Main analysis function."""
        print("🚀 Starting modern MMLU comparison analysis...")
        
        # Create output directory - use run name from run_dir
        if output_dir is None:
            run_name = Path(run_dir).name  # e.g., "2025-08-17_03-57-04"
            output_dir = f"evaluation/evaluation_results/{run_name}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        results = self.load_evaluation_data(run_dir)
        if not results:
            print("❌ No evaluation results found!")
            return
        
        # Find best checkpoints
        best_checkpoints = self.find_best_checkpoints(results)
        if len(best_checkpoints) < 2:
            print("❌ Need at least 2 run types for comparison!")
            return
        
        # Create chart
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        chart_path = output_path / f"modern_mmlu_comparison_{timestamp}.png"
        
        self.create_plotly_chart(best_checkpoints, str(chart_path))
        
        print(f"✅ Analysis complete! Check the interactive HTML file!")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_modern_comparison_chart.py <run_directory>")
        sys.exit(1)
    
    run_directory = sys.argv[1]
    
    analyzer = ModernComparisonChart()
    analyzer.analyze(run_directory)


if __name__ == "__main__":
    main()