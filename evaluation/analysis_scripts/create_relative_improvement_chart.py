#!/usr/bin/env python3
"""
Create modern interactive horizontal bar chart showing relative improvements vs baseline.
Shows percentage improvements of Clean Restart and Green Run compared to T5-Base baseline.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, List

class RelativeImprovementChart:
    def __init__(self):
        # Modern color palette for improvements
        self.colors = {
            'Clean Restart Improvement': '#3498DB',   # Modern blue
            'Green Run Improvement': '#2ECC71',       # Modern green
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
                
            # Extract MMLU category scores for global format
            category_mappings = {
                'humanities': 'global_mmlu_full_en_humanities',
                'social_sciences': 'global_mmlu_full_en_social_sciences',
                'stem': 'global_mmlu_full_en_stem',
                'other': 'global_mmlu_full_en_other'
            }
            for category, global_key in category_mappings.items():
                if global_key in results_data and 'acc,none' in results_data[global_key]:
                    result[f'mmlu_category_{category}'] = results_data[global_key]['acc,none']
                    
            # Extract individual subject scores for global MMLU
            for key, value in results_data.items():
                if key.startswith('global_mmlu_full_en_') and isinstance(value, dict):
                    # Skip category and main keys
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
                
            # Extract MMLU category scores (directly from JSON)
            category_keys = ['humanities', 'stem', 'social_sciences', 'other']
            for category in category_keys:
                if category in results_data and 'acc,none' in results_data[category]:
                    result[f'mmlu_category_{category}'] = results_data[category]['acc,none']
                    
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
    
    def calculate_relative_improvements(self, best_checkpoints: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate percentage improvements relative to baseline."""
        print("📊 Calculating relative improvements...")
        
        if 'baseline' not in best_checkpoints:
            print("❌ No baseline found for comparison!")
            return {}
        
        baseline = best_checkpoints['baseline']
        improvements = {}
        
        # Extract all subjects
        all_subjects = set()
        for checkpoint_data in best_checkpoints.values():
            for key in checkpoint_data.keys():
                if key.startswith('mmlu_subject_'):
                    subject = key.replace('mmlu_subject_', '')
                    all_subjects.add(subject)
        
        subjects = sorted(list(all_subjects))
        
        # Calculate improvements for each trained model
        for run_type in ['clean_restart', 'green_run']:
            if run_type not in best_checkpoints:
                continue
                
            trained_model = best_checkpoints[run_type]
            run_improvements = {}
            
            # Individual subjects
            for subject in subjects:
                subject_key = f'mmlu_subject_{subject}'
                
                baseline_score = baseline.get(subject_key, 0)
                trained_score = trained_model.get(subject_key, 0)
                
                if baseline_score > 0:
                    # Calculate percentage improvement
                    improvement = ((trained_score - baseline_score) / baseline_score) * 100
                    run_improvements[subject] = {
                        'baseline_score': baseline_score,
                        'trained_score': trained_score,
                        'improvement_pct': improvement
                    }
                else:
                    run_improvements[subject] = {
                        'baseline_score': 0,
                        'trained_score': trained_score,
                        'improvement_pct': 0
                    }
            
            # Add MMLU overall score
            baseline_overall = baseline.get('mmlu_overall', 0)
            trained_overall = trained_model.get('mmlu_overall', 0)
            
            if baseline_overall > 0:
                overall_improvement = ((trained_overall - baseline_overall) / baseline_overall) * 100
                run_improvements['MMLU_Overall'] = {
                    'baseline_score': baseline_overall,
                    'trained_score': trained_overall,
                    'improvement_pct': overall_improvement
                }
            
            # Add MMLU category scores (directly from JSON files)
            category_mapping = {
                'mmlu_category_humanities': 'Humanities',
                'mmlu_category_stem': 'STEM', 
                'mmlu_category_social_sciences': 'Social_Sciences',
                'mmlu_category_other': 'Other'
            }
            
            for category_key, display_name in category_mapping.items():
                baseline_score = baseline.get(category_key, 0)
                trained_score = trained_model.get(category_key, 0)
                
                if baseline_score > 0:
                    improvement = ((trained_score - baseline_score) / baseline_score) * 100
                    run_improvements[f'{display_name}_Average'] = {
                        'baseline_score': baseline_score,
                        'trained_score': trained_score,
                        'improvement_pct': improvement
                    }
            
            improvements[run_type] = run_improvements
            
            # Calculate average improvement
            individual_improvements = [data['improvement_pct'] for key, data in run_improvements.items() 
                                     if not key.endswith('_Average') and not key.startswith('MMLU_')]
            avg_improvement = np.mean(individual_improvements) if individual_improvements else 0
            print(f"✅ {run_type}: Average improvement {avg_improvement:.1f}%")
        
        return improvements
    
    def create_improvement_chart(self, improvements: Dict[str, Dict], save_path: str):
        """Create two separate charts: grouped categories and individual subjects."""
        print("🎨 Creating relative improvement charts...")
        
        if not improvements:
            print("❌ No improvement data available!")
            return
        
        # Extract subjects and separate them
        all_subjects = set()
        for run_data in improvements.values():
            all_subjects.update(run_data.keys())
        
        subjects = sorted(list(all_subjects))
        
        # Separate grouped metrics from individual subjects  
        grouped_metrics = [s for s in subjects if s.endswith('_Average') or s.startswith('MMLU_')]
        individual_subjects = [s for s in subjects if s not in grouped_metrics]
        
        # Sort grouped metrics by preference order
        metric_order = ['MMLU_Overall', 'STEM_Average', 'Humanities_Average', 'Social_Sciences_Average', 'Other_Average']
        sorted_grouped_metrics = [m for m in metric_order if m in grouped_metrics]
        sorted_grouped_metrics.extend([m for m in grouped_metrics if m not in metric_order])
        
        # Calculate average improvements for sorting individual subjects
        subject_avg_improvements = {}
        for subject in individual_subjects:
            improvements_list = []
            for run_type in improvements:
                if subject in improvements[run_type]:
                    improvements_list.append(improvements[run_type][subject]['improvement_pct'])
            subject_avg_improvements[subject] = np.mean(improvements_list) if improvements_list else 0
        
        # Sort individual subjects by average improvement (worst to best)
        sorted_individual_subjects = sorted(individual_subjects, key=lambda x: subject_avg_improvements[x])
        
        # Create grouped categories chart
        grouped_chart_path = save_path.replace('.png', '_grouped_categories.png')
        self.create_single_chart(improvements, sorted_grouped_metrics, grouped_chart_path, 
                                'MMLU Category Performance vs Baseline', 'grouped')
        
        # Create individual subjects chart
        individual_chart_path = save_path.replace('.png', '_individual_subjects.png')  
        self.create_single_chart(improvements, sorted_individual_subjects, individual_chart_path,
                                'Individual MMLU Tasks vs Baseline', 'individual')
    
    def create_single_chart(self, improvements: Dict[str, Dict], sorted_subjects: List[str], 
                           save_path: str, title: str, chart_type: str):
        """Create a single chart for either grouped or individual metrics."""
        
        # Create figure
        fig = go.Figure()
        
        run_display_names = {
            'clean_restart': 'Clean Restart Improvement',
            'green_run': 'Green Run Improvement'
        }
        
        # Determine chart height based on number of subjects
        chart_height = max(len(sorted_subjects) * 35, 400)
        if chart_type == 'grouped':
            chart_height = max(len(sorted_subjects) * 60, 400)  # More space for grouped metrics
        
        for run_type, display_name in run_display_names.items():
            if run_type not in improvements:
                continue
                
            improvement_pcts = []
            labels = []
            hover_texts = []
            
            for subject in sorted_subjects:
                if subject in improvements[run_type]:
                    data = improvements[run_type][subject]
                    improvement_pct = data['improvement_pct']
                    baseline_score = data['baseline_score']
                    trained_score = data['trained_score']
                    
                    improvement_pcts.append(improvement_pct)
                    
                    # Format label based on chart type
                    if chart_type == 'grouped':
                        if subject == 'MMLU_Overall':
                            labels.append('MMLU Overall')
                        elif subject == 'STEM_Average':
                            labels.append('STEM Average')
                        elif subject == 'Humanities_Average':
                            labels.append('Humanities Average')
                        elif subject == 'Social_Sciences_Average':
                            labels.append('Social Sciences Average')
                        elif subject == 'Other_Average':
                            labels.append('Other Average')
                        else:
                            labels.append(subject.replace('_', ' ').title())
                    else:
                        labels.append(subject.replace('_', ' ').title())
                    
                    # Create detailed hover text
                    hover_text = (f"<b>{labels[-1]}</b><br>"
                                f"Baseline: {baseline_score:.3f}<br>"
                                f"Trained: {trained_score:.3f}<br>"
                                f"Improvement: {improvement_pct:+.1f}%")
                    hover_texts.append(hover_text)
                else:
                    improvement_pcts.append(0)
                    labels.append(subject.replace('_', ' ').title())
                    hover_texts.append(f"<b>{labels[-1]}</b><br>No data")
            
            # Add trace
            fig.add_trace(go.Bar(
                name=display_name,
                y=labels,
                x=improvement_pcts,
                orientation='h',
                marker=dict(
                    color=self.colors[display_name],
                    line=dict(color='white', width=1)
                ),
                text=[f'{x:+.1f}%' for x in improvement_pcts],
                textposition='outside',
                textfont=dict(size=9),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 18, 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Improvement over Baseline (%)',
                range=[-70, 220],
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.4)',
                zerolinewidth=2,
                tickfont=dict(size=11),
                ticksuffix='%'
            ),
            yaxis=dict(
                title='Tasks (Sorted by Performance)' if chart_type == 'individual' else 'MMLU Categories',
                automargin=True,
                tickfont=dict(size=10)
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=chart_height,
            width=1200,
            margin=dict(l=250, r=100, t=80, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            hovermode='y unified'
        )
        
        # Add reference line and regions
        fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.3, line_width=2,
                     annotation_text="Baseline (0%)", annotation_position="top")
        fig.add_vrect(x0=-70, x1=0, fillcolor="red", opacity=0.1, layer="below", line_width=0)
        fig.add_vrect(x0=0, x1=220, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        
        # Save files
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"📊 {chart_type.title()} chart saved to: {html_path}")
        
        try:
            fig.write_image(save_path, width=1200, height=chart_height)
            print(f"📊 Static {chart_type} image saved to: {save_path}")
        except:
            print("⚠️  Could not save static image (install kaleido: pip install kaleido)")
        
        # Save CSV data
        improvement_data = []
        for subject in sorted_subjects:
            row = {'Subject': subject.replace('_', ' ').title()}
            
            for run_type, display_name in run_display_names.items():
                if run_type in improvements and subject in improvements[run_type]:
                    data = improvements[run_type][subject]
                    row[f'{display_name} (%)'] = data['improvement_pct']
                    row[f'{display_name} Baseline'] = data['baseline_score']
                    row[f'{display_name} Trained'] = data['trained_score']
                else:
                    row[f'{display_name} (%)'] = 0
                    row[f'{display_name} Baseline'] = 0
                    row[f'{display_name} Trained'] = 0
            
            improvement_data.append(row)
        
        improvement_df = pd.DataFrame(improvement_data)
        csv_path = save_path.replace('.png', f'_{chart_type}_improvements.csv')
        improvement_df.to_csv(csv_path, index=False)
        print(f"📋 {chart_type.title()} data saved to: {csv_path}")
    
    def analyze(self, run_dir: str, output_dir: str = None):
        """Main analysis function."""
        print("🚀 Starting relative improvement analysis...")
        
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
        if 'baseline' not in best_checkpoints:
            print("❌ No baseline found for relative comparison!")
            return
        
        # Calculate improvements
        improvements = self.calculate_relative_improvements(best_checkpoints)
        if not improvements:
            print("❌ Could not calculate improvements!")
            return
        
        # Create chart
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        chart_path = output_path / f"relative_improvement_chart_{timestamp}.png"
        
        self.create_improvement_chart(improvements, str(chart_path))
        
        print(f"✅ Relative improvement analysis complete!")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_relative_improvement_chart.py <run_directory>")
        sys.exit(1)
    
    run_directory = sys.argv[1]
    
    analyzer = RelativeImprovementChart()
    analyzer.analyze(run_directory)


if __name__ == "__main__":
    main()