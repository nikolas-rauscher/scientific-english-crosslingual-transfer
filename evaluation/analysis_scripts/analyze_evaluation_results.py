#!/usr/bin/env python3
"""
Comprehensive Evaluation Results Analysis Script

This script analyzes all evaluation results from the eval_pipeline runs,
creates comparison tables, and generates performance plots.

Usage:
    python analyze_evaluation_results.py [run_directory]
    python analyze_evaluation_results.py --all  # Analyze all recent runs
"""

import json
import os
import re
import pandas as pd
import os
# Fix matplotlib display issues on cluster
os.environ['DISPLAY'] = ''  # Disable display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
# Set style and improve default settings
plt.style.use('default')
sns.set_palette("husl")
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

class EvaluationAnalyzer:
    def __init__(self, results_base_dir: str = "logs/eval_pipeline/runs"):
        self.results_base_dir = Path(results_base_dir)
        self.results = []
        
    def extract_model_info(self, model_name: str) -> Dict:
        """Extract step, PPL, and run type from model name."""
        # Pattern examples:
        # clean_restart_best06_52k_ppl1419
        # flan_t5_step01_25k
        # green_run_best95_382k_ppl136215
        
        patterns = [
            # Pattern 1: best checkpoints with PPL
            r'(\w+(?:_\w+)*)_best(\d+)_(\d+)k_ppl(\d+(?:\d{2,})?)',
            # Pattern 2: step checkpoints 
            r'(\w+(?:_\w+)*)_step(\d+)_(\d+)k',
            # Pattern 3: baseline models
            r'(t5_base|flan_t5_base)_original',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                if len(match.groups()) == 4:  # best checkpoint with PPL
                    run_type, checkpoint_num, step_k, ppl = match.groups()
                    return {
                        'run_type': run_type,
                        'checkpoint_type': 'best',
                        'checkpoint_num': int(checkpoint_num),
                        'step': int(step_k) * 1000,
                        'ppl': float(ppl) / 100 if len(ppl) > 2 else float(ppl),  # Handle PPL format
                        'is_baseline': False
                    }
                elif len(match.groups()) == 3:  # step checkpoint
                    run_type, checkpoint_num, step_k = match.groups()
                    return {
                        'run_type': run_type,
                        'checkpoint_type': 'step',
                        'checkpoint_num': int(checkpoint_num),
                        'step': int(step_k) * 1000,
                        'ppl': None,
                        'is_baseline': False
                    }
                elif len(match.groups()) == 1:  # baseline model
                    model_type = match.groups()[0]
                    return {
                        'run_type': 'baseline',
                        'checkpoint_type': 'baseline',
                        'checkpoint_num': 0,
                        'step': 0,
                        'ppl': None,
                        'is_baseline': True,
                        'baseline_type': model_type
                    }
        
        # Fallback for unknown patterns
        return {
            'run_type': 'unknown',
            'checkpoint_type': 'unknown',
            'checkpoint_num': 0,
            'step': 0,
            'ppl': None,
            'is_baseline': False
        }
    
    def extract_mmlu_scores(self, results_data: Dict) -> Dict:
        """Extract MMLU scores from results JSON (supports both FLAN and Global MMLU formats)."""
        scores = {}
        
        if 'results' in results_data:
            results = results_data['results']
            
            # Main MMLU score - support both formats
            if 'global_mmlu_full_en' in results:
                # Global MMLU format
                scores['mmlu_overall'] = results['global_mmlu_full_en']['acc,none']
                scores['mmlu_overall_stderr'] = results['global_mmlu_full_en']['acc_stderr,none']
                
                # Category scores for global MMLU
                category_mappings = {
                    'humanities': 'global_mmlu_full_en_humanities',
                    'social_sciences': 'global_mmlu_full_en_social_sciences', 
                    'stem': 'global_mmlu_full_en_stem',
                    'other': 'global_mmlu_full_en_other'
                }
                
                for category_name, global_key in category_mappings.items():
                    if global_key in results:
                        scores[f'mmlu_{category_name}'] = results[global_key]['acc,none']
                        scores[f'mmlu_{category_name}_stderr'] = results[global_key]['acc_stderr,none']
                
                # Individual subject scores for global MMLU
                for key, value in results.items():
                    if key.startswith('global_mmlu_full_en_') and isinstance(value, dict):
                        # Skip category keys
                        if key not in ['global_mmlu_full_en', 'global_mmlu_full_en_humanities', 
                                      'global_mmlu_full_en_social_sciences', 'global_mmlu_full_en_stem', 
                                      'global_mmlu_full_en_other']:
                            # Extract subject name
                            subject = key.replace('global_mmlu_full_en_', '')
                            if 'acc,none' in value:
                                scores[f'mmlu_subject_{subject}'] = value['acc,none']
                                if 'acc_stderr,none' in value:
                                    scores[f'mmlu_subject_{subject}_stderr'] = value['acc_stderr,none']
                                    
            elif 'mmlu_flan_n_shot_loglikelihood' in results:
                # Original FLAN format (fallback for backwards compatibility)
                scores['mmlu_overall'] = results['mmlu_flan_n_shot_loglikelihood']['acc,none']
                scores['mmlu_overall_stderr'] = results['mmlu_flan_n_shot_loglikelihood']['acc_stderr,none']
                
                # Category scores for FLAN format
                categories = ['humanities', 'social sciences', 'stem', 'other']
                for category in categories:
                    if category in results:
                        scores[f'mmlu_{category.replace(" ", "_")}'] = results[category]['acc,none']
                        scores[f'mmlu_{category.replace(" ", "_")}_stderr'] = results[category]['acc_stderr,none']
                
                # Individual MMLU subject scores for FLAN format
                for key, value in results.items():
                    if key.startswith('mmlu_flan_n_shot_loglikelihood_') and isinstance(value, dict):
                        subject = key.replace('mmlu_flan_n_shot_loglikelihood_', '')
                        if 'acc,none' in value:
                            scores[f'mmlu_subject_{subject}'] = value['acc,none']
                            if 'acc_stderr,none' in value:
                                scores[f'mmlu_subject_{subject}_stderr'] = value['acc_stderr,none']
        
        return scores
    
    def scan_evaluation_results(self, run_dir: Optional[str] = None, experiment_name: Optional[str] = None) -> List[Dict]:
        """Scan for all evaluation results in the specified directory."""
        results = []
        
        if run_dir:
            # Scan specific run directory
            if run_dir.startswith('logs/eval_pipeline/runs/'):
                # Full path provided
                run_dirs = [Path(run_dir)]
            else:
                # Just directory name provided
                run_dirs = [self.results_base_dir / run_dir]
        else:
            print("❌ Please specify a run directory!")
            print("Usage examples:")
            print("  python analyze_evaluation_results.py logs/eval_pipeline/runs/2025-08-17_04-01-04")
            print("  python analyze_evaluation_results.py 2025-08-17_04-01-04")
            return []
        
        for run_dir_path in run_dirs:
            base_universal = run_dir_path / "evaluation" / "results" / "universal"
            eval_dir = None
            
            # Prefer explicit experiment name when provided
            if experiment_name:
                candidate = base_universal / experiment_name
                if candidate.exists():
                    eval_dir = candidate
                else:
                    print(f"⚠️  Experiment '{experiment_name}' not found under {base_universal}")
            
            # Auto-detect experiment subdir when not provided
            if eval_dir is None:
                if not base_universal.exists():
                    # Nothing to analyze for this run_dir
                    continue
                subdirs = [d for d in base_universal.iterdir() if d.is_dir()]
                if not subdirs:
                    continue
                # Prefer legacy 'universal_evaluation' if present
                legacy = [d for d in subdirs if d.name == 'universal_evaluation']
                if legacy:
                    eval_dir = legacy[0]
                else:
                    # Pick the most recently modified subdir
                    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                    eval_dir = subdirs[0]
            
            if not eval_dir.exists():
                continue
                
            print(f"📂 Scanning {run_dir_path.name}...")
            
            for model_dir in eval_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name.split('_mmlu_')[0]  # Remove suffix
                model_info = self.extract_model_info(model_name)
                
                # Find results JSON file
                json_files = list(model_dir.rglob("results_*.json"))
                if not json_files:
                    continue
                
                json_file = json_files[0]  # Take first (should be only one)
                
                try:
                    with open(json_file, 'r') as f:
                        results_data = json.load(f)
                    
                    mmlu_scores = self.extract_mmlu_scores(results_data)
                    
                    result_entry = {
                        'run_directory': run_dir_path.name,
                        'model_name': model_name,
                        'json_file': str(json_file),
                        **model_info,
                        **mmlu_scores
                    }
                    
                    results.append(result_entry)
                    
                except Exception as e:
                    print(f"⚠️  Error processing {json_file}: {e}")
        
        return results
    
    def create_comparison_table(self, results: List[Dict]) -> pd.DataFrame:
        """Create a comprehensive comparison table."""
        df = pd.DataFrame(results)
        
        if df.empty:
            return df
        
        # Sort by run_type, then by step
        df = df.sort_values(['run_type', 'step'], ascending=[True, True])
        
        # Select key columns for display
        display_columns = [
            'run_type', 'checkpoint_type', 'step', 'ppl', 'mmlu_overall',
            'mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other'
        ]
        
        # Filter to available columns
        available_columns = [col for col in display_columns if col in df.columns]
        display_df = df[available_columns].copy()
        
        # Format numbers for better readability (5 decimal places for precision)
        if 'mmlu_overall' in display_df.columns:
            display_df['mmlu_overall'] = display_df['mmlu_overall'].round(5)
        if 'ppl' in display_df.columns:
            # Convert to numeric first, handling any non-numeric values
            display_df['ppl'] = pd.to_numeric(display_df['ppl'], errors='coerce')
            display_df['ppl'] = display_df['ppl'].round(5)
        
        # Round other MMLU categories to 5 decimal places
        mmlu_columns = ['mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other']
        for col in mmlu_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(5)
            
        return display_df
    
    def create_performance_plot(self, results: List[Dict], save_path: str = "evaluation_performance_analysis.png"):
        """Create performance plots showing MMLU progression."""
        df = pd.DataFrame(results)
        
        if df.empty or 'mmlu_overall' not in df.columns:
            print("⚠️  No MMLU data available for plotting")
            return
        
        # Filter out baseline models for main plot
        df_checkpoints = df[~df['is_baseline']].copy()
        
        # Set up the plot with improved size and quality - now with 6 subplots
        plt.figure(figsize=(24, 18))
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        
        # Define improved colors and markers for each run type
        run_styles = {
            'clean_restart': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'alpha': 0.8},  # Blue
            'green_run': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'alpha': 0.8},     # Orange  
            'flan_t5': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-', 'alpha': 0.8},      # Green
            'flan': {'color': '#d62728', 'marker': 'D', 'linestyle': '-', 'alpha': 0.8},         # Red
        }
        
        # Plot 1: MMLU Performance Bar Chart (Top Performing Checkpoints)
        plt.subplot(3, 3, 1)
        
        # Get top 15 checkpoints across all runs for better visualization
        top_checkpoints = df_checkpoints.nlargest(15, 'mmlu_overall').copy()
        
        # Create grouped data for bar chart
        run_types = top_checkpoints['run_type'].unique()
        x_positions = np.arange(len(top_checkpoints))
        bar_width = 0.8
        
        # Add baseline reference line if available
        baseline_mmlu = None
        baseline_data = df[df['is_baseline'] == True]
        if not baseline_data.empty and 'mmlu_overall' in baseline_data.columns:
            baseline_mmlu = baseline_data['mmlu_overall'].mean()
            plt.axhline(y=baseline_mmlu, color='gray', linestyle='--', alpha=0.7, linewidth=2,
                       label=f'T5-Base Baseline ({baseline_mmlu:.5f})')
        
        # Create bar chart with different colors for each run type
        bars = []
        for i, (_, row) in enumerate(top_checkpoints.iterrows()):
            run_type = row['run_type']
            style = run_styles.get(run_type, {'color': '#333333'})
            
            # Different opacity for best vs step checkpoints
            alpha = 0.9 if row['checkpoint_type'] == 'best' else 0.6
            
            bar = plt.bar(i, row['mmlu_overall'], bar_width, 
                         color=style['color'], alpha=alpha, 
                         edgecolor='white', linewidth=1)
            bars.append(bar)
        
        # Customize x-axis labels (show step numbers)
        step_labels = [f"{int(row['step']/1000)}k\n({row['run_type'][:4]})" 
                      for _, row in top_checkpoints.iterrows()]
        plt.xticks(x_positions, step_labels, rotation=45, ha='right', fontsize=9)
        
        plt.xlabel('Training Steps (Run Type)')
        plt.ylabel('MMLU Overall Accuracy')
        plt.title('Top 15 MMLU Performance (All Runs)', fontweight='bold')
        
        # Create custom legend for run types
        legend_elements = []
        for run_type in run_types:
            if run_type != 'unknown':
                style = run_styles.get(run_type, {'color': '#333333'})
                legend_elements.append(plt.Rectangle((0,0),1,1, 
                                     color=style['color'], alpha=0.8, 
                                     label=run_type.replace('_', ' ').title()))
        if baseline_mmlu:
            legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                            label='T5-Base Baseline'))
        
        plt.legend(handles=legend_elements, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.5f}'))
        
        # Plot 2: Best Performance Summary
        plt.subplot(3, 3, 2)
        # Find best performance for each run
        best_performances = []
        for run_type in df_checkpoints['run_type'].unique():
            if run_type == 'unknown':
                continue
            run_data = df_checkpoints[df_checkpoints['run_type'] == run_type]
            if not run_data.empty and 'mmlu_overall' in run_data.columns:
                best_idx = run_data['mmlu_overall'].idxmax()
                best_performances.append({
                    'run_type': run_type,
                    'best_mmlu': run_data.loc[best_idx, 'mmlu_overall'],
                    'best_step': run_data.loc[best_idx, 'step'],
                    'best_ppl': run_data.loc[best_idx, 'ppl'] if 'ppl' in run_data.columns else None
                })
        
        if best_performances:
            best_df = pd.DataFrame(best_performances)
            colors = [run_styles.get(rt, {'color': '#333333'})['color'] for rt in best_df['run_type']]
            bars = plt.bar(range(len(best_df)), best_df['best_mmlu'], 
                          color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            
            plt.xticks(range(len(best_df)), 
                      [rt.replace('_', ' ').title() for rt in best_df['run_type']], rotation=45, ha='right')
            plt.ylabel('Best MMLU Accuracy')
            plt.title('Best Performance Comparison', fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            
            # Add value labels on bars with 5 decimal places
            for i, (bar, val) in enumerate(zip(bars, best_df['best_mmlu'])):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                        f'{val:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Format y-axis to show 5 decimal places
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.5f}'))
        
        # Plot 3: Training Progression Over Steps (Line Chart)
        plt.subplot(3, 3, 3)
        
        # Add baseline reference line if available
        baseline_mmlu = None
        baseline_data = df[df['is_baseline'] == True]
        if not baseline_data.empty and 'mmlu_overall' in baseline_data.columns:
            baseline_mmlu = baseline_data['mmlu_overall'].mean()
            plt.axhline(y=baseline_mmlu, color='gray', linestyle='--', alpha=0.7, linewidth=2,
                       label=f'T5-Base Baseline ({baseline_mmlu:.5f})')
        
        # Plot progression for each run type
        for run_type in df_checkpoints['run_type'].unique():
            if run_type == 'unknown':
                continue
            
            run_data = df_checkpoints[df_checkpoints['run_type'] == run_type].copy()
            run_data = run_data.sort_values('step')
            
            style = run_styles.get(run_type, {'color': '#333333', 'marker': 'o', 'linestyle': '-', 'alpha': 0.8})
            
            # Plot all checkpoints (both best and step) to show full progression
            if not run_data.empty:
                plt.plot(run_data['step'], run_data['mmlu_overall'], 
                        color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                        label=f'{run_type.replace("_", " ").title()}', 
                        markersize=4, linewidth=2, alpha=style['alpha'], markeredgewidth=0.5,
                        markeredgecolor='white')
        
        plt.xlabel('Training Steps')
        plt.ylabel('MMLU Overall Accuracy')
        plt.title('Training Progression Over Steps', fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.5f}'))
        
        # Plot 4: PPL vs MMLU Correlation
        plt.subplot(3, 3, 4)
        df_with_ppl = df_checkpoints.dropna(subset=['ppl', 'mmlu_overall'])
        if not df_with_ppl.empty:
            for run_type in df_with_ppl['run_type'].unique():
                if run_type == 'unknown':
                    continue
                run_data = df_with_ppl[df_with_ppl['run_type'] == run_type]
                style = run_styles.get(run_type, {'color': '#333333'})
                plt.scatter(run_data['ppl'], run_data['mmlu_overall'], 
                           color=style['color'], alpha=0.7, label=run_type.replace('_', ' ').title(),
                           s=40, edgecolors='white', linewidth=0.5)
            
            plt.xlabel('Validation Perplexity')
            plt.ylabel('MMLU Overall Accuracy')
            plt.title('Perplexity vs MMLU Performance', fontweight='bold')
            plt.legend(frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.5f}'))
        
        # Plot 5: MMLU Category Performance Bar Chart (4 main categories)
        plt.subplot(3, 3, 5)
        categories = ['mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other']
        available_categories = [cat for cat in categories if cat in df.columns]
        
        if available_categories and not df_checkpoints.empty:
            # Get best checkpoint for each run for comparison
            category_data = []
            run_performance = {}
            
            for run_type in df_checkpoints['run_type'].unique():
                if run_type == 'unknown':
                    continue
                run_data = df_checkpoints[df_checkpoints['run_type'] == run_type]
                if not run_data.empty:
                    best_idx = run_data['mmlu_overall'].idxmax()
                    best_row = run_data.loc[best_idx]
                    run_performance[run_type] = {}
                    
                    for cat in available_categories:
                        if pd.notna(best_row[cat]):
                            category_name = cat.replace('mmlu_', '').replace('_', ' ').title()
                            run_performance[run_type][category_name] = best_row[cat]
                            category_data.append({
                                'run_type': run_type,
                                'category': category_name,
                                'accuracy': best_row[cat]
                            })
            
            if category_data:
                # Create grouped bar chart
                cat_df = pd.DataFrame(category_data)
                pivot_df = cat_df.pivot(index='category', columns='run_type', values='accuracy')
                
                # Set up bar positions
                categories_list = pivot_df.index.tolist()
                run_types_list = pivot_df.columns.tolist()
                x = np.arange(len(categories_list))
                bar_width = 0.25
                
                # Create bars for each run type
                for i, run_type in enumerate(run_types_list):
                    if run_type != 'unknown':
                        style = run_styles.get(run_type, {'color': '#333333'})
                        values = [pivot_df.loc[cat, run_type] if pd.notna(pivot_df.loc[cat, run_type]) else 0 
                                 for cat in categories_list]
                        
                        bars = plt.bar(x + i * bar_width, values, bar_width, 
                                      color=style['color'], alpha=0.8, 
                                      label=run_type.replace('_', ' ').title(),
                                      edgecolor='white', linewidth=1)
                        
                        # Add value labels on top of bars
                        for j, (bar, val) in enumerate(zip(bars, values)):
                            if val > 0:
                                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.xlabel('MMLU Categories')
                plt.ylabel('Accuracy (Best Checkpoints)')
                plt.title('MMLU Category Performance', fontweight='bold')
                plt.xticks(x + bar_width, categories_list, rotation=45, ha='right')
                plt.legend(frameon=True, fancybox=True, shadow=True)
                plt.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        
        # Plot 6: All Individual MMLU Subjects Performance (Big Chart)
        plt.subplot(3, 1, 3)  # Take full bottom row
        
        # Extract all individual MMLU subjects from the data
        all_subjects = set()
        for result in self.results:
            for key in result.keys():
                if key.startswith('mmlu_subject_'):
                    subject = key.replace('mmlu_subject_', '').replace('_stderr', '')
                    all_subjects.add(subject)
        
        if all_subjects and not df_checkpoints.empty:
            # Sort subjects alphabetically
            subjects_list = sorted(list(all_subjects))
            
            # Get best checkpoint for each run
            subject_data = []
            for run_type in df_checkpoints['run_type'].unique():
                if run_type == 'unknown':
                    continue
                run_data = df_checkpoints[df_checkpoints['run_type'] == run_type]
                if not run_data.empty:
                    best_idx = run_data['mmlu_overall'].idxmax()
                    best_row = run_data.loc[best_idx]
                    
                    for subject in subjects_list:
                        subject_key = f'mmlu_subject_{subject}'
                        if subject_key in best_row and pd.notna(best_row[subject_key]):
                            # Clean up subject name for display
                            clean_subject = subject.replace('_', ' ').title()
                            subject_data.append({
                                'run_type': run_type,
                                'subject': clean_subject,
                                'accuracy': best_row[subject_key]
                            })
            
            if subject_data:
                # Create pivot table for grouped bar chart
                subject_df = pd.DataFrame(subject_data)
                pivot_df = subject_df.pivot(index='subject', columns='run_type', values='accuracy')
                
                # Limit to top 20 subjects for readability
                subject_means = pivot_df.mean(axis=1).sort_values(ascending=False)
                top_subjects = subject_means.head(20).index.tolist()
                pivot_df_limited = pivot_df.loc[top_subjects]
                
                # Set up positions for grouped bar chart
                run_types_list = [rt for rt in pivot_df_limited.columns if rt != 'unknown']
                x = np.arange(len(top_subjects))
                bar_width = 0.25
                
                # Create bars for each run type
                for i, run_type in enumerate(run_types_list):
                    style = run_styles.get(run_type, {'color': '#333333'})
                    values = [pivot_df_limited.loc[subj, run_type] if pd.notna(pivot_df_limited.loc[subj, run_type]) else 0 
                             for subj in top_subjects]
                    
                    bars = plt.bar(x + i * bar_width, values, bar_width, 
                                  color=style['color'], alpha=0.8, 
                                  label=run_type.replace('_', ' ').title(),
                                  edgecolor='white', linewidth=0.5)
                
                plt.xlabel('MMLU Individual Subjects (Top 20)')
                plt.ylabel('Accuracy (Best Checkpoints)')
                plt.title('Individual MMLU Subject Performance Comparison', fontweight='bold')
                plt.xticks(x + bar_width, [s[:25] + '...' if len(s) > 25 else s for s in top_subjects], 
                          rotation=45, ha='right', fontsize=9)
                plt.legend(frameon=True, fancybox=True, shadow=True)
                plt.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        
        plt.tight_layout(pad=3.0)
        
        # Save with high DPI and quality
        try:
            plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png', transparent=False)
            print(f"📊 Plot saved to: {save_path}")
        except Exception as e:
            print(f"⚠️  Error saving plot: {e}")
            # Try alternative save
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved with lower DPI to: {save_path}")
            except Exception as e2:
                print(f"❌ Failed to save plot: {e2}")
                return None
        finally:
            plt.close()  # Always close to free memory
        
        return save_path
    
    def create_subtasks_progression_heatmap(self, results: List[Dict], save_path: str = "subtasks_progression_heatmap.png"):
        """Create a large heatmap showing all MMLU subtasks progression over training steps."""
        df = pd.DataFrame(results)
        
        if df.empty:
            print("⚠️  No data available for subtasks progression")
            return None
        
        # Filter out baseline models for progression analysis
        df_checkpoints = df[~df['is_baseline']].copy()
        
        if df_checkpoints.empty:
            print("⚠️  No checkpoint data available for progression")
            return None
        
        # Extract all individual MMLU subjects from the data
        all_subjects = set()
        for result in results:
            for key in result.keys():
                if key.startswith('mmlu_subject_') and not key.endswith('_stderr'):
                    subject = key.replace('mmlu_subject_', '')
                    all_subjects.add(subject)
        
        if not all_subjects:
            print("⚠️  No individual MMLU subjects found")
            return None
        
        # Create progression data for each run type
        run_types = [rt for rt in df_checkpoints['run_type'].unique() if rt != 'unknown']
        
        for run_type in run_types:
            run_data = df_checkpoints[df_checkpoints['run_type'] == run_type].copy()
            run_data = run_data.sort_values('step')
            
            if run_data.empty:
                continue
            
            # Calculate average performance for each subject to sort by
            subject_avg_performance = {}
            for subject in all_subjects:
                subject_key = f'mmlu_subject_{subject}'
                if subject_key in run_data.columns:
                    subject_scores = run_data[subject_key].dropna()
                    if not subject_scores.empty:
                        subject_avg_performance[subject] = subject_scores.mean()
                    else:
                        subject_avg_performance[subject] = 0.0
                else:
                    subject_avg_performance[subject] = 0.0
            
            # Sort subjects by average performance (best first)
            subjects_list = sorted(all_subjects, key=lambda x: subject_avg_performance[x], reverse=True)
            
            # Create matrix for this run: subjects x steps
            steps = sorted(run_data['step'].unique())
            matrix_data = []
            
            for subject in subjects_list:
                subject_key = f'mmlu_subject_{subject}'
                subject_scores = []
                
                for step in steps:
                    step_data = run_data[run_data['step'] == step]
                    if not step_data.empty and subject_key in step_data.columns:
                        # Get the first (should be only) value for this step
                        score = step_data[subject_key].iloc[0]
                        if pd.notna(score):
                            subject_scores.append(score)
                        else:
                            subject_scores.append(np.nan)
                    else:
                        subject_scores.append(np.nan)
                
                matrix_data.append(subject_scores)
            
            if not matrix_data:
                continue
            
            # Create quadratic heatmap
            fig_size = min(max(max(len(steps), len(subjects_list)) * 0.4, 20), 30)  # Square format
            plt.figure(figsize=(fig_size, fig_size))
            
            # Convert to numpy array and add average column
            matrix = np.array(matrix_data)
            
            # Calculate average for each subject (row) and add as additional column
            averages = np.nanmean(matrix, axis=1, keepdims=True)
            matrix_with_avg = np.concatenate([matrix, averages], axis=1)
            
            # Get official MMLU overall scores for each step
            mmlu_overall_scores = []
            # Check available MMLU overall columns (try different naming patterns)
            mmlu_overall_patterns = ['mmlu_overall', 'mmlu_flan_n_shot_loglikelihood', 'mmlu_0shot', 'mmlu_5shot']
            mmlu_col = None
            for pattern in mmlu_overall_patterns:
                if pattern in run_data.columns:
                    mmlu_col = pattern
                    break
            
            for step in steps:
                # Find checkpoint data for this step
                step_data = run_data[run_data['step'] == step]
                if not step_data.empty and mmlu_col and mmlu_col in step_data.columns:
                    mmlu_score = step_data[mmlu_col].iloc[0]
                    mmlu_overall_scores.append(mmlu_score if pd.notna(mmlu_score) else np.nan)
                else:
                    mmlu_overall_scores.append(np.nan)
            
            # Add average MMLU overall score as last column
            mmlu_avg = np.nanmean(mmlu_overall_scores) if mmlu_overall_scores else np.nan
            mmlu_overall_row = np.array(mmlu_overall_scores + [mmlu_avg]).reshape(1, -1)
            
            # Calculate simple average for each step (column) from individual tasks
            col_averages = np.nanmean(matrix_with_avg, axis=0, keepdims=True)
            
            # Add both rows: MMLU Official first, then Simple Average
            matrix_with_both_avg_rows = np.concatenate([mmlu_overall_row, col_averages, matrix_with_avg], axis=0)
            
            # Create heatmap with both average rows at top
            im = plt.imshow(matrix_with_both_avg_rows, cmap='viridis', aspect='auto', interpolation='nearest')
            
            # Set up the plot
            plt.title(f'MMLU Subtasks Progression Over Training Steps\n{run_type.replace("_", " ").title()} Run', 
                     fontweight='bold', fontsize=16, pad=20)
            
            # Y-axis: add both average rows at top, then subjects with average values
            clean_subjects = ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)']  # Two top row labels
            for i, subj in enumerate(subjects_list):
                avg_val = averages[i, 0]
                clean_name = subj.replace('_', ' ').title()
                clean_subjects.append(f'{clean_name} (Avg: {avg_val:.3f})')
            
            plt.yticks(range(len(clean_subjects)), clean_subjects, fontsize=6)
            plt.ylabel('MMLU Subtasks (with Averages)', fontweight='bold', fontsize=14)
            
            # X-axis: steps + average column  
            step_labels = [f'{int(step/1000)}k' for step in steps] + ['AVG']
            plt.xticks(range(len(step_labels)), step_labels, rotation=45, ha='right', fontsize=8)
            plt.xlabel('Training Steps + Average', fontweight='bold', fontsize=14)
            
            
            # Add vertical line to separate average column from training steps
            ax = plt.gca()
            line_x = len(steps) - 0.5  # Position between last step and average column
            ax.axvline(x=line_x, color='white', linewidth=3, linestyle='-')
            ax.axvline(x=line_x, color='black', linewidth=1, linestyle='--')
            
            # Add horizontal line to separate average rows from individual subjects
            line_y = 1.5  # Position between 2nd average row and first subject
            ax.axhline(y=line_y, color='white', linewidth=3, linestyle='-')
            ax.axhline(y=line_y, color='black', linewidth=1, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('MMLU Accuracy', fontweight='bold', fontsize=12)
            
            # Add text annotations for better readability (only for very small matrices)
            if len(subjects_list) <= 15 and len(steps) <= 15:
                for i in range(len(subjects_list)):
                    for j in range(len(steps)):
                        if not np.isnan(matrix[i, j]):
                            text = plt.text(j, i, f'{matrix[i, j]:.2f}',
                                          ha="center", va="center", color="white", fontsize=6)
            
            # Tight layout
            plt.tight_layout()
            
            # Save with run-specific filename
            run_save_path = save_path.replace('.png', f'_{run_type}.png')
            
            try:
                plt.savefig(run_save_path, dpi=300, bbox_inches='tight', facecolor='white',
                           edgecolor='none', format='png', transparent=False)
                print(f"📊 Subtasks progression heatmap saved to: {run_save_path}")
                
                # Also save the heatmap data as CSV table (with average column and both top rows)
                csv_save_path = run_save_path.replace('.png', '_data.csv')
                row_labels = ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)'] + [f'{subj.replace("_", " ").title()} (Avg: {averages[i,0]:.3f})' 
                                                                               for i, subj in enumerate(subjects_list)]
                heatmap_df = pd.DataFrame(matrix_with_both_avg_rows, 
                                        index=row_labels,
                                        columns=[f'{int(step/1000)}k' for step in steps] + ['AVG'])
                heatmap_df.to_csv(csv_save_path)
                print(f"📋 Heatmap data table with averages saved to: {csv_save_path}")
                
            except Exception as e:
                print(f"⚠️  Error saving subtasks heatmap: {e}")
                return None
            finally:
                plt.close()
        
        return save_path
    
    def create_performance_sorted_heatmap(self, results: List[Dict], save_path: str = "performance_sorted_heatmap.png"):
        """Create heatmap with checkpoints sorted by MMLU official performance (worst to best)."""
        df = pd.DataFrame(results)
        
        if df.empty:
            print("⚠️  No data available for performance sorted heatmap")
            return None
        
        # Filter out baseline models for progression analysis
        df_checkpoints = df[~df['is_baseline']].copy()
        
        if df_checkpoints.empty:
            print("⚠️  No checkpoint data available for performance sorted heatmap")
            return None
        
        # Extract all individual MMLU subjects from the data
        all_subjects = set()
        for result in results:
            for key in result.keys():
                if key.startswith('mmlu_subject_') and not key.endswith('_stderr'):
                    subject = key.replace('mmlu_subject_', '')
                    all_subjects.add(subject)
        
        if not all_subjects:
            print("⚠️  No individual MMLU subjects found")
            return None
        
        # Create performance sorted data for each run type
        run_types = [rt for rt in df_checkpoints['run_type'].unique() if rt != 'unknown']
        
        for run_type in run_types:
            run_data = df_checkpoints[df_checkpoints['run_type'] == run_type].copy()
            
            if run_data.empty:
                continue
            
            # Get MMLU official scores for sorting
            mmlu_overall_patterns = ['mmlu_overall', 'mmlu_flan_n_shot_loglikelihood', 'mmlu_0shot', 'mmlu_5shot']
            mmlu_col = None
            for pattern in mmlu_overall_patterns:
                if pattern in run_data.columns:
                    mmlu_col = pattern
                    break
            
            if not mmlu_col:
                print(f"⚠️  No MMLU overall column found for {run_type}")
                continue
            
            # Sort by MMLU official performance (worst to best)
            run_data = run_data.sort_values(mmlu_col, ascending=True)
            
            # Sort subjects by average performance across all checkpoints (best first)
            subjects_list = sorted(list(all_subjects))
            subject_averages = {}
            
            for subject in subjects_list:
                subject_key = f'mmlu_subject_{subject}'
                if subject_key in run_data.columns:
                    subject_scores = run_data[subject_key].dropna()
                    if not subject_scores.empty:
                        subject_averages[subject] = subject_scores.mean()
                    else:
                        subject_averages[subject] = 0
                else:
                    subject_averages[subject] = 0
            
            # Sort subjects by performance (best first)
            subjects_list = sorted(subjects_list, key=lambda x: subject_averages[x], reverse=True)
            
            # Create matrix data
            matrix_data = []
            steps = run_data['step'].tolist()
            
            for subject in subjects_list:
                row_data = []
                subject_key = f'mmlu_subject_{subject}'
                
                for step in steps:
                    step_data = run_data[run_data['step'] == step]
                    if not step_data.empty and subject_key in step_data.columns:
                        score = step_data[subject_key].iloc[0]
                        row_data.append(score if pd.notna(score) else np.nan)
                    else:
                        row_data.append(np.nan)
                
                matrix_data.append(row_data)
            
            if not matrix_data:
                continue
            
            # Create quadratic heatmap
            fig_size = min(max(max(len(steps), len(subjects_list)) * 0.4, 20), 30)  # Square format
            plt.figure(figsize=(fig_size, fig_size))
            
            # Convert to numpy array and add average column
            matrix = np.array(matrix_data)
            
            # Calculate average for each subject (row) and add as additional column
            averages = np.nanmean(matrix, axis=1, keepdims=True)
            matrix_with_avg = np.concatenate([matrix, averages], axis=1)
            
            # Get official MMLU overall scores for each checkpoint (already sorted)
            mmlu_overall_scores = []
            for step in steps:
                step_data = run_data[run_data['step'] == step]
                if not step_data.empty and mmlu_col in step_data.columns:
                    mmlu_score = step_data[mmlu_col].iloc[0]
                    mmlu_overall_scores.append(mmlu_score if pd.notna(mmlu_score) else np.nan)
                else:
                    mmlu_overall_scores.append(np.nan)
            
            # Add average MMLU overall score as last column
            mmlu_avg = np.nanmean(mmlu_overall_scores) if mmlu_overall_scores else np.nan
            mmlu_overall_row = np.array(mmlu_overall_scores + [mmlu_avg]).reshape(1, -1)
            
            # Calculate simple average for each checkpoint from individual tasks
            col_averages = np.nanmean(matrix_with_avg, axis=0, keepdims=True)
            
            # Add both rows: MMLU Official first, then Simple Average
            matrix_with_both_avg_rows = np.concatenate([mmlu_overall_row, col_averages, matrix_with_avg], axis=0)
            
            # Create heatmap with both average rows at top
            im = plt.imshow(matrix_with_both_avg_rows, cmap='viridis', aspect='auto', interpolation='nearest')
            
            # Set up the plot
            plt.title(f'MMLU Performance - Checkpoints Sorted by Official Score (Worst → Best)\n{run_type.replace("_", " ").title()} Run', 
                     fontweight='bold', fontsize=16, pad=20)
            
            # Y-axis: add both average rows at top, then subjects sorted by performance
            clean_subjects = ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)']
            for i, subj in enumerate(subjects_list):
                avg_val = averages[i, 0]
                clean_name = subj.replace('_', ' ').title()
                clean_subjects.append(f'{clean_name} (Avg: {avg_val:.3f})')
            
            plt.yticks(range(len(clean_subjects)), clean_subjects, fontsize=6)
            plt.ylabel('MMLU Subtasks (Sorted by Performance)', fontweight='bold', fontsize=14)
            
            # X-axis: steps sorted by performance + average column (simple labels only)
            step_labels = [f'{int(step/1000)}k' for step in steps] + ['AVG']
            
            plt.xticks(range(len(step_labels)), step_labels, rotation=45, ha='right', fontsize=6)
            plt.xlabel('Checkpoints Sorted by MMLU Official Score (Worst → Best) + Average', fontweight='bold', fontsize=14)
            
            # Add vertical line to separate average column from training steps
            ax = plt.gca()
            line_x = len(steps) - 0.5
            ax.axvline(x=line_x, color='white', linewidth=3, linestyle='-')
            ax.axvline(x=line_x, color='black', linewidth=1, linestyle='--')
            
            # Add horizontal line to separate average rows from individual subjects
            line_y = 1.5
            ax.axhline(y=line_y, color='white', linewidth=3, linestyle='-')
            ax.axhline(y=line_y, color='black', linewidth=1, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('MMLU Accuracy', fontweight='bold', fontsize=12)
            
            # Tight layout
            plt.tight_layout()
            
            # Save with run-specific filename
            run_save_path = save_path.replace('.png', f'_performance_sorted_{run_type}.png')
            
            try:
                plt.savefig(run_save_path, dpi=300, bbox_inches='tight', facecolor='white',
                           edgecolor='none', format='png', transparent=False)
                print(f"📊 Performance sorted heatmap saved to: {run_save_path}")
                
                # Also save the heatmap data as CSV table
                csv_save_path = run_save_path.replace('.png', '_data.csv')
                row_labels = ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)'] + [f'{subj.replace("_", " ").title()} (Avg: {averages[i,0]:.3f})' 
                                                                               for i, subj in enumerate(subjects_list)]
                heatmap_df = pd.DataFrame(matrix_with_both_avg_rows, 
                                        index=row_labels,
                                        columns=step_labels)
                heatmap_df.to_csv(csv_save_path)
                print(f"📋 Performance sorted data saved to: {csv_save_path}")
                
            except Exception as e:
                print(f"⚠️  Error saving performance sorted heatmap: {e}")
                return None
            finally:
                plt.close()
        
        return save_path
    
    def create_baseline_relative_performance_sorted_heatmap(self, results: List[Dict], save_path: str = "baseline_relative_performance_sorted_heatmap.png"):
        """Create baseline-relative heatmap with checkpoints sorted by MMLU official performance (worst to best)."""
        if not results:
            print("⚠️  No results provided for baseline relative performance sorted heatmap")
            return None
            
        df = pd.DataFrame(results)
        
        # Get baseline results (step 0 or run_type 'baseline')
        baseline_results = df[(df['step'] == 0) | (df['run_type'] == 'baseline')].copy()
        if baseline_results.empty:
            print("⚠️  No baseline results found")
            return None
            
        # Get all MMLU subject columns
        all_subjects = set()
        for col in df.columns:
            if col.startswith('mmlu_subject_'):
                subject = col.replace('mmlu_subject_', '')
                all_subjects.add(subject)
        
        if not all_subjects:
            print("⚠️  No individual MMLU subjects found for baseline comparison")
            return None
        
        # Calculate baseline averages for each subject
        baseline_averages = {}
        for subject in all_subjects:
            subject_key = f'mmlu_subject_{subject}'
            if subject_key in baseline_results.columns:
                baseline_scores = baseline_results[subject_key].dropna()
                if not baseline_scores.empty:
                    baseline_averages[subject] = baseline_scores.mean()
                else:
                    baseline_averages[subject] = 0.25  # default random chance
            else:
                baseline_averages[subject] = 0.25
        
        # Also calculate baseline average for MMLU overall score
        mmlu_overall_patterns = ['mmlu_overall', 'mmlu_flan_n_shot_loglikelihood', 'mmlu_0shot', 'mmlu_5shot']
        for pattern in mmlu_overall_patterns:
            if pattern in baseline_results.columns:
                baseline_mmlu_scores = baseline_results[pattern].dropna()
                if not baseline_mmlu_scores.empty:
                    baseline_averages[pattern] = baseline_mmlu_scores.mean()
                else:
                    baseline_averages[pattern] = 0.25
        
        # Filter out baseline and unknown results for trained models
        df_trained = df[(df['run_type'] != 'baseline') & (df['run_type'] != 'unknown') & (df['step'] > 0)].copy()
        
        if df_trained.empty:
            print("⚠️  No trained model results found")
            return None
        
        # Sort subjects by average improvement across all checkpoints (best first)
        subjects_list = sorted(list(all_subjects))
        
        # Generate improvement data for each run type
        run_types = [rt for rt in df_trained['run_type'].unique() if rt != 'unknown']
        
        for run_type in run_types:
            run_data = df_trained[df_trained['run_type'] == run_type].copy()
            
            if run_data.empty:
                continue
            
            # Get MMLU official scores for sorting
            mmlu_overall_patterns = ['mmlu_overall', 'mmlu_flan_n_shot_loglikelihood', 'mmlu_0shot', 'mmlu_5shot']
            mmlu_col = None
            for pattern in mmlu_overall_patterns:
                if pattern in run_data.columns:
                    mmlu_col = pattern
                    break
            
            if not mmlu_col:
                print(f"⚠️  No MMLU overall column found for {run_type}")
                continue
            
            # Sort by MMLU official performance (worst to best)
            run_data = run_data.sort_values(mmlu_col, ascending=True)
            
            # Calculate average improvement for each subject across all checkpoints to sort subjects
            subject_improvements = {}
            for subject in subjects_list:
                subject_key = f'mmlu_subject_{subject}'
                baseline_val = baseline_averages[subject]
                
                if subject_key in run_data.columns:
                    improvements = []
                    for _, row in run_data.iterrows():
                        current_score = row[subject_key]
                        if pd.notna(current_score) and baseline_val > 0:
                            improvement = ((current_score - baseline_val) / baseline_val) * 100
                            improvements.append(improvement)
                    
                    if improvements:
                        subject_improvements[subject] = np.mean(improvements)
                    else:
                        subject_improvements[subject] = 0
                else:
                    subject_improvements[subject] = 0
            
            # Sort subjects by average improvement (best first)
            subjects_list = sorted(subjects_list, key=lambda x: subject_improvements[x], reverse=True)
            
            # Create matrix data
            matrix_data = []
            steps = run_data['step'].tolist()
            
            for subject in subjects_list:
                row_data = []
                subject_key = f'mmlu_subject_{subject}'
                baseline_val = baseline_averages[subject]
                
                for step in steps:
                    step_data = run_data[run_data['step'] == step]
                    if not step_data.empty and subject_key in step_data.columns:
                        current_score = step_data[subject_key].iloc[0]
                        if pd.notna(current_score) and baseline_val > 0:
                            improvement = ((current_score - baseline_val) / baseline_val) * 100
                            row_data.append(improvement)
                        else:
                            row_data.append(np.nan)
                    else:
                        row_data.append(np.nan)
                
                matrix_data.append(row_data)
            
            if not matrix_data:
                continue
            
            # Create quadratic heatmap
            fig_size = min(max(max(len(steps), len(subjects_list)) * 0.4, 20), 30)  # Square format
            plt.figure(figsize=(fig_size, fig_size))
            
            # Convert to numpy array and add average improvement column
            matrix = np.array(matrix_data)
            
            # Calculate average improvement for each subject (row) and add as additional column
            avg_improvements = np.nanmean(matrix, axis=1, keepdims=True)
            matrix_with_avg = np.concatenate([matrix, avg_improvements], axis=1)
            
            # Get official MMLU overall improvement scores for each checkpoint (already sorted)
            mmlu_overall_improvements = []
            baseline_mmlu_overall = baseline_averages.get(mmlu_col, np.nan) if mmlu_col else np.nan
            
            for step in steps:
                step_data = run_data[run_data['step'] == step]
                if not step_data.empty and mmlu_col and mmlu_col in step_data.columns:
                    mmlu_score = step_data[mmlu_col].iloc[0]
                    if pd.notna(mmlu_score) and pd.notna(baseline_mmlu_overall):
                        improvement = ((mmlu_score - baseline_mmlu_overall) / baseline_mmlu_overall) * 100
                        mmlu_overall_improvements.append(improvement)
                    else:
                        mmlu_overall_improvements.append(np.nan)
                else:
                    mmlu_overall_improvements.append(np.nan)
            
            # Add average MMLU overall improvement as last column
            mmlu_improvement_avg = np.nanmean(mmlu_overall_improvements) if mmlu_overall_improvements and not all(pd.isna(mmlu_overall_improvements)) else np.nan
            mmlu_improvement_row = np.array(mmlu_overall_improvements + [mmlu_improvement_avg]).reshape(1, -1)
            
            # Calculate simple average for each checkpoint from individual tasks
            col_averages = np.nanmean(matrix_with_avg, axis=0, keepdims=True)
            
            # Add both rows: MMLU Official improvement first, then Simple Average
            matrix_with_both_avg_rows = np.concatenate([mmlu_improvement_row, col_averages, matrix_with_avg], axis=0)
            
            # Create heatmap with diverging colormap (red=worse, white=no change, green=better)
            im = plt.imshow(matrix_with_both_avg_rows, cmap='RdYlGn', aspect='auto', interpolation='nearest', 
                           vmin=-50, vmax=50)
            
            # Set up the plot
            plt.title(f'MMLU Improvement vs Baseline - Checkpoints Sorted by Official Score (Worst → Best)\n{run_type.replace("_", " ").title()} Run', 
                     fontweight='bold', fontsize=16, pad=20)
            
            # Y-axis: add both average rows at top, then subjects with baseline and average improvement values
            clean_subjects = ['MMLU OFFICIAL IMPROVEMENT', 'SIMPLE AVERAGE IMPROVEMENT']
            for i, subj in enumerate(subjects_list):
                baseline_val = baseline_averages[subj]
                avg_improvement = avg_improvements[i, 0]
                clean_name = subj.replace('_', ' ').title()
                clean_subjects.append(f'{clean_name} (Base: {baseline_val:.3f}, +{avg_improvement:.1f}%)')
            
            plt.yticks(range(len(clean_subjects)), clean_subjects, fontsize=6)
            plt.ylabel('MMLU Subtasks (Sorted by Improvement)', fontweight='bold', fontsize=14)
            
            # X-axis: checkpoints sorted by performance + average column (simple labels only)
            step_labels = [f'{int(step/1000)}k' for step in steps] + ['AVG']
            
            plt.xticks(range(len(step_labels)), step_labels, rotation=45, ha='right', fontsize=6)
            plt.xlabel('Checkpoints Sorted by MMLU Official Score (Worst → Best) + Average', fontweight='bold', fontsize=14)
            
            # Add vertical line to separate average column from training steps
            ax = plt.gca()
            line_x = len(steps) - 0.5
            ax.axvline(x=line_x, color='white', linewidth=3, linestyle='-')
            ax.axvline(x=line_x, color='black', linewidth=1, linestyle='--')
            
            # Add horizontal line to separate average rows from individual subjects
            line_y = 1.5
            ax.axhline(y=line_y, color='white', linewidth=3, linestyle='-')
            ax.axhline(y=line_y, color='black', linewidth=1, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Improvement over Baseline (%)', fontweight='bold', fontsize=12)
            
            # Tight layout
            plt.tight_layout()
            
            # Save with run-specific filename
            run_save_path = save_path.replace('.png', f'_baseline_relative_performance_sorted_{run_type}.png')
            
            try:
                plt.savefig(run_save_path, dpi=300, bbox_inches='tight', facecolor='white',
                           edgecolor='none', format='png', transparent=False)
                print(f"📊 Baseline relative performance sorted heatmap saved to: {run_save_path}")
                
                # Also save the improvement data as CSV table
                csv_save_path = run_save_path.replace('.png', '_data.csv')
                row_labels = ['MMLU OFFICIAL IMPROVEMENT', 'SIMPLE AVERAGE IMPROVEMENT'] + [f'{subj.replace("_", " ").title()} (Base: {baseline_averages[subj]:.3f}, +{avg_improvements[i,0]:.1f}%)' 
                                                                                           for i, subj in enumerate(subjects_list)]
                improvement_df = pd.DataFrame(matrix_with_both_avg_rows, 
                                            index=row_labels,
                                            columns=step_labels)
                improvement_df.to_csv(csv_save_path)
                print(f"📋 Baseline relative performance sorted data saved to: {csv_save_path}")
                
            except Exception as e:
                print(f"⚠️  Error saving baseline relative performance sorted heatmap: {e}")
                return None
            finally:
                plt.close()
        
        return save_path
    
    def create_baseline_relative_heatmap(self, results: List[Dict], save_path: str) -> str:
        """Create heatmap showing performance relative to baseline models."""
        if not results:
            print("⚠️  No results provided for baseline relative heatmap")
            return None
            
        df = pd.DataFrame(results)
        
        # Get baseline results (step 0 or run_type 'baseline')
        baseline_results = df[(df['step'] == 0) | (df['run_type'] == 'baseline')].copy()
        if baseline_results.empty:
            print("⚠️  No baseline results found")
            return None
            
        # Get all MMLU subject columns
        all_subjects = set()
        for col in df.columns:
            if col.startswith('mmlu_subject_'):
                subject = col.replace('mmlu_subject_', '')
                all_subjects.add(subject)
        
        if not all_subjects:
            print("⚠️  No individual MMLU subjects found for baseline comparison")
            return None
        
        # Calculate baseline averages for each subject
        baseline_averages = {}
        for subject in all_subjects:
            subject_key = f'mmlu_subject_{subject}'
            if subject_key in baseline_results.columns:
                baseline_scores = baseline_results[subject_key].dropna()
                if not baseline_scores.empty:
                    baseline_averages[subject] = baseline_scores.mean()
                else:
                    baseline_averages[subject] = 0.25  # default random chance
            else:
                baseline_averages[subject] = 0.25
        
        # Also calculate baseline average for MMLU overall score
        mmlu_overall_patterns = ['mmlu_overall', 'mmlu_flan_n_shot_loglikelihood', 'mmlu_0shot', 'mmlu_5shot']
        for pattern in mmlu_overall_patterns:
            if pattern in baseline_results.columns:
                baseline_mmlu_scores = baseline_results[pattern].dropna()
                if not baseline_mmlu_scores.empty:
                    baseline_averages[pattern] = baseline_mmlu_scores.mean()
                else:
                    baseline_averages[pattern] = 0.25
        
        # Filter out baseline and unknown results for trained models
        df_trained = df[(df['run_type'] != 'baseline') & (df['run_type'] != 'unknown') & (df['step'] > 0)].copy()
        
        if df_trained.empty:
            print("⚠️  No trained model results found")
            return None
        
        # Create relative improvement data for each run type
        run_types = [rt for rt in df_trained['run_type'].unique()]
        
        for run_type in run_types:
            run_data = df_trained[df_trained['run_type'] == run_type].copy()
            run_data = run_data.sort_values('step')
            
            if run_data.empty:
                continue
            
            # Create matrix for this run: subjects x steps (relative to baseline)
            steps = sorted(run_data['step'].unique())
            
            # First pass: calculate improvements for all subjects to determine sorting
            subject_avg_improvements = {}
            temp_matrix_data = {}
            
            for subject in all_subjects:
                subject_key = f'mmlu_subject_{subject}'
                baseline_score = baseline_averages[subject]
                subject_improvements = []
                
                for step in steps:
                    step_data = run_data[run_data['step'] == step]
                    if not step_data.empty and subject_key in step_data.columns:
                        score = step_data[subject_key].iloc[0]
                        if pd.notna(score):
                            # Calculate improvement: (trained - baseline) / baseline * 100
                            improvement = ((score - baseline_score) / baseline_score) * 100
                            subject_improvements.append(improvement)
                        else:
                            subject_improvements.append(0.0)
                    else:
                        subject_improvements.append(0.0)
                
                temp_matrix_data[subject] = subject_improvements
                # Calculate average improvement for this subject
                if subject_improvements:
                    subject_avg_improvements[subject] = np.mean(subject_improvements)
                else:
                    subject_avg_improvements[subject] = 0.0
            
            # Sort subjects by average improvement (best improvements first - highest values at top)
            subjects_list = sorted(all_subjects, key=lambda x: subject_avg_improvements[x], reverse=True)
            
            # Second pass: create matrix in sorted order
            matrix_data = []
            for subject in subjects_list:
                matrix_data.append(temp_matrix_data[subject])
            
            if not matrix_data:
                continue
            
            # Create quadratic heatmap
            fig_size = min(max(max(len(steps), len(subjects_list)) * 0.4, 20), 30)  # Square format
            plt.figure(figsize=(fig_size, fig_size))
            
            # Convert to numpy array and add average improvement column
            matrix = np.array(matrix_data)
            
            # Calculate average improvement for each subject (row) and add as additional column
            avg_improvements = np.nanmean(matrix, axis=1, keepdims=True)
            matrix_with_avg = np.concatenate([matrix, avg_improvements], axis=1)
            
            # Get official MMLU overall improvement scores for each step
            mmlu_overall_improvements = []
            # Check available MMLU overall columns (try different naming patterns)
            mmlu_overall_patterns = ['mmlu_overall', 'mmlu_flan_n_shot_loglikelihood', 'mmlu_0shot', 'mmlu_5shot']
            mmlu_col = None
            for pattern in mmlu_overall_patterns:
                if pattern in run_data.columns:
                    mmlu_col = pattern
                    break
            baseline_mmlu_overall = baseline_averages.get(mmlu_col, np.nan) if mmlu_col else np.nan
            
            for step in steps:
                # Find checkpoint data for this step
                step_data = run_data[run_data['step'] == step]
                if not step_data.empty and mmlu_col and mmlu_col in step_data.columns:
                    mmlu_score = step_data[mmlu_col].iloc[0]
                    if pd.notna(mmlu_score) and pd.notna(baseline_mmlu_overall):
                        improvement = ((mmlu_score - baseline_mmlu_overall) / baseline_mmlu_overall) * 100
                        mmlu_overall_improvements.append(improvement)
                    else:
                        mmlu_overall_improvements.append(np.nan)
                else:
                    mmlu_overall_improvements.append(np.nan)
            
            # Add average MMLU overall improvement as last column
            mmlu_improvement_avg = np.nanmean(mmlu_overall_improvements) if mmlu_overall_improvements and not all(pd.isna(mmlu_overall_improvements)) else np.nan
            mmlu_improvement_row = np.array(mmlu_overall_improvements + [mmlu_improvement_avg]).reshape(1, -1)
            
            # Calculate simple average for each step (column) from individual tasks
            col_averages = np.nanmean(matrix_with_avg, axis=0, keepdims=True)
            
            # Add both rows: MMLU Official improvement first, then Simple Average
            matrix_with_both_avg_rows = np.concatenate([mmlu_improvement_row, col_averages, matrix_with_avg], axis=0)
            
            # Create heatmap with diverging colormap (red=worse, white=no change, green=better)
            im = plt.imshow(matrix_with_both_avg_rows, cmap='RdYlGn', aspect='auto', interpolation='nearest', 
                           vmin=-50, vmax=50)
            
            # Set up the plot
            plt.title(f'MMLU Subtasks Improvement vs Baseline\n{run_type.replace("_", " ").title()} Run', 
                     fontweight='bold', fontsize=16, pad=20)
            
            # Y-axis: add both average rows at top, then subjects with baseline and average improvement values
            clean_subjects = ['MMLU OFFICIAL IMPROVEMENT', 'SIMPLE AVERAGE IMPROVEMENT']  # Two top row labels
            for i, subj in enumerate(subjects_list):
                baseline_val = baseline_averages[subj]
                avg_improvement = avg_improvements[i, 0]
                clean_name = subj.replace('_', ' ').title()
                clean_subjects.append(f'{clean_name} (Base: {baseline_val:.3f}, +{avg_improvement:.1f}%)')
            
            plt.yticks(range(len(clean_subjects)), clean_subjects, fontsize=6)
            plt.ylabel('MMLU Subtasks (Baseline + Avg Improvement)', fontweight='bold', fontsize=14)
            
            # X-axis: steps + average improvement column  
            step_labels = [f'{int(step/1000)}k' for step in steps] + ['AVG']
            plt.xticks(range(len(step_labels)), step_labels, rotation=45, ha='right', fontsize=8)
            plt.xlabel('Training Steps + Average Improvement %', fontweight='bold', fontsize=14)
            
            
            # Add vertical line to separate average column from training steps
            ax = plt.gca()
            line_x = len(steps) - 0.5  # Position between last step and average column
            ax.axvline(x=line_x, color='white', linewidth=3, linestyle='-')
            ax.axvline(x=line_x, color='black', linewidth=1, linestyle='--')
            
            # Add horizontal line to separate average rows from individual subjects
            line_y = 1.5  # Position between 2nd average row and first subject
            ax.axhline(y=line_y, color='white', linewidth=3, linestyle='-')
            ax.axhline(y=line_y, color='black', linewidth=1, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Improvement over Baseline (%)', fontweight='bold', fontsize=12)
            
            # Save with run-specific filename
            run_save_path = save_path.replace('.png', f'_baseline_relative_{run_type}.png')
            
            try:
                plt.savefig(run_save_path, dpi=300, bbox_inches='tight', facecolor='white',
                           edgecolor='none', format='png', transparent=False)
                print(f"📊 Baseline relative heatmap saved to: {run_save_path}")
                
                # Also save the improvement data as CSV table (with average column and both top rows)
                csv_save_path = run_save_path.replace('.png', '_data.csv')
                row_labels = ['MMLU OFFICIAL IMPROVEMENT', 'SIMPLE AVERAGE IMPROVEMENT'] + [f'{subj.replace("_", " ").title()} (Base: {baseline_averages[subj]:.3f}, +{avg_improvements[i,0]:.1f}%)' 
                                                                                           for i, subj in enumerate(subjects_list)]
                improvement_df = pd.DataFrame(matrix_with_both_avg_rows, 
                                            index=row_labels,
                                            columns=[f'{int(step/1000)}k' for step in steps] + ['AVG'])
                improvement_df.to_csv(csv_save_path)
                print(f"📋 Baseline relative data saved to: {csv_save_path}")
                
            except Exception as e:
                print(f"⚠️  Error saving baseline relative heatmap: {e}")
                return None
            finally:
                plt.close()
        
        return save_path
    
    def generate_summary_report(self, results: List[Dict]) -> str:
        """Generate a markdown summary report."""
        df = pd.DataFrame(results)
        
        if df.empty:
            return "# Evaluation Results Summary\n\nNo results found."
        
        report = ["# Evaluation Results Summary", ""]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Models Evaluated:** {len(df)}")
        report.append("")
        
        # Summary by run type
        if 'run_type' in df.columns:
            report.append("## Summary by Run Type")
            report.append("")
            
            for run_type in sorted(df['run_type'].unique()):
                if run_type == 'unknown':
                    continue
                    
                run_data = df[df['run_type'] == run_type]
                report.append(f"### {run_type.replace('_', ' ').title()}")
                report.append(f"- **Models evaluated:** {len(run_data)}")
                
                if 'mmlu_overall' in run_data.columns:
                    best_mmlu = run_data['mmlu_overall'].max()
                    best_idx = run_data['mmlu_overall'].idxmax()
                    best_step = run_data.loc[best_idx, 'step'] if 'step' in run_data.columns else 'N/A'
                    report.append(f"- **Best MMLU:** {best_mmlu:.5f} (step {best_step})")
                
                if 'ppl' in run_data.columns:
                    ppl_data = run_data.dropna(subset=['ppl'])
                    if not ppl_data.empty:
                        best_ppl = ppl_data['ppl'].min()
                        report.append(f"- **Best PPL:** {best_ppl:.5f}")
                
                report.append("")
        
        # Top performing models
        if 'mmlu_overall' in df.columns:
            report.append("## Top 10 Performing Models")
            report.append("")
            top_models = df.nlargest(10, 'mmlu_overall')[['model_name', 'run_type', 'step', 'mmlu_overall', 'ppl']]
            
            report.append("| Rank | Model | Run Type | Step | MMLU | PPL |")
            report.append("|------|-------|----------|------|------|-----|")
            
            for i, (_, row) in enumerate(top_models.iterrows(), 1):
                ppl_str = f"{row['ppl']:.5f}" if pd.notna(row['ppl']) else "N/A"
                report.append(f"| {i} | {row['model_name'][:30]}... | {row['run_type']} | {row['step']} | {row['mmlu_overall']:.5f} | {ppl_str} |")
        
        return "\n".join(report)
    
    def get_run_name(self, run_dir: str) -> str:
        """Extract meaningful run name from directory or results."""
        if not self.results:
            return "unknown_run"
        
        # Count run types to determine the main run
        run_types = {}
        for result in self.results:
            run_type = result.get('run_type', 'unknown')
            run_types[run_type] = run_types.get(run_type, 0) + 1
        
        # Get most common run type (excluding unknown)
        main_run_types = [(count, rt) for rt, count in run_types.items() if rt != 'unknown']
        main_run_types.sort(reverse=True)
        
        if len(main_run_types) == 1:
            # Single run type
            return main_run_types[0][1].replace('_', '-')
        elif len(main_run_types) == 2:
            # Two run types - combine names
            run1, run2 = main_run_types[0][1], main_run_types[1][1]
            return f"{run1.replace('_', '-')}-and-{run2.replace('_', '-')}"
        else:
            # Multiple run types - use directory name
            return Path(run_dir).name if run_dir else "multiple-runs"
    
    def analyze(self, run_dir: Optional[str] = None, save_results: bool = True, experiment_name: Optional[str] = None):
        """Main analysis function."""
        print("🔍 Starting evaluation results analysis...")
        
        # Scan for results
        self.results = self.scan_evaluation_results(run_dir, experiment_name=experiment_name)
        
        if not self.results:
            print("❌ No evaluation results found!")
            return
        
        print(f"✅ Found {len(self.results)} evaluation results")
        
        # Determine run name for file naming
        run_name = self.get_run_name(run_dir)
        
        # Create output directory
        output_dir = Path(f"evaluation_results/{run_name}")
        if save_results:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Output directory: {output_dir}")
        
        # Create comparison table
        comparison_df = self.create_comparison_table(self.results)
        
        # Generate plot with run-specific name
        plot_filename = f"{run_name}_performance_analysis.png"
        plot_path = self.create_performance_plot(self.results, 
                                                str(output_dir / plot_filename) if save_results else plot_filename)
        
        # Generate subtasks progression heatmap
        heatmap_filename = f"{run_name}_subtasks_progression.png"
        heatmap_path = self.create_subtasks_progression_heatmap(self.results,
                                                              str(output_dir / heatmap_filename) if save_results else heatmap_filename)
        
        # Generate performance sorted heatmap 
        performance_sorted_filename = f"{run_name}_performance_sorted.png"
        performance_sorted_path = self.create_performance_sorted_heatmap(self.results,
                                                                        str(output_dir / performance_sorted_filename) if save_results else performance_sorted_filename)
        
        # Generate baseline relative heatmap
        baseline_heatmap_filename = f"{run_name}_baseline_relative.png"
        baseline_heatmap_path = self.create_baseline_relative_heatmap(self.results,
                                                                    str(output_dir / baseline_heatmap_filename) if save_results else baseline_heatmap_filename)
        
        # Generate baseline relative performance sorted heatmap
        baseline_performance_sorted_filename = f"{run_name}_baseline_relative_performance_sorted.png"
        baseline_performance_sorted_path = self.create_baseline_relative_performance_sorted_heatmap(self.results,
                                                                                                   str(output_dir / baseline_performance_sorted_filename) if save_results else baseline_performance_sorted_filename)
        
        # Generate summary report
        summary_report = self.generate_summary_report(self.results)
        
        if save_results:
            # Save comparison table
            table_path = output_dir / f"{run_name}_comparison_table.csv"
            comparison_df.to_csv(table_path, index=False)
            print(f"📋 Comparison table saved to: {table_path}")
            
            # Save summary report
            report_path = output_dir / f"{run_name}_summary_report.md"
            with open(report_path, "w") as f:
                f.write(summary_report)
            print(f"📄 Summary report saved to: {report_path}")
        
        # Display results
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(summary_report)
        print("\n" + "="*80)
        print("📋 COMPARISON TABLE (Top 20)")
        print("="*80)
        print(comparison_df.head(20).to_string(index=False))
        
        return {
            'results': self.results,
            'comparison_df': comparison_df,
            'plot_path': plot_path,
            'heatmap_path': heatmap_path,
            'summary_report': summary_report,
            'output_dir': str(output_dir) if save_results else None
        }

def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results for a specific run')
    parser.add_argument('run_dir', help='Run directory to analyze (e.g., logs/eval_pipeline/runs/2025-08-17_04-01-04 or 2025-08-17_04-01-04)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to files')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment name under evaluation/results/universal (auto-detected if omitted)')
    
    args = parser.parse_args()
    
    if not args.run_dir:
        print("❌ Please specify a run directory!")
        print("Examples:")
        print("  python analyze_evaluation_results.py logs/eval_pipeline/runs/2025-08-17_04-01-04")
        print("  python analyze_evaluation_results.py 2025-08-17_04-01-04")
        return
    
    analyzer = EvaluationAnalyzer()
    
    run_dir = args.run_dir
    save_results = not args.no_save
    
    try:
        results = analyzer.analyze(run_dir=run_dir, save_results=save_results, experiment_name=args.experiment)
        print(f"\n🎉 Analysis complete! Found {len(results['results'])} models.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
