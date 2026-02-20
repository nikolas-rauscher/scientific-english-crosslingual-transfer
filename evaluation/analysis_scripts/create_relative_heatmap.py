import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load T5-base results
with open('/netscratch/nrauscher/projects/BA-hydra/logs/eval_pipeline/runs/2025-08-22_00-35-12/evaluation/results/universal/comprehensive_crosslingual_global_mmlu/t5-base-original_global_mmlu_english_full_0shot_20250822_004350/t5-base/results_2025-08-22T00-48-50.175124.json', 'r') as f:
    t5_base_results = json.load(f)

# Extract T5-base scores
t5_base_scores = {}
for task_name, task_data in t5_base_results['results'].items():
    if 'acc,none' in task_data and 'global_mmlu_full_en_' in task_name:
        clean_name = task_name.replace('global_mmlu_full_en_', '').replace('_', ' ')
        t5_base_scores[clean_name] = task_data['acc,none']

# Load Clean Restart data
df = pd.read_csv('/netscratch/nrauscher/projects/BA-hydra/evaluation_results/clean-restart/clean-restart_subtasks_progression_clean_restart_data.csv')

# Prepare data for heatmap
task_names = []
improvements_matrix = []

for i, row in df.iterrows():
    task_name = str(row.iloc[0])
    
    if not task_name or task_name.startswith('SIMPLE AVERAGE') or i < 3:
        continue
    
    # Extract clean task name
    clean_task = task_name.split('(')[0].strip()
    
    # Find matching baseline
    baseline_score = None
    for t5_task, score in t5_base_scores.items():
        if t5_task.lower().replace(' ', '') in clean_task.lower().replace(' ', '') or \
           clean_task.lower().replace(' ', '') in t5_task.lower().replace(' ', ''):
            baseline_score = score
            break
    
    if baseline_score and baseline_score > 0:
        task_names.append(clean_task[:35])  # Truncate for readability
        
        # Calculate improvements
        improvements = []
        for j in range(1, len(row)-1):  # Skip name and AVG
            current = float(row.iloc[j])
            imp = ((current - baseline_score) / baseline_score) * 100
            improvements.append(imp)
        
        improvements_matrix.append(improvements)

# Create heatmap
fig, ax = plt.subplots(figsize=(24, 14))

# Convert to numpy array
improvements_array = np.array(improvements_matrix)

# Get step labels
step_labels = [col.replace('k', '') for col in df.columns[1:-1]]

# Calculate optimal scale based on actual data
vmin = np.min(improvements_array)
vmax = np.max(improvements_array)

# Use tighter bounds for better contrast (just 5% padding)
padding = 5  # Fixed 5% padding
vmin = vmin - padding
vmax = vmax + padding

print(f'Data range: {np.min(improvements_array):.1f}% to {np.max(improvements_array):.1f}%')
print(f'Using scale: {vmin:.1f}% to {vmax:.1f}%')

# Create heatmap with high contrast colormap
# Using RdYlGn for strong red-yellow-green contrast
import matplotlib.colors as mcolors
colors = ['darkred', 'red', 'lightcoral', 'white', 'lightgreen', 'green', 'darkgreen']
n_bins = 100
cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

sns.heatmap(improvements_array,
            xticklabels=step_labels,
            yticklabels=task_names,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Relative Improvement vs T5-Base (%)', 'shrink': 0.8},
            linewidths=0.3,
            linecolor='gray',
            ax=ax)

# Styling
ax.set_title('Relative Performance Improvement: Clean Restart vs T5-Base Baseline\nAcross All MMLU Subtasks', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Training Steps (k)', fontsize=14, fontweight='bold')
ax.set_ylabel('MMLU Subtasks', fontsize=14, fontweight='bold')

# Rotate labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

# Add vertical line at best checkpoint (250k)
best_step_idx = 21  # 250k is at index 21
ax.axvline(x=best_step_idx, color='gold', linewidth=2, linestyle='--', alpha=0.7)
ax.text(best_step_idx, -1, 'Best (250k)', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/netscratch/nrauscher/projects/BA-hydra/evaluation_results/clean-restart/relative_improvement_heatmap_vs_t5base.png', 
            dpi=150, bbox_inches='tight')

print(f'✅ Heatmap saved: evaluation_results/clean-restart/relative_improvement_heatmap_vs_t5base.png')

# Calculate statistics
print(f'\n📊 Summary Statistics:')
print(f'  Tasks analyzed: {len(improvements_matrix)}')
print(f'  Average improvement at 250k: {np.mean(improvements_array[:, best_step_idx]):.1f}%')
print(f'  Max single improvement: {np.max(improvements_array):.1f}%')
print(f'  Max single decline: {np.min(improvements_array):.1f}%')

# Find consistently improved tasks
always_better = []
for i, task in enumerate(task_names):
    if all(improvements_array[i, :] > 0):
        always_better.append(task)

print(f'\n✨ Tasks always better than T5-Base: {len(always_better)}/{len(task_names)}')