# MMLU Evaluation Analysis Scripts

Collection of analysis scripts for MMLU evaluation results visualization and comparison.

## Scripts Overview

### 1. `analyze_evaluation_results.py`
**Purpose**: Comprehensive MMLU performance analysis with heatmaps  
**Output**: Performance heatmaps, progression charts, sorted comparisons  
**Usage**: `python analyze_evaluation_results.py <run_directory>`

### 2. `create_comparison_bar_chart.py`  
**Purpose**: Horizontal bar chart comparing absolute MMLU scores  
**Output**: Bar chart showing best checkpoints vs baseline (absolute scores)  
**Usage**: `python create_comparison_bar_chart.py <run_directory>`

### 3. `create_modern_comparison_chart.py`
**Purpose**: Interactive Plotly version of comparison charts  
**Output**: Modern HTML interactive charts (absolute scores)  
**Usage**: `python create_modern_comparison_chart.py <run_directory>`

### 4. `create_relative_improvement_chart.py`
**Purpose**: Interactive relative improvement analysis (categories + individual)  
**Output**: Two separate charts - grouped categories & individual tasks (relative %)  
**Usage**: `python create_relative_improvement_chart.py <run_directory>`

### 5. `create_relative_bar_chart.py`
**Purpose**: Horizontal bar chart showing percentage improvements vs baseline  
**Output**: Bar chart with relative improvements (percentage changes)  
**Usage**: `python create_relative_bar_chart.py <run_directory>`

### 6. `plot_mmlu_subtasks_across_models.py`
**Purpose**: Grouped horizontal bars per MMLU subtask across multiple models (EN/DE).  
**Output**: Interactive Plotly HTML (+ PNG if kaleido) and CSV with per‑subject accuracies.  
**Usage**:
```
python evaluation/analysis_scripts/plot_mmlu_subtasks_across_models.py \
  <results_dir_with_jsons> \
  --model-alias "wechsel-transfer=Wechsel‑15k" \
  --model-alias "german-native=Native‑15k" \
  --model-alias "germanT5=HF‑GermanT5" \
  --model-alias "clean_restart=EN‑Gold"
```
Saves to `evaluation/evaluation_results/<results_dir_name>/` by default.

## Input Data Structure

All scripts expect evaluation results from:
```
<run_directory>/
├── evaluation/results/universal/universal_evaluation/
│   ├── baseline_*/results_*.json
│   ├── clean_restart_*/results_*.json
│   └── green_run_*/results_*.json
```

## Key Differences

| Script | Chart Type | Values | Categories | Interactive |
|--------|------------|--------|------------|-------------|
| `analyze_evaluation_results` | Heatmaps | Absolute | Individual | No |
| `create_comparison_bar_chart` | Horizontal Bar | Absolute | Individual | No |
| `create_modern_comparison_chart` | Horizontal Bar | Absolute | Individual | Yes (HTML) |
| `create_relative_improvement_chart` | Horizontal Bar | Relative % | Grouped + Individual | Yes (HTML) |
| `create_relative_bar_chart` | Horizontal Bar | Relative % | Individual | No |

## Output Files

- **PNG**: Static images for papers/presentations
- **HTML**: Interactive charts (Plotly-based)  
- **CSV**: Raw data and summary statistics
- **Output Directory**: `evaluation/evaluation_results/<run_name>/`

## Usage Examples

```bash
# Comprehensive heatmap analysis
python evaluation/analysis_scripts/analyze_evaluation_results.py logs/eval_pipeline/runs/2025-08-17_03-57-04

# Quick relative improvements
python evaluation/analysis_scripts/create_relative_bar_chart.py logs/eval_pipeline/runs/2025-08-17_03-57-04

# Interactive category analysis  
python evaluation/analysis_scripts/create_relative_improvement_chart.py logs/eval_pipeline/runs/2025-08-17_03-57-04
```
