#!/usr/bin/env python3
"""
Enhanced Cross-Lingual MMLU Comparison Plot for Thesis

Creates comprehensive visualizations:
1. Overall comparison (EN vs DE)
2. Category comparison (EN vs DE)
3. Improved subtask comparison (EN vs DE)
4. Regressed subtask comparison (EN vs DE)
5. Combined EN+DE Top 10 / Bottom 10 subtasks (averaged scores)
6. English-only Top 10 / Bottom 10 subtasks
7. German-only Top 10 / Bottom 10 subtasks
8. Summary CSV with overall and category scores
9. Detailed subtask CSV
10. Ranked subtask CSV (best to worst)

Saves results in organized subdirectories.

Usage:
    python create_crosslingual_comparison_plot_v2.py <eval_run_dir> <experiment_name>
    python create_crosslingual_comparison_plot_v2.py logs/eval_pipeline/runs/2025-09-23_19-00-53 scientific_crosslingual_transfer_eval_full_15k
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
import seaborn as sns
from matplotlib.patches import Patch

# Category definitions for MMLU
MMLU_CATEGORIES = ['Humanities', 'STEM', 'Social Sciences', 'Other']


MODEL_VISUAL_PRIORITY = {
    'gold-continued-pretraind-on-english-487k': 0,
    'wechsel-transfer-en-487k-continued-pretraind-on-german-15k': 1,
    'wechsel-transfer-en-487k': 2,
    'hf-germant5-continued-pretraind-on-german-15k': 3,
    'hf-germant5-base': 4,
    't5-base': 5,
}


def get_model_visual_priority(model_name):
    key = model_name.lower()
    for candidate, priority in MODEL_VISUAL_PRIORITY.items():
        if key == candidate:
            return priority
        if key.endswith(candidate):
            return priority
        if candidate in key:
            return priority
    return 99


def identify_baseline_model(results):
    """Return the key of the T5-Base reference model."""
    if 't5-base' in results:
        return 't5-base'

    for key in results:
        lower = key.lower()
        if lower.endswith('t5-base') or lower == 't5-base-original':
            return key

    for key in results:
        if 't5-base' in key.lower():
            return key

    return None


def identify_highlight_model(results):
    """Return the key of the Wechsel + 15k DE model."""
    preferred = 'wechsel-transfer-en-487k-continued-pretraind-on-german-15k'
    if preferred in results:
        return preferred

    candidates = [key for key in results if 'wechsel' in key.lower() and '15k' in key.lower()]
    return candidates[0] if candidates else None


def compute_model_improvements(results, baseline_key):
    """Compute per-model improvements over the baseline."""
    improvements = {}

    baseline = results.get(baseline_key, {'en': {}, 'de': {}})
    baseline_en = baseline.get('en', {}).get('overall', np.nan)
    baseline_de = baseline.get('de', {}).get('overall', np.nan)

    for model, model_results in results.items():
        en_score = model_results.get('en', {}).get('overall', np.nan)
        de_score = model_results.get('de', {}).get('overall', np.nan)

        delta_en = en_score - baseline_en if not np.isnan(en_score) and not np.isnan(baseline_en) else np.nan
        delta_de = de_score - baseline_de if not np.isnan(de_score) and not np.isnan(baseline_de) else np.nan

        if np.isnan(delta_en) and np.isnan(delta_de):
            mean_delta = np.nan
        else:
            mean_delta = np.nanmean([delta_en, delta_de])

        improvements[model] = {
            'delta_en': delta_en,
            'delta_de': delta_de,
            'mean_delta': mean_delta,
        }

    return improvements


def collect_unique_subtasks(results):
    """Return the set of all unique subtasks across languages."""
    subtasks = set()
    for model_results in results.values():
        for lang in ('en', 'de'):
            subtasks.update(model_results.get(lang, {}).get('subtasks', {}).keys())
    return subtasks


def load_evaluation_results(eval_run_dir, experiment_name):
    """Load all evaluation results including individual subtasks."""
    base_path = Path(eval_run_dir) / "evaluation" / "results" / "universal" / experiment_name

    if not base_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {base_path}")

    results = {}

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        dir_name = model_dir.name

        # Extract model name and language
        if '_global_mmlu_en_' in dir_name:
            model_name = dir_name.split('_global_mmlu_en_')[0]
            language = 'en'
        elif '_global_mmlu_de_' in dir_name:
            model_name = dir_name.split('_global_mmlu_de_')[0]
            language = 'de'
        else:
            continue

        json_files = list(model_dir.rglob('results_*.json'))
        if not json_files:
            continue

        json_file = json_files[0]

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'results' not in data:
                continue

            results_data = data['results']
            lang_key = f'global_mmlu_full_{language}'

            if lang_key not in results_data:
                continue

            if model_name not in results:
                results[model_name] = {'en': {}, 'de': {}}

            # Overall score
            results[model_name][language]['overall'] = results_data[lang_key].get('acc,none')
            results[model_name][language]['overall_stderr'] = results_data[lang_key].get('acc_stderr,none')

            # Category scores
            for category in MMLU_CATEGORIES:
                cat_key = f'{lang_key}_{category.lower().replace(" ", "_")}'
                if cat_key in results_data:
                    results[model_name][language][category] = results_data[cat_key].get('acc,none')

            # Individual subtask scores
            results[model_name][language]['subtasks'] = {}
            for key, value in results_data.items():
                if key.startswith(f'{lang_key}_') and isinstance(value, dict):
                    # Skip category aggregates
                    if key in [f'{lang_key}', f'{lang_key}_humanities', f'{lang_key}_stem',
                              f'{lang_key}_social_sciences', f'{lang_key}_other']:
                        continue

                    subtask_name = key.replace(f'{lang_key}_', '')
                    if 'acc,none' in value:
                        results[model_name][language]['subtasks'][subtask_name] = value['acc,none']

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def get_nice_model_name(model_name):
    """Convert technical model names to readable display names matching methodology section."""
    mappings = {
        't5-base': 'EN-T5-Base',
        'gold-continued-pretraind-on-english-487k': 'EN-T5-Sci',
        'hf-germanT5-base': 'DE-T5-Base',
        'hf-germanT5-continued-pretraind-on-german-15k': 'DE-T5-Base-15k',
        'wechsel-transfer-en-487k': 'DE-T5-Sci-Transfer-Init',
        'wechsel-transfer-en-487k-continued-pretraind-on-german-15k': 'DE-T5-Sci-Transfer-15k',
        't5-base-original': 'EN-T5-Base',
        'run4-t5-optimized-640k-english-source': 'EN-T5-Sci',
        'german-t5-base-nl36': 'DE-T5-Base',
        'wechsel-german-from-640k-en-robust': 'DE-T5-Sci-Transfer-Init',
        'wechsel-de-zu-en-from-efficient-gc4-nl36': 'DE-EN-Transfer'
    }
    return mappings.get(model_name, model_name.replace('-', ' ').title())


def get_model_color(model_name):
    """Assign colors to models based on type."""
    # T5-Base: Grau
    if 't5-base' == model_name.lower() or 't5-base-original' in model_name.lower():
        return '#808080'  # Gray - T5 baseline

    # Gold EN 487k: Gelb/Gold
    elif 'gold' in model_name.lower() or ('487k' in model_name.lower() and 'wechsel' not in model_name.lower()):
        return '#FFB300'  # Amber/Gold - English trained

    # Wechsel + 15k: Dunkel Lila
    elif 'wechsel' in model_name.lower() and '15k' in model_name.lower():
        return '#5E2A84'  # Deep Purple - Wechsel + continued (best model!)

    # Wechsel (ohne 15k): Hell Lila
    elif 'wechsel' in model_name.lower():
        return '#BA68C8'  # Light Purple - Wechsel transfer

    # German-T5 Base: Hellblau
    elif 'german' in model_name.lower() and '15k' not in model_name.lower():
        return '#64B5F6'  # Light Blue - German baseline

    # German-T5 + 15k: Dunkelblau
    elif 'german' in model_name.lower() and '15k' in model_name.lower():
        return '#1976D2'  # Dark Blue - German + continued

    else:
        return '#424242'  # Dark Gray - other


def create_main_comparison_plot(results, output_dir, baseline_key, improvements):
    """Create main overview plot with overall and category scores."""

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.8], hspace=0.3, wspace=0.32)

    def improvement_key(model):
        model_delta = improvements.get(model, {}).get('mean_delta', np.nan)
        return model_delta if not np.isnan(model_delta) else float('-inf')

    models_sorted = sorted(results.keys(), key=improvement_key, reverse=True)
    ordered_models = sorted(models_sorted, key=get_model_visual_priority)

    # Plot 1: Overall Performance - English
    ax1 = fig.add_subplot(gs[0, 0])
    en_scores = []
    model_labels = []
    colors = []
    models_en = []

    for model in ordered_models:
        if 'overall' in results[model]['en']:
            en_scores.append(results[model]['en']['overall'])
            model_labels.append(get_nice_model_name(model))
            colors.append(get_model_color(model))
            models_en.append(model)

    y_pos = np.arange(len(model_labels))
    bars = ax1.barh(y_pos, en_scores, color=colors, height=0.65,
                    edgecolor='black', linewidth=1.2, alpha=0.9)
    ax1.set_xlim(0, 0.35)
    ax1.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Overall MMLU - English', fontsize=14, fontweight='bold', pad=12)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_labels, fontsize=11)
    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)

    for i, (model, score) in enumerate(zip(models_en, en_scores)):
        delta_en = improvements.get(model, {}).get('delta_en', np.nan)
        delta_text = f' ({delta_en:+.3f})' if not np.isnan(delta_en) else ''
        ax1.text(score + 0.003, i, f'{score:.4f}{delta_text}',
                va='center', fontsize=10, fontweight='bold')

    # Plot 2: Overall Performance - German
    ax2 = fig.add_subplot(gs[0, 1])
    de_scores = []
    model_labels_de = []
    colors_de = []
    models_de = []

    for model in ordered_models:
        if 'overall' in results[model]['de']:
            de_scores.append(results[model]['de']['overall'])
            model_labels_de.append(get_nice_model_name(model))
            colors_de.append(get_model_color(model))
            models_de.append(model)

    y_pos_de = np.arange(len(model_labels_de))
    bars = ax2.barh(y_pos_de, de_scores, color=colors_de, height=0.65,
                    edgecolor='black', linewidth=1.2, alpha=0.9)
    ax2.set_xlim(0, 0.35)
    ax2.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Overall MMLU - German', fontsize=14, fontweight='bold', pad=12)
    ax2.set_yticks(y_pos_de)
    ax2.set_yticklabels(model_labels_de, fontsize=11)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)

    for i, (model, score) in enumerate(zip(models_de, de_scores)):
        delta_de = improvements.get(model, {}).get('delta_de', np.nan)
        delta_text = f' ({delta_de:+.3f})' if not np.isnan(delta_de) else ''
        ax2.text(score + 0.003, i, f'{score:.4f}{delta_text}',
                va='center', fontsize=10, fontweight='bold')

    # Plot 3: Category Performance - English
    ax3 = fig.add_subplot(gs[1, 0])

    category_width = 0.13
    x_positions = np.arange(len(MMLU_CATEGORIES))

    for i, model in enumerate(ordered_models):
        if not results[model]['en']:
            continue

        cat_scores = [results[model]['en'].get(cat, np.nan) for cat in MMLU_CATEGORIES]
        offset = (i - len(models_sorted)/2) * category_width

        ax3.bar(x_positions + offset, cat_scores, category_width,
               label=get_nice_model_name(model),
               color=get_model_color(model),
               edgecolor='black', linewidth=0.8, alpha=0.9)

    ax3.set_xlabel('MMLU Categories', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax3.set_title('Category Performance - English', fontsize=14, fontweight='bold', pad=12)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(MMLU_CATEGORIES, rotation=0, fontsize=11)
    ax3.set_ylim(0, 0.35)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Plot 4: Category Performance - German
    ax4 = fig.add_subplot(gs[1, 1])

    for i, model in enumerate(ordered_models):
        if not results[model]['de']:
            continue

        cat_scores = [results[model]['de'].get(cat, np.nan) for cat in MMLU_CATEGORIES]
        offset = (i - len(models_sorted)/2) * category_width

        ax4.bar(x_positions + offset, cat_scores, category_width,
               label=get_nice_model_name(model),
               color=get_model_color(model),
               edgecolor='black', linewidth=0.8, alpha=0.9)

    ax4.set_xlabel('MMLU Categories', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax4.set_title('Category Performance - German', fontsize=14, fontweight='bold', pad=12)
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(MMLU_CATEGORIES, rotation=0, fontsize=11)
    ax4.set_ylim(0, 0.35)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    legend_handles = []
    legend_labels = []
    for model in ordered_models:
        label = get_nice_model_name(model)
        if label in legend_labels:
            continue
        legend_handles.append(Patch(facecolor=get_model_color(model), edgecolor='black'))
        legend_labels.append(label)

    fig.legend(legend_handles, legend_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=6, fontsize=13, frameon=True,
               fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])

    output_path = output_dir / "01_main_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved main comparison plot: {output_path}")

    pdf_path = output_dir / "01_main_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()


def create_subtask_comparison_plot(results, output_dir, baseline_key, highlight_key,
                                   improvements, all_subtasks, mode='improved'):
    """Create comparison plot grouped by improvement sign vs. baseline."""

    if mode not in {'improved', 'regressed'}:
        raise ValueError("mode must be 'improved' or 'regressed'")

    def improvement_key(model):
        model_delta = improvements.get(model, {}).get('mean_delta', np.nan)
        return model_delta if not np.isnan(model_delta) else float('-inf')

    models_sorted = sorted(results.keys(), key=improvement_key, reverse=True)
    ordered_models = sorted(models_sorted, key=get_model_visual_priority)

    baseline = results.get(baseline_key, {})
    highlight = results.get(highlight_key, {})

    baseline_en_subtasks = baseline.get('en', {}).get('subtasks', {})
    baseline_de_subtasks = baseline.get('de', {}).get('subtasks', {})
    highlight_en_subtasks = highlight.get('en', {}).get('subtasks', {})
    highlight_de_subtasks = highlight.get('de', {}).get('subtasks', {})

    delta_relevant_subtasks = (set(highlight_en_subtasks.keys()) |
                               set(highlight_de_subtasks.keys()) |
                               set(baseline_en_subtasks.keys()) |
                               set(baseline_de_subtasks.keys()))

    subtask_deltas = {}

    for subtask in delta_relevant_subtasks:
        delta_en = np.nan
        delta_de = np.nan

        if subtask in highlight_en_subtasks and subtask in baseline_en_subtasks:
            delta_en = highlight_en_subtasks[subtask] - baseline_en_subtasks[subtask]

        if subtask in highlight_de_subtasks and subtask in baseline_de_subtasks:
            delta_de = highlight_de_subtasks[subtask] - baseline_de_subtasks[subtask]

        valid_deltas = [d for d in [delta_en, delta_de] if not np.isnan(d)]
        mean_delta = np.nanmean(valid_deltas) if valid_deltas else np.nan

        if not np.isnan(mean_delta):
            subtask_deltas[subtask] = {
                'mean_delta': mean_delta,
                'delta_en': delta_en,
                'delta_de': delta_de,
            }

    # Add remaining subtasks without deltas so that coverage is complete
    for subtask in all_subtasks:
        if subtask not in subtask_deltas:
            subtask_deltas[subtask] = {
                'mean_delta': np.nan,
                'delta_en': np.nan,
                'delta_de': np.nan,
            }

    improved = [item for item in subtask_deltas.items()
                if not np.isnan(item[1]['mean_delta']) and item[1]['mean_delta'] > 0]
    regressed = [item for item in subtask_deltas.items()
                 if np.isnan(item[1]['mean_delta']) or item[1]['mean_delta'] <= 0]

    improved.sort(key=lambda x: x[1]['mean_delta'], reverse=True)

    def regression_key(item):
        mean_delta = item[1]['mean_delta']
        if np.isnan(mean_delta):
            return float('inf')
        return mean_delta

    regressed.sort(key=regression_key)

    if mode == 'improved':
        selected = improved
        title_prefix = 'Improved Subtasks'
        suffix = 'improved'
    else:
        selected = regressed
        title_prefix = 'Regressed / Unchanged Subtasks'
        suffix = 'regressed'

    if not selected:
        # Fallback: ensure we draw something even if there are no items in this category
        selected = improved if improved else regressed
        if not selected:
            raise ValueError('No subtasks available to plot')

    selected_names = [s[0] for s in selected]
    delta_lookup = {name: stats for name, stats in selected}

    # Create plot for English
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    fig.suptitle(f'{title_prefix} vs. T5-Base (n={len(selected_names)})',
                 fontsize=16, fontweight='bold')

    # English subtasks
    y_pos = np.arange(len(selected_names))
    bar_width = 0.12

    for i, model in enumerate(ordered_models):
        if 'subtasks' not in results[model]['en']:
            continue

        scores = []
        for subtask in selected_names:
            score = results[model]['en']['subtasks'].get(subtask, 0)
            scores.append(score)

        offset = (i - len(models_sorted)/2) * bar_width
        ax1.barh(y_pos + offset, scores, bar_width,
                label=get_nice_model_name(model),
                color=get_model_color(model),
                edgecolor='black', linewidth=0.5, alpha=0.9)

    ax1.set_yticks(y_pos)
    def format_label(name):
        pretty = name.replace('_', ' ').title()
        stats = delta_lookup.get(name, {})
        mean_delta = stats.get('mean_delta', np.nan)
        suffix = f' ({mean_delta:+.3f})' if not np.isnan(mean_delta) else ''
        return f'{pretty[:40]}{suffix}'

    ax1.set_yticklabels([format_label(s) for s in selected_names], fontsize=8)
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title(f'English MMLU Subtasks ({title_prefix})', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlim(0, 0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=8, ncol=1)
    ax1.invert_yaxis()

    # German subtasks
    for i, model in enumerate(ordered_models):
        if 'subtasks' not in results[model]['de']:
            continue

        scores = []
        for subtask in selected_names:
            score = results[model]['de']['subtasks'].get(subtask, 0)
            scores.append(score)

        offset = (i - len(models_sorted)/2) * bar_width
        ax2.barh(y_pos + offset, scores, bar_width,
                label=get_nice_model_name(model),
                color=get_model_color(model),
                edgecolor='black', linewidth=0.5, alpha=0.9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([format_label(s) for s in selected_names], fontsize=8)
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title(f'German MMLU Subtasks ({title_prefix})', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlim(0, 0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=8, ncol=1)
    ax2.invert_yaxis()

    plt.tight_layout()

    index_prefix = "02" if mode == 'improved' else "03"
    output_path = output_dir / f"{index_prefix}_{suffix}_subtasks.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved subtask comparison plot: {output_path}")

    pdf_path = output_dir / f"{index_prefix}_{suffix}_subtasks.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()

    return selected_names


def create_top_bottom_plot_combined(results, output_dir, baseline_key, highlight_key,
                                     improvements, all_subtasks, n=10):
    """Create separate Top N and Bottom N plots for combined EN+DE subtasks."""

    def improvement_key(model):
        model_delta = improvements.get(model, {}).get('mean_delta', np.nan)
        return model_delta if not np.isnan(model_delta) else float('-inf')

    models_sorted = sorted(results.keys(), key=improvement_key, reverse=True)
    ordered_models = sorted(models_sorted, key=get_model_visual_priority)

    baseline_en_subtasks = results.get(baseline_key, {}).get('en', {}).get('subtasks', {})
    baseline_de_subtasks = results.get(baseline_key, {}).get('de', {}).get('subtasks', {})
    highlight_en_subtasks = results.get(highlight_key, {}).get('en', {}).get('subtasks', {})
    highlight_de_subtasks = results.get(highlight_key, {}).get('de', {}).get('subtasks', {})

    subtask_stats = {}
    for subtask in all_subtasks:
        en_baseline = baseline_en_subtasks.get(subtask, np.nan)
        de_baseline = baseline_de_subtasks.get(subtask, np.nan)
        en_highlight = highlight_en_subtasks.get(subtask, np.nan)
        de_highlight = highlight_de_subtasks.get(subtask, np.nan)

        valid_baseline = [s for s in [en_baseline, de_baseline] if not np.isnan(s)]
        valid_highlight = [s for s in [en_highlight, de_highlight] if not np.isnan(s)]

        if valid_baseline and valid_highlight:
            mean_baseline = np.mean(valid_baseline)
            mean_highlight = np.mean(valid_highlight)
            delta = mean_highlight - mean_baseline

            subtask_stats[subtask] = {
                'mean_baseline': mean_baseline,
                'mean_highlight': mean_highlight,
                'delta': delta
            }

    sorted_subtasks = sorted(subtask_stats.items(), key=lambda x: x[1]['delta'], reverse=True)

    top_n = sorted_subtasks[:n]
    bottom_n = sorted_subtasks[-n:][::-1]

    bar_width = 0.12

    # Create Top N plot
    fig_top = plt.figure(figsize=(12, 10))
    ax_top = fig_top.add_subplot(111)
    fig_top.suptitle(f'Top {n} MMLU Subtasks (EN+DE Combined)',
                     fontsize=18, fontweight='bold')

    top_names = [item[0] for item in top_n]
    y_pos = np.arange(len(top_names))

    for i, model in enumerate(ordered_models):
        en_subtasks = results[model].get('en', {}).get('subtasks', {})
        de_subtasks = results[model].get('de', {}).get('subtasks', {})

        combined_scores = []
        for subtask in top_names:
            en_score = en_subtasks.get(subtask, np.nan)
            de_score = de_subtasks.get(subtask, np.nan)
            valid = [s for s in [en_score, de_score] if not np.isnan(s)]
            combined_scores.append(np.mean(valid) if valid else 0)

        offset = (i - len(models_sorted)/2) * bar_width
        ax_top.barh(y_pos + offset, combined_scores, bar_width,
                    label=get_nice_model_name(model),
                    color=get_model_color(model),
                    edgecolor='black', linewidth=0.5, alpha=0.9)

    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels([f"{name.replace('_', ' ').title()[:40]} ({stats['delta']:+.3f})"
                            for name, stats in top_n], fontsize=10)
    ax_top.set_xlabel('Accuracy (EN+DE Average)', fontsize=13, fontweight='bold')
    ax_top.set_title(f'Top {n} Improved Subtasks', fontsize=14, fontweight='bold', pad=10)
    ax_top.set_xlim(0, 0.5)
    ax_top.grid(axis='x', alpha=0.3, linestyle='--')
    ax_top.legend(loc='lower right', fontsize=9)
    ax_top.invert_yaxis()

    plt.tight_layout()

    output_path = output_dir / f"06_top{n}_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined top plot: {output_path}")

    pdf_path = output_dir / f"06_top{n}_combined.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()

    # Create Bottom N plot
    fig_bottom = plt.figure(figsize=(12, 10))
    ax_bottom = fig_bottom.add_subplot(111)
    fig_bottom.suptitle(f'Bottom {n} MMLU Subtasks (EN+DE Combined)',
                        fontsize=18, fontweight='bold')

    bottom_names = [item[0] for item in bottom_n]
    y_pos_bottom = np.arange(len(bottom_names))

    for i, model in enumerate(ordered_models):
        en_subtasks = results[model].get('en', {}).get('subtasks', {})
        de_subtasks = results[model].get('de', {}).get('subtasks', {})

        combined_scores = []
        for subtask in bottom_names:
            en_score = en_subtasks.get(subtask, np.nan)
            de_score = de_subtasks.get(subtask, np.nan)
            valid = [s for s in [en_score, de_score] if not np.isnan(s)]
            combined_scores.append(np.mean(valid) if valid else 0)

        offset = (i - len(models_sorted)/2) * bar_width
        ax_bottom.barh(y_pos_bottom + offset, combined_scores, bar_width,
                       label=get_nice_model_name(model),
                       color=get_model_color(model),
                       edgecolor='black', linewidth=0.5, alpha=0.9)

    ax_bottom.set_yticks(y_pos_bottom)
    ax_bottom.set_yticklabels([f"{name.replace('_', ' ').title()[:40]} ({stats['delta']:+.3f})"
                               for name, stats in bottom_n], fontsize=10)
    ax_bottom.set_xlabel('Accuracy (EN+DE Average)', fontsize=13, fontweight='bold')
    ax_bottom.set_title(f'Bottom {n} Regressed Subtasks', fontsize=14, fontweight='bold', pad=10)
    ax_bottom.set_xlim(0, 0.5)
    ax_bottom.grid(axis='x', alpha=0.3, linestyle='--')
    ax_bottom.legend(loc='lower right', fontsize=9)
    ax_bottom.invert_yaxis()

    plt.tight_layout()

    output_path = output_dir / f"07_bottom{n}_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined bottom plot: {output_path}")

    pdf_path = output_dir / f"07_bottom{n}_combined.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()


def create_top_bottom_plot_separate(results, output_dir, baseline_key, highlight_key,
                                     improvements, all_subtasks, language='en', n=10):
    """Create separate Top N and Bottom N plots for English or German subtasks."""

    if language not in ['en', 'de']:
        raise ValueError("language must be 'en' or 'de'")

    lang_display = 'English' if language == 'en' else 'German'

    def improvement_key(model):
        model_delta = improvements.get(model, {}).get('mean_delta', np.nan)
        return model_delta if not np.isnan(model_delta) else float('-inf')

    models_sorted = sorted(results.keys(), key=improvement_key, reverse=True)
    ordered_models = sorted(models_sorted, key=get_model_visual_priority)

    baseline_subtasks = results.get(baseline_key, {}).get(language, {}).get('subtasks', {})
    highlight_subtasks = results.get(highlight_key, {}).get(language, {}).get('subtasks', {})

    subtask_deltas = {}
    for subtask in all_subtasks:
        baseline_score = baseline_subtasks.get(subtask, np.nan)
        highlight_score = highlight_subtasks.get(subtask, np.nan)

        if not np.isnan(baseline_score) and not np.isnan(highlight_score):
            delta = highlight_score - baseline_score
            subtask_deltas[subtask] = {
                'baseline': baseline_score,
                'highlight': highlight_score,
                'delta': delta
            }

    sorted_subtasks = sorted(subtask_deltas.items(), key=lambda x: x[1]['delta'], reverse=True)

    top_n = sorted_subtasks[:n]
    bottom_n = sorted_subtasks[-n:][::-1]

    bar_width = 0.12

    # Create Top N plot
    fig_top = plt.figure(figsize=(12, 10))
    ax_top = fig_top.add_subplot(111)

    top_names = [item[0] for item in top_n]
    y_pos = np.arange(len(top_names))

    for i, model in enumerate(ordered_models):
        subtasks = results[model].get(language, {}).get('subtasks', {})
        scores = [subtasks.get(subtask, 0) for subtask in top_names]

        offset = (i - len(models_sorted)/2) * bar_width
        ax_top.barh(y_pos + offset, scores, bar_width,
                    label=get_nice_model_name(model),
                    color=get_model_color(model),
                    edgecolor='black', linewidth=0.5, alpha=0.9)

    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels([f"{name.replace('_', ' ').title()[:40]} ({stats['delta']:+.3f})"
                            for name, stats in top_n], fontsize=10)
    ax_top.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax_top.set_title(f'Top {n} MMLU Subtasks - {lang_display}', fontsize=16, fontweight='bold', pad=15)
    ax_top.set_xlim(0, 0.5)
    ax_top.grid(axis='x', alpha=0.3, linestyle='--')
    ax_top.legend(loc='lower right', fontsize=9)
    ax_top.invert_yaxis()

    plt.tight_layout()

    file_prefix = "06" if language == 'en' else "08"
    output_path = output_dir / f"{file_prefix}_top{n}_{language}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {lang_display} top plot: {output_path}")

    pdf_path = output_dir / f"{file_prefix}_top{n}_{language}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()

    # Create Bottom N plot
    fig_bottom = plt.figure(figsize=(12, 10))
    ax_bottom = fig_bottom.add_subplot(111)

    bottom_names = [item[0] for item in bottom_n]
    y_pos_bottom = np.arange(len(bottom_names))

    for i, model in enumerate(ordered_models):
        subtasks = results[model].get(language, {}).get('subtasks', {})
        scores = [subtasks.get(subtask, 0) for subtask in bottom_names]

        offset = (i - len(models_sorted)/2) * bar_width
        ax_bottom.barh(y_pos_bottom + offset, scores, bar_width,
                       label=get_nice_model_name(model),
                       color=get_model_color(model),
                       edgecolor='black', linewidth=0.5, alpha=0.9)

    ax_bottom.set_yticks(y_pos_bottom)
    ax_bottom.set_yticklabels([f"{name.replace('_', ' ').title()[:40]} ({stats['delta']:+.3f})"
                               for name, stats in bottom_n], fontsize=10)
    ax_bottom.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax_bottom.set_title(f'Bottom {n} MMLU Subtasks - {lang_display}', fontsize=16, fontweight='bold', pad=15)
    ax_bottom.set_xlim(0, 0.5)
    ax_bottom.grid(axis='x', alpha=0.3, linestyle='--')
    ax_bottom.legend(loc='lower right', fontsize=9)
    ax_bottom.invert_yaxis()

    plt.tight_layout()

    file_prefix = "07" if language == 'en' else "09"
    output_path = output_dir / f"{file_prefix}_bottom{n}_{language}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {lang_display} bottom plot: {output_path}")

    pdf_path = output_dir / f"{file_prefix}_bottom{n}_{language}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()


def create_combined_en_de_plot(results, output_dir, baseline_key, highlight_key,
                                improvements, all_subtasks, mode='top', n=10):
    """Create side-by-side EN and DE plots (Top or Bottom N)."""

    if mode not in ['top', 'bottom']:
        raise ValueError("mode must be 'top' or 'bottom'")

    mode_display = 'Top' if mode == 'top' else 'Bottom'

    def improvement_key(model):
        model_delta = improvements.get(model, {}).get('mean_delta', np.nan)
        return model_delta if not np.isnan(model_delta) else float('-inf')

    models_sorted = sorted(results.keys(), key=improvement_key, reverse=True)
    ordered_models = sorted(models_sorted, key=get_model_visual_priority)

    # Get subtasks for EN
    baseline_en_subtasks = results.get(baseline_key, {}).get('en', {}).get('subtasks', {})
    highlight_en_subtasks = results.get(highlight_key, {}).get('en', {}).get('subtasks', {})

    subtask_deltas_en = {}
    for subtask in all_subtasks:
        baseline_score = baseline_en_subtasks.get(subtask, np.nan)
        highlight_score = highlight_en_subtasks.get(subtask, np.nan)
        if not np.isnan(baseline_score) and not np.isnan(highlight_score):
            delta = highlight_score - baseline_score
            subtask_deltas_en[subtask] = {'baseline': baseline_score, 'highlight': highlight_score, 'delta': delta}

    sorted_en = sorted(subtask_deltas_en.items(), key=lambda x: x[1]['delta'], reverse=True)

    # Get subtasks for DE
    baseline_de_subtasks = results.get(baseline_key, {}).get('de', {}).get('subtasks', {})
    highlight_de_subtasks = results.get(highlight_key, {}).get('de', {}).get('subtasks', {})

    subtask_deltas_de = {}
    for subtask in all_subtasks:
        baseline_score = baseline_de_subtasks.get(subtask, np.nan)
        highlight_score = highlight_de_subtasks.get(subtask, np.nan)
        if not np.isnan(baseline_score) and not np.isnan(highlight_score):
            delta = highlight_score - baseline_score
            subtask_deltas_de[subtask] = {'baseline': baseline_score, 'highlight': highlight_score, 'delta': delta}

    sorted_de = sorted(subtask_deltas_de.items(), key=lambda x: x[1]['delta'], reverse=True)

    # Select top or bottom
    if mode == 'top':
        selected_en = sorted_en[:n]
        selected_de = sorted_de[:n]
    else:
        selected_en = sorted_en[-n:][::-1]
        selected_de = sorted_de[-n:][::-1]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    fig.suptitle(f'{mode_display} {n} MMLU Subtasks: English vs German',
                 fontsize=18, fontweight='bold', y=0.995)

    bar_width = 0.12

    # English plot
    en_names = [item[0] for item in selected_en]
    y_pos_en = np.arange(len(en_names))

    for i, model in enumerate(ordered_models):
        subtasks = results[model].get('en', {}).get('subtasks', {})
        scores = [subtasks.get(subtask, 0) for subtask in en_names]

        offset = (i - len(models_sorted)/2) * bar_width
        ax1.barh(y_pos_en + offset, scores, bar_width,
                label=get_nice_model_name(model),
                color=get_model_color(model),
                edgecolor='black', linewidth=0.5, alpha=0.9)

    ax1.set_yticks(y_pos_en)
    ax1.set_yticklabels([f"{name.replace('_', ' ').title()[:40]} ({stats['delta']:+.3f})"
                         for name, stats in selected_en], fontsize=10)
    ax1.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title(f'{mode_display} {n} - English', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlim(0, 0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()

    # German plot
    de_names = [item[0] for item in selected_de]
    y_pos_de = np.arange(len(de_names))

    for i, model in enumerate(ordered_models):
        subtasks = results[model].get('de', {}).get('subtasks', {})
        scores = [subtasks.get(subtask, 0) for subtask in de_names]

        offset = (i - len(models_sorted)/2) * bar_width
        ax2.barh(y_pos_de + offset, scores, bar_width,
                label=get_nice_model_name(model),
                color=get_model_color(model),
                edgecolor='black', linewidth=0.5, alpha=0.9)

    ax2.set_yticks(y_pos_de)
    ax2.set_yticklabels([f"{name.replace('_', ' ').title()[:40]} ({stats['delta']:+.3f})"
                         for name, stats in selected_de], fontsize=10)
    ax2.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'{mode_display} {n} - German', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlim(0, 0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    # Create centered legend between both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               fontsize=13, ncol=6, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.99])

    file_num = "10" if mode == 'top' else "11"
    output_path = output_dir / f"{file_num}_{mode}{n}_en_de_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined EN+DE {mode} plot: {output_path}")

    pdf_path = output_dir / f"{file_num}_{mode}{n}_en_de_combined.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()


def create_detailed_subtask_csv(results, output_dir, baseline_key, all_subtasks, summary_models):
    """Create detailed CSV with all subtask scores for all models."""

    baseline_en_subtasks = results.get(baseline_key, {}).get('en', {}).get('subtasks', {})
    baseline_de_subtasks = results.get(baseline_key, {}).get('de', {}).get('subtasks', {})

    detailed_data = []

    for subtask in sorted(all_subtasks):
        for model in summary_models:
            en_subtasks = results[model].get('en', {}).get('subtasks', {})
            de_subtasks = results[model].get('de', {}).get('subtasks', {})

            en_score = en_subtasks.get(subtask, np.nan)
            de_score = de_subtasks.get(subtask, np.nan)

            baseline_en_score = baseline_en_subtasks.get(subtask, np.nan)
            baseline_de_score = baseline_de_subtasks.get(subtask, np.nan)

            en_delta = en_score - baseline_en_score if not np.isnan(en_score) and not np.isnan(baseline_en_score) else np.nan
            de_delta = de_score - baseline_de_score if not np.isnan(de_score) and not np.isnan(baseline_de_score) else np.nan

            valid_deltas = [d for d in [en_delta, de_delta] if not np.isnan(d)]
            mean_delta = np.mean(valid_deltas) if valid_deltas else np.nan

            valid_scores = [s for s in [en_score, de_score] if not np.isnan(s)]
            mean_score = np.mean(valid_scores) if valid_scores else np.nan

            row = {
                'subtask': subtask,
                'model': get_nice_model_name(model),
                'en_score': en_score,
                'de_score': de_score,
                'mean_score': mean_score,
                'en_delta_vs_t5_base': en_delta,
                'de_delta_vs_t5_base': de_delta,
                'mean_delta_vs_t5_base': mean_delta,
            }
            detailed_data.append(row)

    df_detailed = pd.DataFrame(detailed_data)
    csv_path = output_dir / "04_detailed_subtask_scores.csv"
    df_detailed.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved detailed subtask CSV: {csv_path}")
    print(f"  Total rows: {len(detailed_data)} (subtasks: {len(all_subtasks)}, models: {len(summary_models)})")


def create_ranked_subtask_csv(results, output_dir, baseline_key, highlight_key, all_subtasks, summary_models):
    """Create single ranked CSV (best to worst) based on highlight model improvement."""

    baseline_en_subtasks = results.get(baseline_key, {}).get('en', {}).get('subtasks', {})
    baseline_de_subtasks = results.get(baseline_key, {}).get('de', {}).get('subtasks', {})

    highlight_en_subtasks = results.get(highlight_key, {}).get('en', {}).get('subtasks', {})
    highlight_de_subtasks = results.get(highlight_key, {}).get('de', {}).get('subtasks', {})

    ranked_data = []

    for subtask in all_subtasks:
        baseline_en_score = baseline_en_subtasks.get(subtask, np.nan)
        baseline_de_score = baseline_de_subtasks.get(subtask, np.nan)

        highlight_en_score = highlight_en_subtasks.get(subtask, np.nan)
        highlight_de_score = highlight_de_subtasks.get(subtask, np.nan)

        en_delta = highlight_en_score - baseline_en_score if not np.isnan(highlight_en_score) and not np.isnan(baseline_en_score) else np.nan
        de_delta = highlight_de_score - baseline_de_score if not np.isnan(highlight_de_score) and not np.isnan(baseline_de_score) else np.nan

        valid_deltas = [d for d in [en_delta, de_delta] if not np.isnan(d)]
        mean_delta = np.mean(valid_deltas) if valid_deltas else np.nan

        row = {
            'rank': 0,
            'subtask': subtask,
            'highlight_model_delta': mean_delta,
        }

        for model in summary_models:
            en_subtasks = results[model].get('en', {}).get('subtasks', {})
            de_subtasks = results[model].get('de', {}).get('subtasks', {})

            en_score = en_subtasks.get(subtask, np.nan)
            de_score = de_subtasks.get(subtask, np.nan)

            valid_scores = [s for s in [en_score, de_score] if not np.isnan(s)]
            mean_score = np.mean(valid_scores) if valid_scores else np.nan

            model_name = get_nice_model_name(model)
            row[f'{model_name}_EN'] = en_score
            row[f'{model_name}_DE'] = de_score
            row[f'{model_name}_Mean'] = mean_score

        ranked_data.append(row)

    df_ranked = pd.DataFrame(ranked_data)
    df_ranked = df_ranked.sort_values('highlight_model_delta', ascending=False, na_position='last')
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    cols = ['rank', 'subtask', 'highlight_model_delta'] + [col for col in df_ranked.columns if col not in ['rank', 'subtask', 'highlight_model_delta']]
    df_ranked = df_ranked[cols]

    csv_path = output_dir / "05_ranked_subtasks_best_to_worst.csv"
    df_ranked.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved ranked subtasks CSV (best to worst): {csv_path}")
    print(f"  Sorted by: {get_nice_model_name(highlight_key)} improvement vs. {get_nice_model_name(baseline_key)}")


def create_comprehensive_analysis(eval_run_dir, experiment_name):
    """Create all analysis plots and save to organized directory."""

    # Load results
    print(f"Loading evaluation results for: {experiment_name}")
    results = load_evaluation_results(eval_run_dir, experiment_name)

    if not results:
        raise ValueError("No results found")

    print(f"Found {len(results)} models")

    baseline_key = identify_baseline_model(results)
    highlight_key = identify_highlight_model(results)

    if baseline_key is None:
        raise ValueError("Could not identify T5-Base baseline model in results")

    if highlight_key is None:
        raise ValueError("Could not identify Wechsel + 15k DE model in results")

    improvements = compute_model_improvements(results, baseline_key)
    all_subtasks = collect_unique_subtasks(results)

    print(f"Baseline model: {get_nice_model_name(baseline_key)}")
    print(f"Highlight model: {get_nice_model_name(highlight_key)}")
    print(f"Total unique subtasks (EN ∪ DE): {len(all_subtasks)}")

    # Create output directory
    output_dir = Path("evaluation_results") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create plots
    print("\n1. Creating main comparison plot...")
    create_main_comparison_plot(results, output_dir, baseline_key, improvements)

    print("\n2. Creating improved subtask comparison plot...")
    improved_subtasks = create_subtask_comparison_plot(results, output_dir, baseline_key, highlight_key,
                                                       improvements, all_subtasks, mode='improved')

    print("\n3. Creating regressed subtask comparison plot...")
    create_subtask_comparison_plot(results, output_dir, baseline_key, highlight_key,
                                   improvements, all_subtasks, mode='regressed')

    print("\n4. Creating English Top 10 / Bottom 10 plot...")
    create_top_bottom_plot_separate(results, output_dir, baseline_key, highlight_key,
                                    improvements, all_subtasks, language='en', n=10)

    print("\n5. Creating German Top 10 / Bottom 10 plot...")
    create_top_bottom_plot_separate(results, output_dir, baseline_key, highlight_key,
                                    improvements, all_subtasks, language='de', n=10)

    print("\n6. Creating combined EN vs DE Top 10 plot...")
    create_combined_en_de_plot(results, output_dir, baseline_key, highlight_key,
                               improvements, all_subtasks, mode='top', n=10)

    print("\n7. Creating combined EN vs DE Bottom 10 plot...")
    create_combined_en_de_plot(results, output_dir, baseline_key, highlight_key,
                               improvements, all_subtasks, mode='bottom', n=10)

    # Create summary CSV
    print("\n8. Creating summary CSV...")
    summary_data = []

    def summary_improvement_key(model):
        model_delta = improvements.get(model, {}).get('mean_delta', np.nan)
        return model_delta if not np.isnan(model_delta) else float('-inf')

    summary_models = sorted(results.keys(), key=summary_improvement_key, reverse=True)

    for model in summary_models:
        row = {
            'model': get_nice_model_name(model),
            'en_overall': results[model]['en'].get('overall', np.nan),
            'de_overall': results[model]['de'].get('overall', np.nan),
            'en_overall_delta_vs_t5_base': improvements.get(model, {}).get('delta_en', np.nan),
            'de_overall_delta_vs_t5_base': improvements.get(model, {}).get('delta_de', np.nan),
            'mean_overall_delta_vs_t5_base': improvements.get(model, {}).get('mean_delta', np.nan),
        }
        for cat in MMLU_CATEGORIES:
            row[f'en_{cat.lower().replace(" ", "_")}'] = results[model]['en'].get(cat, np.nan)
            row[f'de_{cat.lower().replace(" ", "_")}'] = results[model]['de'].get(cat, np.nan)
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    csv_path = output_dir / "00_summary_scores.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved summary CSV: {csv_path}")

    # Create detailed subtask CSV
    print("\n9. Creating detailed subtask CSV...")
    create_detailed_subtask_csv(results, output_dir, baseline_key, all_subtasks, summary_models)

    # Create ranked subtask CSV (best to worst)
    print("\n10. Creating ranked subtask CSV (best to worst)...")
    create_ranked_subtask_csv(results, output_dir, baseline_key, highlight_key, all_subtasks, summary_models)

    print(f"\n✓ All visualizations saved to: {output_dir}")
    return output_dir


def main():
    if len(sys.argv) < 3:
        print("Usage: python create_crosslingual_comparison_plot_v2.py <eval_run_dir> <experiment_name>")
        print("Example: python create_crosslingual_comparison_plot_v2.py logs/eval_pipeline/runs/2025-09-23_19-00-53 scientific_crosslingual_transfer_eval_full_15k")
        sys.exit(1)

    eval_run_dir = sys.argv[1]
    experiment_name = sys.argv[2]

    if not Path(eval_run_dir).exists():
        print(f"Error: Directory not found: {eval_run_dir}")
        sys.exit(1)

    try:
        output_dir = create_comprehensive_analysis(eval_run_dir, experiment_name)
        print(f"\n✓ Success! All plots created in: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
