#!/usr/bin/env python3
"""
Aggregate and visualize MMLU per‑subtask accuracies across multiple models.

Input: a directory containing evaluation JSON files produced by lm‑evaluation‑harness.
Output: interactive Plotly HTML (and PNG if kaleido installed) showing grouped bars per subtask
        for both languages (EN/DE) when available, plus a CSV export of the aggregated table.

Usage:
  python evaluation/analysis_scripts/plot_mmlu_subtasks_across_models.py <results_dir> \
         [--model-alias "substr=Display Name" ...] [--out <output_dir>]

Examples:
  python evaluation/analysis_scripts/plot_mmlu_subtasks_across_models.py \
      cross_lingual_transfer/evaluation/results \
      --model-alias wechsel-transfer=Wechsel‑15k \
      --model-alias german-native=Native‑15k \
      --model-alias germanT5=HF‑GermanT5 \
      --model-alias clean_restart=EN‑Gold

Notes:
 - The script searches JSON recursively and supports both Global‑MMLU format keys
   (global_mmlu_full_en / global_mmlu_full_de and their per‑subject entries)
   and will extract 'acc,none' and 'acc_stderr,none' when available.
 - Model display names are derived from file paths but can be overridden with --model-alias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_all_results(results_dir: Path) -> List[Tuple[Path, Dict]]:
    items: List[Tuple[Path, Dict]] = []
    for fp in results_dir.rglob("*.json"):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            # Expect lm-eval structure with top-level 'results'
            if isinstance(data, dict) and "results" in data:
                items.append((fp, data["results"]))
        except Exception:
            # Skip broken files silently
            continue
    return items


def derive_model_name_from_json(json_path: Path) -> str:
    """Derive a compact display name for the model from the lm-eval output path.

    Expected layout (as seen in this repo):
      .../<experiment>/<model_label>_global_mmlu_<lang>_0shot_<ts>/<sanitized_model_dir>/results_....json
    We extract the `<model_label>` portion (prefix before `_global_mmlu_`).
    Fallbacks keep the parent directory name.
    """
    try:
        # parent: <sanitized_model_dir>
        # parent.parent: <model_label>_global_mmlu_<lang>_0shot_<ts>
        label_dir = json_path.parent.parent.name
        if "_global_mmlu_" in label_dir:
            model_label = label_dir.split("_global_mmlu_")[0]
            return model_label
        return label_dir
    except Exception:
        return json_path.parent.name


def apply_alias(name: str, aliases: Dict[str, str]) -> str:
    for needle, alias in aliases.items():
        if needle.lower() in name.lower():
            return alias
    # Built-in shorteners for common labels in this project
    builtin = {
        "wechsel-transfer-continued-15k": "Wechsel-15k",
        "german-native-baseline-15k": "Native-15k",
        "hf-germanT5-base": "HF-GermanT5",
        "wechsel-init-clean-restart-487k": "Wechsel-init",
    }
    return builtin.get(name, name)


def extract_global_mmlu_subjects(results: Dict, lang: str) -> Dict[str, Dict[str, float]]:
    """Return mapping subject -> {acc, acc_stderr} for given language (en/de).

    Expected keys like 'global_mmlu_full_en_<subject>' or 'global_mmlu_full_de_<subject>'
    with nested metrics dicts containing 'acc,none' and 'acc_stderr,none'.
    """
    prefix = f"global_mmlu_full_{lang}_"
    out: Dict[str, Dict[str, float]] = {}
    for key, val in results.items():
        if not key.startswith(prefix):
            continue
        if key in {
            f"global_mmlu_full_{lang}",
            f"global_mmlu_full_{lang}_humanities",
            f"global_mmlu_full_{lang}_social_sciences",
            f"global_mmlu_full_{lang}_stem",
            f"global_mmlu_full_{lang}_other",
        }:
            continue
        if isinstance(val, dict):
            acc = val.get("acc,none")
            acc_stderr = val.get("acc_stderr,none")
            if acc is not None:
                subject = key.replace(prefix, "")
                out[subject] = {
                    "acc": float(acc),
                    "acc_stderr": float(acc_stderr) if acc_stderr is not None else np.nan,
                }
    return out


def aggregate_dataframe(items: List[Tuple[Path, Dict]], aliases: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for path, results in items:
        model_name_raw = derive_model_name_from_json(path)
        model_name = apply_alias(model_name_raw, aliases)
        for lang in ("en", "de"):
            subj_map = extract_global_mmlu_subjects(results, lang)
            for subject, metrics in subj_map.items():
                rows.append({
                    "model": model_name,
                    "language": lang,
                    "subject": subject,
                    "acc": metrics.get("acc", np.nan),
                    "acc_stderr": metrics.get("acc_stderr", np.nan),
                    "source_file": str(path),
                })
    df = pd.DataFrame(rows)
    return df


def plot_grouped_bars(
    df: pd.DataFrame,
    out_dir: Path,
    lang: str,
    with_errorbar: bool = False,
    engine: str = "mpl",
) -> None:
    lang_df = df[df["language"] == lang].copy()
    if lang_df.empty:
        print(f"⚠️  No data for language: {lang}")
        return

    # Sort subjects by average accuracy across models for a pleasant order
    order = (
        lang_df.groupby("subject")["acc"].mean().sort_values(ascending=True).index.tolist()
    )
    subjects = [s.replace("_", " ").title() for s in order]
    models = list(lang_df["model"].unique())

    fig = go.Figure()
    # Stable color mapping for known labels
    base_colors = {
        "Native-15k": "#F39C12",       # orange
        "HF-GermanT5": "#95A5A6",      # gray
        "Wechsel-15k": "#2ECC71",      # green
        "Wechsel-init": "#3498DB",     # blue
    }
    # Assign colors deterministically
    fallback_palette = ["#E74C3C", "#9B59B6", "#1ABC9C", "#8E44AD", "#34495E"]
    color_map: Dict[str, str] = {}
    for i, m in enumerate(models):
        color_map[m] = base_colors.get(m, fallback_palette[i % len(fallback_palette)])

    for m in models:
        mdf = lang_df[lang_df["model"] == m]
        # Align to ordered subjects
        accs = []
        errs = []
        for subj in order:
            row = mdf[mdf["subject"] == subj]
            if row.empty:
                accs.append(0.0)
                errs.append(0.0)
            else:
                accs.append(float(row.iloc[0]["acc"]))
                err = float(row.iloc[0]["acc_stderr"]) if pd.notna(row.iloc[0]["acc_stderr"]) else 0.0
                errs.append(err)
        bar_kwargs = dict(
            name=m,
            y=subjects,
            x=accs,
            orientation="h",
            marker=dict(color=color_map[m], line=dict(color="white", width=1)),
        )
        if with_errorbar:
            bar_kwargs["error_x"] = dict(type="data", array=errs, visible=True)
            bar_kwargs["hovertemplate"] = f"<b>%{{y}}</b><br>{m}: %{{x:.3f}} ± %{{error_x.array:.3f}}<extra></extra>"
        else:
            bar_kwargs["hovertemplate"] = f"<b>%{{y}}</b><br>{m}: %{{x:.3f}}<extra></extra>"
        fig.add_trace(go.Bar(**bar_kwargs))

    fig.update_layout(
        title=dict(
            text=f"Global MMLU {lang.upper()} — Per‑Subtask Accuracy (Grouped by Model)",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="Accuracy", range=[0, 1.0], showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Subtasks (sorted by avg)", automargin=True),
        barmode="group",
        bargap=0.18,
        bargroupgap=0.12,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(len(subjects) * 22, 900),
        width=1200,
        # More top space and move legend to bottom to avoid overlap with title
        margin=dict(l=220, r=80, t=120, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="right", x=1),
        hovermode="y unified",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"mmlu_{lang}_subtasks_grouped"
    html = str(base.with_suffix(".html"))
    png = str(base.with_suffix(".png"))
    fig.write_html(html)
    print(f"📊 Saved interactive chart: {html}")
    # Static export: prefer engine requested
    def export_with_plotly():
        fig.write_image(png, width=1200, height=max(len(subjects) * 22, 900))
        print(f"📊 Saved static chart (plotly+kaleido): {png}")

    def export_with_mpl():
        save_png_with_matplotlib(lang_df, subjects, order, models, color_map, png)
        print(f"📊 Saved static chart (matplotlib): {png}")

    if engine == "plotly":
        try:
            export_with_plotly()
        except Exception as e:
            print(f"⚠️  Plotly export failed: {e}\n   Trying Matplotlib fallback…")
            try:
                export_with_mpl()
            except Exception as e2:
                print(f"❌ Matplotlib fallback failed: {e2}\n   To use Plotly PNG export, install kaleido + Chrome: pip install kaleido && plotly_get_chrome")
    else:  # engine == 'mpl'
        try:
            export_with_mpl()
        except Exception as e:
            print(f"⚠️  Matplotlib export failed: {e}\n   Trying Plotly+kaleido fallback…")
            try:
                export_with_plotly()
            except Exception as e2:
                print(f"❌ Plotly fallback failed: {e2}\n   For Plotly: pip install kaleido && plotly_get_chrome")


def save_png_with_matplotlib(lang_df: pd.DataFrame, subjects_readable: List[str], order_keys: List[str], models: List[str], color_map: Dict[str, str], out_path: str):
    """Paper‑style PNG using matplotlib/seaborn as fallback (no error bars)."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Data matrix: rows=subjects, cols=models
    data = np.zeros((len(order_keys), len(models)))
    for i, subj in enumerate(order_keys):
        for j, m in enumerate(models):
            row = lang_df[(lang_df["subject"] == subj) & (lang_df["model"] == m)]
            data[i, j] = float(row.iloc[0]["acc"]) if not row.empty else 0.0

    # Figure size tuned for paper — 0.18in per subject row
    h = max(2.0 + 0.18 * len(order_keys), 6)
    w = 10
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)

    # Horizontal grouped bars
    y = np.arange(len(order_keys))
    total_width = 0.8
    bar_h = total_width / max(1, len(models))

    for j, m in enumerate(models):
        ax.barh(y + (j - (len(models)-1)/2) * bar_h, data[:, j], height=bar_h,
                color=color_map[m], label=m, edgecolor='white', linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(subjects_readable, fontsize=8)
    ax.set_xlabel('Accuracy', fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', linestyle='--', alpha=0.25)
    ax.set_title(f'Global MMLU {lang_df.iloc[0]["language"].upper()} — Per‑Subtask Accuracy', fontsize=12, pad=12)

    # Legend below plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=min(4, len(models)), frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot MMLU per‑subtask accuracies across models")
    ap.add_argument("results_dir", type=str, help="Directory containing evaluation JSONs (recursive)")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: evaluation_results/<name>")
    ap.add_argument("--with-errorbar", action="store_true", help="Show standard error bars (default: off)")
    ap.add_argument("--engine", choices=["mpl", "plotly"], default="mpl", help="Renderer for PNG export (default: mpl)")
    ap.add_argument(
        "--model-alias",
        action="append",
        default=[],
        help="Mapping substr=Alias to prettify model names; can be provided multiple times",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    # Parse aliases
    aliases: Dict[str, str] = {}
    for item in args.model_alias:
        if "=" in item:
            k, v = item.split("=", 1)
            aliases[k.strip()] = v.strip()

    items = load_all_results(results_dir)
    if not items:
        raise SystemExit("No evaluation JSONs with 'results' found")

    df = aggregate_dataframe(items, aliases)
    if df.empty:
        raise SystemExit("No Global MMLU per‑subject entries found in results")

    # Output directory
    if args.out is None:
        name = results_dir.name
        out_dir = Path("evaluation/evaluation_results") / name
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / "mmlu_per_subject_across_models.csv"
    df.sort_values(["language", "subject", "model"]).to_csv(csv_path, index=False)
    print(f"📋 Saved table: {csv_path}")

    # Plot per language when available
    for lang in ("de", "en"):
        plot_grouped_bars(df, out_dir, lang, with_errorbar=args.with_errorbar, engine=args.engine)


if __name__ == "__main__":
    main()
