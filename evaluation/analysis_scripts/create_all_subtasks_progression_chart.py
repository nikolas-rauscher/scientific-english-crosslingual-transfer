#!/usr/bin/env python3
"""
Create a training progression line chart for all MMLU subtasks over steps.
Shows how each subtask performance evolves during training.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np

def load_progression_data(csv_path):
    """Load the progression data from CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    
    # Extract step numbers from column headers
    steps = []
    for col in df.columns:
        if col != 'AVG':
            # Convert format like '105k' to 105000
            step_str = col.replace('k', '000')
            steps.append(int(step_str))
    
    return df, steps

def create_progression_chart(df, steps, output_path):
    """Create an interactive line chart showing all subtasks progression."""
    
    # Filter out AVG column and get task names
    df_filtered = df.drop(columns=['AVG'])
    task_names = df_filtered.index.tolist()
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for better visibility
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Add a trace for each subtask
    for i, task in enumerate(task_names):
        values = df_filtered.loc[task].values
        
        # Extract average from task name if present
        avg_match = task.split('(Avg: ')
        if len(avg_match) > 1:
            baseline_avg = float(avg_match[1].rstrip(')'))
            display_name = avg_match[0].strip()
        else:
            baseline_avg = None
            display_name = task
        
        # Determine if this is a main category or subtask
        is_main_category = task in ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)']
        
        # Set line properties based on task type
        if is_main_category:
            line_width = 3
            line_dash = 'solid'
            opacity = 1.0
        else:
            line_width = 1
            line_dash = 'solid'
            opacity = 0.6
        
        color_idx = i % len(colors)
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            mode='lines+markers' if is_main_category else 'lines',
            name=display_name[:40] + '...' if len(display_name) > 40 else display_name,
            line=dict(
                color=colors[color_idx],
                width=line_width,
                dash=line_dash
            ),
            opacity=opacity,
            marker=dict(size=3) if is_main_category else dict(size=0),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Step: %{x:,}<br>' +
                          'Score: %{y:.4f}<br>' +
                          '<extra></extra>',
            visible=True if is_main_category else 'legendonly'  # Hide subtasks by default
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'MMLU Subtasks Training Progression<br><sub>Click legend items to show/hide tasks</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Training Steps',
            tickformat=',',
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title='Accuracy',
            tickformat='.2%',
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            range=[0, 1]
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=10),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=8),
            itemsizing='constant'
        ),
        width=1400,
        height=800,
        margin=dict(r=250)  # Extra margin for legend
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.05
        )
    )
    
    # Save interactive HTML
    html_output = output_path.replace('.png', '_interactive.html')
    fig.write_html(html_output)
    print(f"Saved interactive chart to: {html_output}")
    
    # Save static PNG (skip if Chrome not available)
    try:
        fig.write_image(output_path, width=1400, height=800)
        print(f"Saved static chart to: {output_path}")
    except Exception as e:
        print(f"Could not save PNG (Chrome required): {e}")
    
    return fig

def create_category_chart(df, steps, output_path):
    """Create a cleaner chart focusing on main categories."""
    
    # Define categories and their tasks
    categories = {
        'STEM': [
            'High School Biology', 'High School Chemistry', 'High School Computer Science',
            'High School Mathematics', 'High School Physics', 'High School Statistics',
            'Abstract Algebra', 'Astronomy', 'College Biology', 'College Chemistry',
            'College Computer Science', 'College Mathematics', 'College Physics',
            'Computer Security', 'Conceptual Physics', 'Electrical Engineering',
            'Elementary Mathematics', 'Machine Learning'
        ],
        'Humanities': [
            'High School European History', 'High School US History', 'High School World History',
            'Logical Fallacies', 'Moral Disputes', 'Moral Scenarios', 'Philosophy',
            'Prehistory', 'Professional Law', 'World Religions'
        ],
        'Social Sciences': [
            'High School Geography', 'High School Government and Politics',
            'High School Macroeconomics', 'High School Microeconomics', 'High School Psychology',
            'Econometrics', 'Human Sexuality', 'Professional Psychology', 'Public Relations',
            'Security Studies', 'Sociology', 'US Foreign Policy'
        ],
        'Other': [
            'Anatomy', 'Business Ethics', 'Clinical Knowledge', 'College Medicine',
            'Global Facts', 'Human Aging', 'Management', 'Marketing', 'Medical Genetics',
            'Miscellaneous', 'Nutrition', 'Professional Accounting', 'Professional Medicine',
            'Virology'
        ]
    }
    
    # Calculate category averages
    category_data = {}
    df_filtered = df.drop(columns=['AVG'])
    
    for category, tasks in categories.items():
        category_values = []
        for step_idx in range(len(steps)):
            step_values = []
            for task in tasks:
                # Find matching task in dataframe
                for idx in df_filtered.index:
                    if task.lower() in idx.lower():
                        step_values.append(df_filtered.iloc[df_filtered.index.get_loc(idx), step_idx])
                        break
            if step_values:
                category_values.append(np.mean(step_values))
            else:
                category_values.append(0)
        category_data[category] = category_values
    
    # Create figure
    fig = go.Figure()
    
    # Colors for categories
    category_colors = {
        'STEM': '#2E7D32',  # Green
        'Humanities': '#1565C0',  # Blue
        'Social Sciences': '#E65100',  # Orange
        'Other': '#6A1B9A',  # Purple
        'MMLU OFFICIAL': '#D32F2F',  # Red
        'SIMPLE AVERAGE': '#424242'  # Gray
    }
    
    # Add category lines
    for category, values in category_data.items():
        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            mode='lines+markers',
            name=f'{category} Average',
            line=dict(
                color=category_colors[category],
                width=2.5
            ),
            marker=dict(size=5),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Step: %{x:,}<br>' +
                          'Score: %{y:.4f}<br>' +
                          '<extra></extra>'
        ))
    
    # Add MMLU OFFICIAL and SIMPLE AVERAGE
    for task in ['MMLU OFFICIAL', 'SIMPLE AVERAGE (All Tasks)']:
        if task in df.index:
            values = df_filtered.loc[task].values
            name = 'MMLU Official' if 'OFFICIAL' in task else 'Simple Average'
            color_key = 'MMLU OFFICIAL' if 'OFFICIAL' in task else 'SIMPLE AVERAGE'
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(
                    color=category_colors[color_key],
                    width=3,
                    dash='dash' if 'SIMPLE' in task else 'solid'
                ),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Step: %{x:,}<br>' +
                              'Score: %{y:.4f}<br>' +
                              '<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'MMLU Category Progression During Training',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Training Steps',
            tickformat=',',
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title='Accuracy',
            tickformat='.1%',
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            range=[0.15, 0.5]
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        width=1200,
        height=700
    )
    
    # Save
    category_output = output_path.replace('.png', '_categories.html')
    fig.write_html(category_output)
    print(f"Saved category chart to: {category_output}")
    
    # Save static PNG (skip if Chrome not available)
    try:
        category_png = output_path.replace('.png', '_categories.png')
        fig.write_image(category_png, width=1200, height=700)
        print(f"Saved category PNG to: {category_png}")
    except Exception as e:
        print(f"Could not save category PNG (Chrome required): {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_all_subtasks_progression_chart.py <csv_path> [output_dir]")
        print("Example: python create_all_subtasks_progression_chart.py evaluation_results/pretraining-logs-lr-001-OPTIMIZED-clean-restart/progression_data.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Determine output directory
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = csv_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {csv_path}")
    df, steps = load_progression_data(csv_path)
    
    print(f"Found {len(df)} tasks across {len(steps)} checkpoints")
    print(f"Step range: {min(steps):,} to {max(steps):,}")
    
    # Create charts
    output_path = output_dir / "all_subtasks_progression.png"
    create_progression_chart(df, steps, str(output_path))
    
    # Create category-focused chart
    create_category_chart(df, steps, str(output_path))
    
    print("\nCharts created successfully!")

if __name__ == "__main__":
    main()