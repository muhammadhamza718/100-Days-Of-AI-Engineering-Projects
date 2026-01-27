"""
Visualization Gallery demonstrating matplotlib, seaborn, and plotly capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_data():
    """Create sample data for visualization examples"""
    n_samples = 100

    # Create sample dataset
    data = {
        'x': np.linspace(0, 10, n_samples),
        'y': np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'size_var': np.random.uniform(10, 100, n_samples),
        'values': np.random.normal(50, 15, n_samples)
    }

    df = pd.DataFrame(data)
    return df

def matplotlib_examples(df):
    """Create matplotlib visualizations"""
    print("Creating matplotlib visualizations...")

    # Line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['x'], df['y'], label='Sin Wave with Noise')
    ax.set_title('Matplotlib Line Plot: Sin Wave with Noise')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('matplotlib_line_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['x'], df['y'], c=df['values'], s=df['size_var'], alpha=0.6, cmap='viridis')
    ax.set_title('Matplotlib Scatter Plot: Colored by Values')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    plt.colorbar(scatter)
    ax.legend(*scatter.legend_elements(), title="Size")
    plt.tight_layout()
    plt.savefig('matplotlib_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['values'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title('Matplotlib Histogram: Distribution of Values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('matplotlib_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def seaborn_examples(df):
    """Create seaborn visualizations"""
    print("Creating seaborn visualizations...")

    # Set style
    sns.set_style("whitegrid")

    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='category', y='values')
    plt.title('Seaborn Box Plot: Values by Category')
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.savefig('seaborn_box_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='category', y='values')
    plt.title('Seaborn Violin Plot: Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.savefig('seaborn_violin_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Heatmap
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Seaborn Heatmap: Correlation Matrix')
    plt.tight_layout()
    plt.savefig('seaborn_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plotly_examples(df):
    """Create plotly visualizations"""
    print("Creating plotly visualizations...")

    # Interactive line plot
    fig = px.line(df, x='x', y='y', title='Plotly Interactive Line Plot: Sin Wave with Noise')
    fig.update_layout(xaxis_title='X Values', yaxis_title='Y Values')
    fig.write_image("plotly_line_plot.png", width=800, height=600, scale=2)
    fig.show()

    # Interactive scatter plot
    fig = px.scatter(df, x='x', y='y', color='category', size='size_var',
                     title='Plotly Interactive Scatter Plot: Colored by Category')
    fig.update_layout(xaxis_title='X Values', yaxis_title='Y Values')
    fig.write_image("plotly_scatter_plot.png", width=800, height=600, scale=2)
    fig.show()

    # Interactive histogram
    fig = px.histogram(df, x='values', nbins=30, title='Plotly Interactive Histogram: Distribution of Values')
    fig.update_layout(xaxis_title='Value', yaxis_title='Count')
    fig.write_image("plotly_histogram.png", width=800, height=600, scale=2)
    fig.show()

    # 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='values', color='category',
                        title='Plotly 3D Scatter Plot')
    fig.update_layout(scene=dict(
        xaxis_title='X Values',
        yaxis_title='Y Values',
        zaxis_title='Values'))
    fig.write_image("plotly_3d_scatter.png", width=800, height=600, scale=2)
    fig.show()

def main():
    """Main function to run all visualization examples"""
    print("Starting Visualization Gallery...")

    # Create sample data
    df = create_sample_data()

    # Create visualizations with each library
    matplotlib_examples(df)
    seaborn_examples(df)
    plotly_examples(df)

    print("Visualization Gallery completed!")
    print("PNG files saved in the current directory.")

if __name__ == "__main__":
    main()