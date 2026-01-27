# Quickstart Guide: Pandas Mastery Module

## Prerequisites

- Python 3.x installed
- Jupyter Notebook or VS Code with Jupyter extension
- Package manager: uv (or pip)

## Setup Instructions

1. Clone or navigate to the project directory
2. Install dependencies using uv:
   ```bash
   uv pip install pandas matplotlib seaborn
   ```
   OR if using pip:
   ```bash
   pip install pandas matplotlib seaborn jupyter
   ```

3. Verify installation:
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   print("Dependencies installed successfully!")
   ```

## Running the Notebooks

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the notebooks in order:
   - First: `pandas_fundamentals.ipynb` (User Story 1 - Foundations)
   - Second: `real_dataset_analysis.ipynb` (User Story 2 - Practical Application)
   - Third: `COVID19_Analysis.ipynb` (User Story 3 - Capstone Project)

## Expected Execution Time

- Pandas Fundamentals: 30-45 minutes
- Real Dataset Analysis: 45-60 minutes
- COVID-19 Dashboard: 60-90 minutes

## Troubleshooting

- If you encounter memory issues with large datasets, restart the kernel and run cells sequentially
- For missing data errors, ensure the required datasets are in the `data/` directory
- For visualization issues, check that matplotlib and seaborn are properly installed

## Key Learning Objectives

By completing these notebooks, you will master:
- Creating and manipulating Series and DataFrames
- Data cleaning techniques (handling NaN values, type conversion)
- GroupBy operations, Pivot Tables, and Merging
- Visualizing trends from time-series data