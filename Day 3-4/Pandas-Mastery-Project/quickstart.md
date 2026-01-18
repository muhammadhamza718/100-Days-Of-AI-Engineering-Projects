# Quickstart Guide: Pandas Mastery Project

Welcome to the Pandas Mastery Project! This guide will help you set up your environment and get started with the educational notebooks.

## Prerequisites

- Python 3.8 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Setup Instructions

### 1. Clone or Navigate to the Project Directory

```bash
cd Pandas-Mastery-Project
```

### 2. Install Dependencies

Using UV (recommended):
```bash
uv pip install pandas matplotlib seaborn jupyter
```

Or using pip:
```bash
pip install pandas matplotlib seaborn jupyter
```

### 3. Activate Virtual Environment (if created)

If you created a virtual environment:
```bash
# On Windows
source .venv/Scripts/activate

# On macOS/Linux
source .venv/bin/activate
```

### 4. Verify Installation

Start a Jupyter notebook to verify that everything is working:

```bash
jupyter notebook
```

Or run a simple test in Python:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("Seaborn version:", sns.__version__)

# Create a simple test DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
```

## Running the Notebooks

The project includes three main notebooks organized by learning objectives:

### 1. Pandas Fundamentals (`notebooks/pandas_fundamentals.ipynb`)
- Learn basic Series and DataFrame operations
- Practice indexing with `loc` and `iloc`
- Work with data cleaning techniques

### 2. Real Dataset Analysis (`notebooks/real_dataset_analysis.ipynb`)
- Apply your skills to the Titanic dataset in the `data/` directory
- Practice data inspection, cleaning, and analysis
- Answer specific questions about the data

### 3. COVID-19 Dashboard (`notebooks/COVID19_Analysis.ipynb`)
- Build a comprehensive data analysis project
- Practice all skills learned in a real-world scenario
- Create visualizations and draw insights

## Data Directory

Sample datasets are available in the `data/` directory:
- `titanic.csv` - Titanic passenger data for analysis exercises

## Utilities

Helper functions are available in the `utils/` directory:
- `utils/utils.py` - Contains data processing and validation functions
- Functions for loading data, cleaning missing values, and validation

## Troubleshooting

- If you encounter memory issues with large datasets, restart the kernel and run cells sequentially
- For missing data errors, ensure the required datasets are in the `data/` directory
- For visualization issues, check that matplotlib and seaborn are properly installed

## Next Steps

1. Start with `notebooks/pandas_fundamentals.ipynb` to learn the basics
2. Move to `notebooks/real_dataset_analysis.ipynb` for practical application
3. Complete the project with `notebooks/COVID19_Analysis.ipynb` for a comprehensive challenge