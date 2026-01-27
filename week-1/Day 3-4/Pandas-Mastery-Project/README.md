# Pandas Mastery Project

This project is designed to help you master Pandas for data science applications. It includes educational notebooks covering fundamental concepts, real-world data analysis exercises, and a comprehensive COVID-19 dashboard project.

## Overview

The project consists of three main learning modules:

1. **Pandas Fundamentals** (`notebooks/pandas_fundamentals.ipynb`) - Learn core concepts of Pandas including Series and DataFrame creation, indexing, data cleaning, and aggregation methods.
2. **Real Dataset Analysis** (`notebooks/real_dataset_analysis.ipynb`) - Apply your skills to a real-world dataset (Titanic) with data inspection, cleaning, and analysis.
3. **COVID-19 Dashboard** (`notebooks/COVID19_Analysis.ipynb`) - Build a comprehensive data analysis project with time-series data, grouping, and visualization.

## Prerequisites

- Python 3.8+
- Required packages: pandas, matplotlib, seaborn, jupyter
- Sample datasets in the `data/` directory

## Setup

1. Install the required dependencies:
   ```bash
   pip install pandas matplotlib seaborn jupyter
   ```

2. Or if using uv:
   ```bash
   uv pip install pandas matplotlib seaborn jupyter
   ```

## Usage

1. Start Jupyter notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory and open the notebooks in the following order:
   - `pandas_fundamentals.ipynb` - Start here to learn basic concepts
   - `real_dataset_analysis.ipynb` - Apply skills to real-world data
   - `COVID19_Analysis.ipynb` - Complete the comprehensive project

## Directory Structure

```
Pandas-Mastery-Project/
├── notebooks/                 # Educational notebooks
│   ├── pandas_fundamentals.ipynb
│   ├── real_dataset_analysis.ipynb
│   └── COVID19_Analysis.ipynb
├── data/                      # Sample datasets (e.g., titanic.csv)
├── utils/                     # Utility functions for data processing
│   └── utils.py
├── quickstart.md              # Setup and usage instructions
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── .venv/                     # Virtual environment (if created)
```

## Key Learning Objectives

- Creating and manipulating Series and DataFrames
- Data cleaning techniques (NaN handling, type conversion)
- GroupBy operations, Pivot Tables, and Merging
- Skill in visualizing trends from time-series data
- Applying Pandas to real-world data analysis problems

## Utilities

The `utils/utils.py` file contains helper functions for:
- Data loading and inspection
- Missing value handling
- Outlier detection
- Data validation

## Contributing

Feel free to extend the notebooks with additional examples or create new analysis notebooks following the same patterns.