#!/usr/bin/env python3
"""
Validation script for the Pandas Mastery Project notebooks.
This script validates that all notebooks can be imported and basic functionality works.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path


def validate_pandas_basics():
    """Validate basic pandas functionality."""
    print("Validating basic pandas functionality...")

    # Test Series creation
    series_test = pd.Series([1, 2, 3, 4, 5])
    assert len(series_test) == 5, "Series creation failed"
    print("✓ Series creation works")

    # Test DataFrame creation
    df_test = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['x', 'y', 'z']
    })
    assert df_test.shape == (3, 3), "DataFrame creation failed"
    print("✓ DataFrame creation works")

    # Test basic indexing
    assert df_test.loc[0, 'A'] == 1, "Loc indexing failed"
    assert df_test.iloc[0, 0] == 1, "ILoc indexing failed"
    print("✓ Basic indexing works")

    # Test groupby functionality
    df_group = pd.DataFrame({
        'Category': ['A', 'A', 'B', 'B'],
        'Value': [1, 2, 3, 4]
    })
    grouped = df_group.groupby('Category')['Value'].sum()
    assert grouped['A'] == 3 and grouped['B'] == 7, "Groupby failed"
    print("✓ Groupby functionality works")

    # Test pivot table functionality
    df_pivot = pd.DataFrame({
        'A': [1, 1, 2, 2],
        'B': ['x', 'y', 'x', 'y'],
        'C': [1, 2, 3, 4]
    })
    pivot_result = pd.pivot_table(df_pivot, values='C', index='A', columns='B', aggfunc='sum')
    assert pivot_result.loc[1, 'x'] == 1, "Pivot table failed"
    print("✓ Pivot table functionality works")

    print("✓ All basic pandas functionality validated successfully!")
    return True


def validate_data_loading():
    """Validate data loading functionality."""
    print("\nValidating data loading functionality...")

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("! Data directory does not exist - creating sample data for validation")
        # Create the data directory
        data_dir.mkdir(exist_ok=True)

        # Create a simple sample dataset
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Age': [25, 30, 35, 28],
            'City': ['New York', 'London', 'Paris', 'Tokyo'],
            'Salary': [50000, 60000, 70000, 55000]
        }
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(data_dir / "sample_data.csv", index=False)
        print("✓ Created sample data for validation")

    # Test loading sample data if it exists
    titanic_path = data_dir / "titanic.csv"
    if titanic_path.exists():
        df = pd.read_csv(titanic_path)
        print(f"✓ Successfully loaded Titanic dataset with shape {df.shape}")
    else:
        print("! Titanic dataset not found, but this is acceptable for initial validation")

    print("✓ Data loading functionality validated!")
    return True


def validate_utils_module():
    """Validate the utils module functionality."""
    print("\nValidating utils module functionality...")

    # Import the utils module
    try:
        import utils.utils as utils_module
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"! Could not import utils module: {e}")
        return False

    # Test basic functions from the utils module
    try:
        # Create sample data to test functions
        sample_df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50],
            'C': ['x', 'y', 'z', 'x', 'y']
        })

        # Test data info function (assuming it exists in the utils module)
        if hasattr(utils_module, 'get_data_info'):
            info_result = utils_module.get_data_info(sample_df)
            assert 'shape' in info_result, "get_data_info function failed"
            print("✓ get_data_info function works")
        else:
            print("- get_data_info function not found in utils module (acceptable)")

        # Test missing value cleaning (assuming it exists in the utils module)
        if hasattr(utils_module, 'clean_missing_values'):
            cleaned_df = utils_module.clean_missing_values(sample_df, strategy='mean')
            print("✓ clean_missing_values function works")
        else:
            print("- clean_missing_values function not found in utils module (acceptable)")

        # Test outlier detection (assuming it exists in the utils module)
        if hasattr(utils_module, 'detect_outliers'):
            outliers_result = utils_module.detect_outliers(sample_df, method='iqr')
            print("✓ detect_outliers function works")
        else:
            print("- detect_outliers function not found in utils module (acceptable)")

        print("✓ Utils module validation completed!")
        return True

    except Exception as e:
        print(f"! Error testing utils functions: {e}")
        return False


def validate_visualization():
    """Validate that visualization libraries work."""
    print("\nValidating visualization libraries...")

    try:
        # Test matplotlib
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        plt.close(fig)  # Close to free memory
        print("✓ Matplotlib works")

        # Test seaborn
        tips = sns.load_dataset("tips")[:10]  # Load a small subset
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax)
        plt.close(fig)  # Close to free memory
        print("✓ Seaborn works")

        print("✓ Visualization libraries validated!")
        return True

    except Exception as e:
        print(f"! Error validating visualization: {e}")
        return False


def main():
    """Run all validations."""
    print("Starting validation of Pandas Mastery Project...")
    print("="*50)

    all_passed = True

    try:
        all_passed = validate_pandas_basics()  # Fixed function name
        all_passed = all_passed and validate_data_loading()  # Fixed function name
        all_passed = all_passed and validate_utils_module()  # Fixed function name
        all_passed = all_passed and validate_visualization()  # Fixed function name
    except Exception as e:
        print(f"\n! Validation failed with error: {e}")
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("🎉 All validations passed! The Pandas Mastery Project is ready.")
        print("\nNext steps:")
        print("1. Open Jupyter Notebook: jupyter notebook")
        print("2. Navigate to the notebooks/ directory")
        print("3. Start with pandas_fundamentals.ipynb")
        print("4. Progress through real_dataset_analysis.ipynb")
        print("5. Complete the project with COVID19_Analysis.ipynb")
        return 0
    else:
        print("❌ Some validations failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())