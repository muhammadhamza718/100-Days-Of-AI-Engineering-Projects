"""
Utility functions for data processing in the Pandas Mastery Project.
"""

import pandas as pd
import numpy as np
from typing import Union


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


def clean_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: list = None) -> pd.DataFrame:
    """
    Clean missing values from a DataFrame using specified strategy.

    Args:
        df: Input DataFrame
        strategy: How to handle missing values ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill')
        columns: Specific columns to clean; if None, clean all columns with missing values

    Returns:
        Cleaned DataFrame
    """
    df_cleaned = df.copy()

    if columns is None:
        columns = df_cleaned.columns[df_cleaned.isnull().any()].tolist()

    for col in columns:
        if df_cleaned[col].dtype in ['object', 'category']:
            if strategy == 'mode':
                mode_value = df_cleaned[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col].fillna(mode_value[0], inplace=True)
            elif strategy == 'drop':
                df_cleaned.dropna(subset=[col], inplace=True)
        else:  # numeric columns
            if strategy == 'mean':
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == 'median':
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif strategy == 'drop':
                df_cleaned.dropna(subset=[col], inplace=True)
            elif strategy == 'forward_fill':
                df_cleaned[col].fillna(method='ffill', inplace=True)
            elif strategy == 'backward_fill':
                df_cleaned[col].fillna(method='bfill', inplace=True)

    return df_cleaned


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with data information
    """
    info_dict = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
    }
    return info_dict


def detect_outliers(df: pd.DataFrame, columns: list = None, method: str = 'iqr') -> dict:
    """
    Detect outliers in numeric columns.

    Args:
        df: Input DataFrame
        columns: Specific numeric columns to check; if None, check all numeric columns
        method: Method to detect outliers ('iqr' or 'zscore')

    Returns:
        Dictionary with outlier information for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers_info = {}

    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]

        outliers_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'indices': outliers.index.tolist()
        }

    return outliers_info


def validate_data_types(df: pd.DataFrame, expected_types: dict) -> dict:
    """
    Validate that columns have expected data types.

    Args:
        df: Input DataFrame
        expected_types: Dictionary mapping column names to expected types

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'validations_passed': True,
        'errors': [],
        'type_comparisons': {}
    }

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            validation_results['validations_passed'] = False
            validation_results['errors'].append(f"Column '{col}' not found in DataFrame")
            continue

        actual_type = str(df[col].dtype)
        validation_results['type_comparisons'][col] = {
            'expected': expected_type,
            'actual': actual_type,
            'match': expected_type.lower() in actual_type.lower()
        }

        if not validation_results['type_comparisons'][col]['match']:
            validation_results['validations_passed'] = False
            validation_results['errors'].append(
                f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
            )

    return validation_results


def validate_no_missing_values(df: pd.DataFrame, columns: list = None) -> dict:
    """
    Validate that there are no missing values in specified columns.

    Args:
        df: Input DataFrame
        columns: Specific columns to check; if None, check all columns

    Returns:
        Dictionary with validation results
    """
    if columns is None:
        columns = df.columns.tolist()

    missing_check = {
        'validations_passed': True,
        'errors': [],
        'missing_counts': {}
    }

    for col in columns:
        if col not in df.columns:
            missing_check['validations_passed'] = False
            missing_check['errors'].append(f"Column '{col}' not found in DataFrame")
            continue

        missing_count = df[col].isnull().sum()
        missing_check['missing_counts'][col] = missing_count

        if missing_count > 0:
            missing_check['validations_passed'] = False
            missing_check['errors'].append(f"Column '{col}' has {missing_count} missing values")

    return missing_check


def validate_column_values(df: pd.DataFrame, column_rules: dict) -> dict:
    """
    Validate column values against specific rules.

    Args:
        df: Input DataFrame
        column_rules: Dictionary mapping column names to validation rules
                     Rules can include: min_val, max_val, allowed_values, forbidden_values

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'validations_passed': True,
        'errors': [],
        'column_checks': {}
    }

    for col, rules in column_rules.items():
        if col not in df.columns:
            validation_results['validations_passed'] = False
            validation_results['errors'].append(f"Column '{col}' not found in DataFrame")
            continue

        col_results = {'rules_applied': [], 'violations': []}

        # Check min value
        if 'min_val' in rules:
            violations = df[df[col] < rules['min_val']]
            if len(violations) > 0:
                col_results['violations'].append(f"Found {len(violations)} values below minimum {rules['min_val']}")
                validation_results['validations_passed'] = False

        # Check max value
        if 'max_val' in rules:
            violations = df[df[col] > rules['max_val']]
            if len(violations) > 0:
                col_results['violations'].append(f"Found {len(violations)} values above maximum {rules['max_val']}")
                validation_results['validations_passed'] = False

        # Check allowed values
        if 'allowed_values' in rules:
            invalid_vals = df[~df[col].isin(rules['allowed_values'])]
            if len(invalid_vals) > 0:
                unique_invalid = invalid_vals[col].unique()
                col_results['violations'].append(f"Found invalid values: {list(unique_invalid)}")
                validation_results['validations_passed'] = False

        # Check forbidden values
        if 'forbidden_values' in rules:
            forbidden_vals = df[df[col].isin(rules['forbidden_values'])]
            if len(forbidden_vals) > 0:
                unique_forbidden = forbidden_vals[col].unique()
                col_results['violations'].append(f"Found forbidden values: {list(unique_forbidden)}")
                validation_results['validations_passed'] = False

        validation_results['column_checks'][col] = col_results

    return validation_results