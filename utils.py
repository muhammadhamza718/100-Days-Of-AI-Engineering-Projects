"""
Utility functions for data processing in the Pandas Mastery Module.
"""

import pandas as pd
import numpy as np


def clean_missing_values(df, strategy='mean', columns=None):
    """
    Clean missing values from a DataFrame using specified strategy

    Args:
        df: Input DataFrame
        strategy: How to handle missing values ('mean', 'mode', 'drop', 'forward_fill')
        columns: Specific columns to clean (None for all columns)

    Returns:
        Cleaned DataFrame
    """
    df_cleaned = df.copy()

    if columns is None:
        columns = df_cleaned.columns

    for col in columns:
        if df_cleaned[col].isna().any():
            if strategy == 'mean' and df_cleaned[col].dtype in ['float64', 'int64']:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == 'mode':
                mode_value = df_cleaned[col].mode()
                if not mode_value.empty:
                    df_cleaned[col].fillna(mode_value[0], inplace=True)
            elif strategy == 'forward_fill':
                df_cleaned[col].fillna(method='ffill', inplace=True)
            elif strategy == 'drop':
                df_cleaned.dropna(subset=[col], inplace=True)

    return df_cleaned


def aggregate_by_country(covid_df):
    """
    Aggregate COVID-19 data by country

    Args:
        covid_df: DataFrame with COVID-19 data including 'Country/Region' column

    Returns:
        Aggregated DataFrame grouped by country
    """
    # Group by Country/Region and sum the numerical values
    country_agg = covid_df.groupby('Country/Region').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum'
    }).reset_index()

    return country_agg


def calculate_daily_new_cases(cumulative_series):
    """
    Calculate daily new cases from cumulative data

    Args:
        cumulative_series: Series with cumulative case counts

    Returns:
        Series with daily new case counts
    """
    # Calculate difference between consecutive days
    daily_cases = cumulative_series.diff()
    # Fill the first value (which will be NaN) with the first cumulative value
    daily_cases.iloc[0] = cumulative_series.iloc[0]
    # Ensure no negative values (can happen with data corrections)
    daily_cases = daily_cases.clip(lower=0)

    return daily_cases


def validate_data_types(df):
    """
    Validate data types in a DataFrame

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with validation results
    """
    validation_results = {}

    for col in df.columns:
        validation_results[col] = {
            'type': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isna().sum(),
            'unique_count': df[col].nunique()
        }

    return validation_results