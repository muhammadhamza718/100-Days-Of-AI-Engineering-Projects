# Contracts: Pandas Mastery Module

## Educational Data Science Contracts

This project is an educational module focused on teaching Pandas data manipulation, so traditional API contracts are not applicable. Instead, the contracts here represent:

1. **Function Interfaces**: Well-defined functions in utility files that demonstrate best practices
2. **Data Contracts**: Expected input/output formats for data processing operations
3. **Notebook Interfaces**: Expected cell structures and outputs for educational purposes

## Function Signatures (Example)

```python
def clean_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Clean missing values from a DataFrame using specified strategy

    Args:
        df: Input DataFrame
        strategy: How to handle missing values ('mean', 'mode', 'drop', 'forward_fill')

    Returns:
        Cleaned DataFrame
    """
    pass

def aggregate_by_country(covid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate COVID-19 data by country

    Args:
        covid_df: DataFrame with COVID-19 data including 'Country/Region' column

    Returns:
        Aggregated DataFrame grouped by country
    """
    pass
```

## Data Format Contracts

- Input CSV files should have consistent column headers
- Date columns should be parseable by pandas.to_datetime()
- Numeric columns should contain valid numeric data or NaN
- Output visualizations should follow matplotlib/seaborn best practices