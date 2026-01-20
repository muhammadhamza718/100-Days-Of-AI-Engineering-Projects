# Quickstart Guide: Data Visualization Mastery

## Setup

1. Ensure you have Python 3.x and uv installed
2. Navigate to the project directory
3. Install dependencies:
   ```
   uv add matplotlib seaborn plotly pandas numpy scipy
   ```

## Running the Visualization Gallery

```bash
uv run python visualization_gallery.py
```

This will generate examples using all three visualization libraries (matplotlib, seaborn, plotly) with proper titles, axis labels, and legends.

## Running the Titanic EDA

```bash
uv run python Titanic_EDA.py
```

This will perform the 7-step EDA on the Titanic dataset, generating 15+ plots with proper formatting.

## Expected Outputs

- Multiple PNG files of key visualizations
- Insights documented in insights.md
- All plots with titles, axis labels, and legends