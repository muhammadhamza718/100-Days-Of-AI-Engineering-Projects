# Day 3-4: Pandas for Data Manipulation

## Overview

Welcome to Days 3-4! In this phase, we master **Pandas**, the industry standard for data manipulation and analysis in Python. You will learn to clean messy data, perform complex aggregations, and gain insights from real-world datasets.

## Tech Stack

- **Language:** Python 3.x
- **Core Library:** Pandas
- **Visualization:** Matplotlib, Seaborn
- **Package Manager:** uv
- **Environment:** Jupyter Notebook or VS Code

---

## 1. Core Concepts: DataFrames & Series

### 1.1 Data Structures

- **Series:** One-dimensional labeled array generally capable of holding any data type.
- **DataFrame:** Two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).

```python
import pandas as pd

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Creating a DataFrame
df = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20230102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})
```

### 1.2 Indexing and Selection

- Selecting columns: `df['A']`
- Slicing rows: `df[0:3]`
- Selection by label (`loc`): `df.loc[:, ['A', 'B']]`
- Selection by position (`iloc`): `df.iloc[3]`
- Boolean indexing: `df[df['A'] > 0]`

---

## 2. Data Cleaning & Handling Missing Values

Real-world data is messy. You must learn to handle `NaN`, `None`, and missing data.

- **Inspection:** `df.head()`, `df.tail()`, `df.info()`, `df.describe()`
- **Handling Missing Data:**
  - Drop missing values: `df.dropna()`
  - Fill missing values: `df.fillna(value=5)` or `df.fillna(method='ffill')`
- **Data Types:** Converting types with `df.astype()`.

---

## 3. Advanced Operations: Groupby, Pivot, & Merge

### 3.1 GroupBy

Split-Apply-Combine strategy.

```python
# Group by column 'A' and sum column 'B'
df.groupby('A')['B'].sum()
```

### 3.2 Pivot Tables

Create a spreadsheet-style pivot table as a DataFrame.

```python
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
```

### 3.3 Merging (Joins)

Combining DataFrames similarly to SQL joins.

- `pd.concat()`: Stacking together objects along an axis.
- `pd.merge()`: Database-style join operations.
  ```python
  pd.merge(left, right, on='key')
  ```

---

## 4. Exercise: Analyze a Real Dataset

**Goal:** Load a CSV dataset and perform exploratory data analysis (EDA).
**Dataset:** Use a dataset like the Titanic dataset or global CO2 emissions from Kaggle.

**Tasks:**

1.  Load the dataset into a DataFrame.
2.  Clean the data (handle missing rows/columns).
3.  Calculate summary statistics.
4.  Answer 3 specific questions about the data using filtering and sorting (e.g., "What is the average age of survivors?").

---

## 5. Mini Project: COVID-19 Data Analysis Dashboard

**Goal:** Build a script/notebook that analyzes COVID-19 trends.

**Requirements:**

1.  **Data Source:** Download the [CSSEGISandData COVID-19 dataset](https://github.com/CSSEGISandData/COVID-19) (or use a direct URL).
2.  **Preprocessing:**
    - Aggregate data by Country/Region.
    - Convert dates to datetime objects.
    - Handle missing province data.
3.  **Analysis:**
    - Calculate daily new cases (using `diff()`).
    - Find the top 5 countries with the highest total cases.
4.  **Visualization (Dashboard):**
    - Plot the "Total Cases" curve for the top 5 countries.
    - Create a bar chart comparing "Mortality Rate" among widely affected countries.

**Deliverables:**

- A clean Jupyter Notebook (`COVID19_Analysis.ipynb`).
- Visualizations saved as PNG files.
