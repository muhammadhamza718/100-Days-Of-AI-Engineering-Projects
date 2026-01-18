# Data Model: Pandas Mastery Module

## Core Entities

### DataFrame
- **Description**: Tabular data structure representing datasets with labeled axes (rows and columns)
- **Fields**: Index (row labels), Columns (column labels), Values (actual data)
- **Relationships**: Contains multiple Series as columns
- **Validation**: Should have consistent data types per column, valid index structure

### Series
- **Description**: One-dimensional labeled array representing a single column of data
- **Fields**: Index (row labels), Values (single column data)
- **Relationships**: Component of DataFrame
- **Validation**: Should have consistent data type, valid index structure

### COVID-19 Dataset
- **Description**: Time-series dataset containing confirmed cases, deaths, and recoveries by geographic region and date
- **Fields**:
  - Date (datetime)
  - Country/Region (string)
  - Province/State (string, optional)
  - Confirmed (int)
  - Deaths (int)
  - Recovered (int)
- **Relationships**: Aggregated from provincial data to country-level
- **Validation**: Date values must be chronological, numeric values must be non-negative

## Data Relationships

### Hierarchical Structure
- Raw COVID-19 data exists at Province/State level
- Aggregation occurs to Country/Region level by summing values
- Time-series structure connects dates for trend analysis

### Transformation Pipeline
- Raw data → Cleaned data (missing values handled)
- Province-level data → Country-level data (aggregated)
- Cumulative data → Daily data (calculated differences)

## Data Validation Rules

### From Functional Requirements
- **FR-006**: Data structure inspection using `info()` and `describe()` methods
- **FR-007**: Missing values identified and handled appropriately per column type
- **FR-011**: Country/Region grouping with summation over provinces
- **FR-012**: Daily new cases calculated from cumulative data (using diff() or shift())

### Quality Checks
- Data types should be appropriate for each column
- Missing values should be explicitly handled (not silently dropped)
- Aggregated values should maintain mathematical correctness
- Time-series data should maintain chronological order