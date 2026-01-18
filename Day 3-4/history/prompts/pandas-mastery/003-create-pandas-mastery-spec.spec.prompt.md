---
id: 003
title: Create Pandas Mastery Spec
stage: spec
date: 2026-01-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: pandas-mastery
branch: 001-pandas-mastery
user: user
command: /sp.specify
labels: [specification, pandas, data-science, education]
links:
  spec: F:/Courses/Hamza/100-Days-Of-AI-Engineering-Projects/specs/001-pandas-mastery/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-pandas-mastery/spec.md
  - specs/001-pandas-mastery/checklists/requirements.md
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

```text
# Project Specifications: Day 3-4 Pandas Mastery

# Overview
We are building a comprehensive learning module for Pandas, covering foundational structures (DataFrames, Series), intermediate data cleaning, and advanced aggregation techniques. This culminates in a real-world analysis exercise and a COVID-19 data dashboard.

# Functional Requirements

## 1. Core Concepts & Cleaning
Create educational notebooks/scripts that demonstrate:
- **Structures**: Creating Series and DataFrames from dictionaries, lists, and arrays.
- **Indexing**: Selection using `loc`, `iloc`, and boolean masking.
- **Cleaning**: Identifying `NaN` values, filling them (mean/mode/forward-fill), and dropping dirty rows.
- **Aggregations**: Using `groupby` vs. `pivot_table` for summarizing data.

## 2. Exercise: Real Dataset Analysis
Develop a script/notebook to analyze a CSV dataset (e.g., Titanic or similar public data) that:
- Loads and inspects the data structure (`info`, `describe`).
- Cleans missing values appropriate to the column (e.g., age vs. cabin).
- Answers specific queries:
    - What is the survival rate by gender? (if Titanic)
    - What is the distribution of numeric values?

## 3. Mini-Project: COVID-19 Dashboard
Develop a distinct notebook `COVID19_Analysis.ipynb` that:
- **Ingestion**: Loads CSSEGISandData COVID-19 time-series CSVs directly from URL or local file.
- **Preprocessing**:
    - Groups data by `Country/Region` (summing over provinces).
    - Pivots or melts data into a time-series format.
- **Analysis**:
    - Calculates daily new cases.
    - Identifies top 5 countries by total confirmed cases.
- **Visualization**:
    - Plots "Total Cases vs. Time" for the top 5 countries.
    - Plots "Mortality Rate" (Deaths/Cases) for top affected countries.

# Technical Stack & Environment
- **Language**: Python 3.x
- **Core Library**: Pandas
- **Visualization**: Matplotlib, Seaborn
- **Package Manager**: uv
- **Environment**: Jupyter Notebook or VS Code
```

## Response snapshot

Successfully created the Pandas Mastery module specification with three prioritized user stories: (1) Pandas Fundamentals Learning (P1), (2) Real Dataset Analysis Exercise (P2), and (3) COVID-19 Dashboard Creation (P3). The specification includes detailed acceptance scenarios, 15 functional requirements (FR-001 through FR-015), key entities, and 5 measurable success criteria (SC-001 through SC-005). Also created a validation checklist and updated the project constitution to v1.0.2 to reflect the new data science focus.

## Outcome

- ✅ Impact: Established comprehensive requirements for the Pandas learning module with clear user journeys
- 🧪 Tests: none
- 📁 Files: Created spec.md, requirements.md checklist, and updated constitution.md
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: Specification aligns with educational objectives and includes all required functionality

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A