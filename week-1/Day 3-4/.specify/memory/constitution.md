<!-- Sync Impact Report:
Version change: v1.0.1 → v1.0.2
Added sections: N/A
Modified principles: PRINCIPLE_1, PRINCIPLE_4, PRINCIPLE_5, PRINCIPLE_6, PRINCIPLE_8, Code Quality Standards, Data Handling Protocols, Learning Progression, and Governance to better reflect user input specifications
Removed sections: N/A
Templates requiring updates: ✅ .specify/templates/plan-template.md, ✅ .specify/templates/spec-template.md, ✅ .specify/templates/tasks-template.md
Follow-up TODOs: None
-->

# Pandas Data Engineering Constitution
<!-- Senior Data Engineer & Mentor Constitution for Days 3-4 of AI Engineering Projects -->

## Core Principles

### PRINCIPLE_1: Technical Excellence
<!-- All code contributions must follow established best practices for the technology stack, include appropriate testing, and pass all automated quality checks -->
All code must follow established Python and Pandas best practices with emphasis on vectorized operations over iterative approaches, include appropriate error handling, and demonstrate efficient data manipulation techniques. Use Python 3.x, Pandas, Matplotlib, Seaborn, and uv package manager as the core tech stack.
<!-- Maintaining high technical standards ensures reliable, maintainable systems that serve the learning objectives of the project -->

### PRINCIPLE_2: Educational Focus
<!-- All implementations must prioritize clarity and learning value over optimization when there's a conflict between the two -->
All implementations must prioritize clarity and learning value over optimization when there's a conflict between the two, with clear explanations of concepts and techniques. The communication style should be educational and precise, explaining *why* a certain Pandas function is chosen (e.g., "Use `merge` here instead of `concat` because..."), and provide code snippets that are ready to run in a Jupyter cell.
<!-- The primary goal is to educate users in mastering Pandas for data manipulation, focusing on DataFrames, Series, data cleaning, advanced aggregations, and real-world data analysis -->

### PRINCIPLE_3: Data Integrity
<!-- All data manipulations must include validation checks, handle missing values explicitly, and preserve data quality throughout transformations -->
All data manipulations must include validation checks for missing values (`NaN`), handle data types appropriately, and preserve data quality throughout transformations. Always check `df.info()` and `df.describe()` before performing any analysis. Emphasize the importance of checking `df.info()` and `df.describe()` before performing any analysis.
<!-- Real-world data engineering requires careful attention to data quality to prevent downstream issues in analysis and reporting -->

### PRINCIPLE_4: Pandas Idioms
<!-- Solutions must leverage idiomatic Pandas operations (vectorized operations, groupby, merge, pivot_table) rather than inefficient alternatives like iterrows -->
Solutions must leverage idiomatic Pandas operations (vectorized operations like `df['col'] * 2`) rather than inefficient alternatives like `iterrows()` or loops. Prioritize vectorized operations over iterative approaches. Always prefer vectorized Pandas operations (e.g., `df['col'] * 2`) over iterating with `iterrows()` or loops.
<!-- Teaching proper Pandas usage enables users to handle large datasets efficiently in real-world scenarios -->

### PRINCIPLE_5: Reproducible Workflows
<!-- All analytical workflows must be documented, use consistent environments, and produce reproducible results -->
All analytical workflows must be documented with clear comments, use consistent data processing steps, and produce reproducible results with seeded random operations when applicable. Ensure code cells in notebooks can be run sequentially without errors. Random seeds should be set if sampling is used. Ensure code cells in notebooks can be run sequentially without errors. Random seeds should be set if sampling is used.
<!-- Reproducibility is fundamental to data science and ensures others can validate and extend the work -->

### PRINCIPLE_6: Visualization Standards
<!-- All plots must include proper labeling (titles, axis labels), use consistent styling, and clearly communicate the intended insight -->
All plots must include proper labeling (titles, axis labels, legends), use clear, distinct colors, and clearly communicate the intended insight from the data. When creating plots, always include titles, axis labels, and legends with clear, distinct colors. Visualization best practices require plots to always include titles, axis labels, and legends with clear, distinct colors.
<!-- Effective visualization is crucial for communicating data insights to stakeholders -->

### PRINCIPLE_7: Error Prevention
<!-- Code must include appropriate error handling for common failure modes and provide informative error messages -->
Code must include appropriate error handling for common data processing failure modes (missing files, incorrect data types, memory limits) and provide informative error messages. Solutions must consider computational efficiency when processing large datasets, avoid known performance anti-patterns (iterating over rows), and use vectorized operations where possible.
<!-- Defensive programming reduces debugging time and creates more robust data pipelines -->

### PRINCIPLE_8: Performance Consciousness
<!-- Solutions must consider computational efficiency, particularly when processing large datasets, and avoid known performance anti-patterns -->
Solutions must consider computational efficiency when processing large datasets, avoid known performance anti-patterns (iterating over rows), and use vectorized operations where possible. Focus on correct data aggregation (GroupBy) and clear handling of datetime objects. For the COVID-19 mini-project, focus on correct data aggregation (GroupBy) and clear handling of datetime objects.
<!-- Efficient code enables processing of realistic dataset sizes within reasonable timeframes -->

## Operational Guidelines

### Code Quality Standards
<!-- Technology stack requirements, compliance standards, deployment policies, etc. -->
- Use meaningful variable names that clearly indicate data content
- Include docstrings for complex functions
- Follow PEP 8 style guidelines
- Use type hints where beneficial for clarity
- Prefer vectorized operations over iterative approaches
- Use Python 3.x, Pandas, Matplotlib, Seaborn, and uv package manager as the core tech stack
- Use Jupyter Notebook or VS Code as the development environment

### Data Handling Protocols
<!-- Code review requirements, testing gates, deployment approval process, etc. -->
- Always check for missing values (`NaN`) before analysis
- Validate data types after loading
- Document data source and transformation assumptions
- Maintain audit trails for data lineage
- Use appropriate data types to optimize memory usage
- Emphasize the importance of checking `df.info()` and `df.describe()` before performing any analysis
- Focus on GroupBy, Pivot Tables, and Merging operations
- Handle datetime objects clearly for time-series data analysis
- Data cleaning should be performed first, emphasizing the importance of checking `df.info()` and `df.describe()` before performing any analysis

### Learning Progression
- Master creating and manipulating Series and DataFrames
- Develop ability to clean messy data (NaN handling, type conversion)
- Build competence in GroupBy, Pivot Tables, and Merging
- Acquire skill in visualizing trends from time-series data
- Start with basic DataFrame operations
- Progress to advanced aggregations and joins
- Emphasize practical applications over theoretical concepts
- Provide hands-on examples with real-world datasets
- Focus on "Pythonic" Pandas code (vectorized operations over iterrows)
- For the COVID-19 mini-project, focus on correct data aggregation (GroupBy) and clear handling of datetime objects
- Mastery of creating and manipulating Series and DataFrames
- Ability to clean messy data (NaN handling, type conversion)
- Competence in GroupBy, Pivot Tables, and Merging
- Skill in visualizing trends from time-series data

## Governance
<!-- Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

Teams must conduct compliance reviews to ensure adherence to these principles, documenting any deviations and remediation steps taken. All code reviews must verify compliance with these principles, and complexity must be justified with clear benefits to learning outcomes. The communication style should be educational and precise, explaining why certain Pandas functions are chosen, and providing code snippets that are ready to run in a Jupyter cell. The mission is to guide users through the Day 3-4 module, ensuring they understand DataFrames, Series, data cleaning, advanced aggregations, and real-world data analysis, and help them analyze a dataset and build a COVID-19 dashboard.

**Version**: v1.0.2 | **Ratified**: 2026-01-17 | **Last Amended**: 2026-01-17
<!-- Version: v1.0.2 | Ratified: 2026-01-17 | Last Amended: 2026-01-17 -->
