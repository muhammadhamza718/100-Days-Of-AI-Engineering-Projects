# Data Model: Data Visualization Mastery

## Entities

### TitanicDataset
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Binary indicator (0 = No, 1 = Yes) for survival status
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger's name
- **Sex**: Passenger's gender
- **Age**: Passenger's age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### VisualizationGallery
- **PlotType**: Type of visualization (histogram, scatter, bar, etc.)
- **LibraryUsed**: Visualization library (matplotlib, seaborn, plotly)
- **Title**: Descriptive title for the plot
- **XAxisLabel**: Label for x-axis
- **YAxisLabel**: Label for y-axis
- **Legend**: Legend description if applicable

### EDAAnalysis
- **AnalysisStep**: Step in the 7-step EDA methodology
- **AnalysisType**: Type of analysis (univariate, bivariate, multivariate)
- **VariablesAnalyzed**: Variables involved in the analysis
- **StatisticalMeasure**: Statistical measure computed
- **Visualization**: Associated visualization
- **Insight**: Key insight extracted from analysis

## Relationships

- **TitanicDataset** → **EDAAnalysis**: One dataset can have multiple analysis steps
- **EDAAnalysis** → **VisualizationGallery**: Each analysis can produce multiple visualizations