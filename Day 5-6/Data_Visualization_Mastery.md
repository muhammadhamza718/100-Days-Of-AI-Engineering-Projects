# Day 5-6: Data Visualization Mastery

## Overview

Welcome to Days 5-6 of your AI Engineering journey! In this module, we dive deep into **Data Visualization**, a critical skill for communicating insights from data. You'll master the three most powerful Python visualization libraries: **Matplotlib** (the foundation), **Seaborn** (statistical elegance), and **Plotly** (interactive brilliance). By the end, you'll create stunning, publication-ready visualizations and perform comprehensive Exploratory Data Analysis (EDA) on the Titanic dataset.

## Tech Stack

- **Language:** Python 3.x
- **Core Libraries:** Matplotlib, Seaborn, Plotly
- **Data Manipulation:** Pandas, NumPy
- **Package Manager:** uv

---

## 1. Matplotlib: The Foundation

**Matplotlib** is the grandfather of Python visualization libraries. It provides fine-grained control over every element of a plot.

### 1.1 Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('Sine Wave', fontsize=16, fontweight='bold')
plt.xlabel('X axis', fontsize=12)
plt.ylabel('Y axis', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.2 Multiple Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Line plot
axes[0, 0].plot(x, np.sin(x), 'r-')
axes[0, 0].set_title('Sin(x)')

# Subplot 2: Scatter plot
axes[0, 1].scatter(x, np.cos(x), c='green', alpha=0.5)
axes[0, 1].set_title('Cos(x)')

# Subplot 3: Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 0].bar(categories, values, color='purple')
axes[1, 0].set_title('Bar Chart')

# Subplot 4: Histogram
data = np.random.randn(1000)
axes[1, 1].hist(data, bins=30, color='orange', edgecolor='black')
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### 1.3 Customization

```python
# Advanced customization
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='#FF5733', linestyle='--', marker='o',
         markevery=10, markersize=8, label='Custom Style')

# Styling
plt.style.use('seaborn-v0_8-darkgrid')  # or 'ggplot', 'fivethirtyeight'
plt.title('Advanced Customization', fontsize=18, pad=20)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.show()
```

**Common Plot Types:**

- Line plots: `plt.plot()`
- Scatter plots: `plt.scatter()`
- Bar charts: `plt.bar()`, `plt.barh()`
- Histograms: `plt.hist()`
- Pie charts: `plt.pie()`
- Box plots: `plt.boxplot()`

---

## 2. Seaborn: Statistical Elegance

**Seaborn** is built on top of Matplotlib and provides a high-level interface for drawing attractive statistical graphics.

### 2.1 Statistical Plots

```python
import seaborn as sns
import pandas as pd

# Sample dataset
tips = sns.load_dataset('tips')

# Distribution plot (Histogram + KDE)
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, color='skyblue')
plt.title('Distribution of Total Bill')
plt.show()

# Count plot (categorical)
plt.figure(figsize=(8, 6))
sns.countplot(data=tips, x='day', palette='Set2')
plt.title('Number of Visits per Day')
plt.show()
```

### 2.2 Relationship Plots

```python
# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'alpha':0.5})
plt.title('Tip vs Total Bill with Regression Line')
plt.show()

# Pair plot (multiple relationships at once)
sns.pairplot(tips, hue='sex', palette='husl')
plt.show()

# Heatmap (correlation matrix)
plt.figure(figsize=(8, 6))
correlation = tips.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Correlation Heatmap')
plt.show()
```

### 2.3 Categorical Plots

```python
# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', palette='Set1')
plt.title('Total Bill Distribution by Day and Gender')
plt.show()

# Violin plot (box plot + KDE)
plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex',
               split=True, palette='muted')
plt.title('Violin Plot: Total Bill Distribution')
plt.show()

# Strip/Swarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex', palette='Set2')
plt.title('Swarm Plot: Individual Data Points')
plt.show()
```

### 2.4 Themes and Styling

```python
# Set theme
sns.set_theme(style='whitegrid')  # Options: 'darkgrid', 'white', 'dark', 'ticks'

# Set color palette
sns.set_palette('husl')  # Or 'deep', 'muted', 'bright', 'pastel', 'dark'

# Context for scaling
sns.set_context('talk')  # Options: 'paper', 'talk', 'poster'
```

---

## 3. Plotly: Interactive Visualizations

**Plotly** creates interactive, publication-quality graphs that can be embedded in web applications.

### 3.1 Basic Interactive Plots

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive line chart
fig = px.line(x=x, y=y, title='Interactive Sine Wave',
              labels={'x': 'X-axis', 'y': 'Y-axis'})
fig.update_traces(line_color='red', line_width=3)
fig.show()

# Interactive scatter plot
iris = px.data.iris()
fig = px.scatter(iris, x='sepal_width', y='sepal_length',
                 color='species', size='petal_length',
                 hover_data=['petal_width'],
                 title='Iris Dataset Interactive Scatter')
fig.show()
```

### 3.2 Advanced Interactive Plots

```python
# 3D Scatter plot
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species', size='petal_length',
                    title='3D Visualization of Iris Dataset')
fig.show()

# Interactive heatmap
df = pd.DataFrame(np.random.randn(50, 20))
fig = go.Figure(data=go.Heatmap(z=df.values, colorscale='Viridis'))
fig.update_layout(title='Interactive Heatmap')
fig.show()

# Choropleth map
gapminder = px.data.gapminder().query("year == 2007")
fig = px.choropleth(gapminder, locations='iso_alpha',
                    color='lifeExp', hover_name='country',
                    color_continuous_scale='Plasma',
                    title='Life Expectancy by Country (2007)')
fig.show()
```

### 3.3 Subplots and Dashboards

```python
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Scatter', 'Bar', 'Line', 'Pie')
)

# Add traces
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[10, 20, 15]), row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 4, 6], mode='lines'), row=2, col=1)
fig.add_trace(go.Pie(labels=['X', 'Y', 'Z'], values=[30, 40, 30]), row=2, col=2)

fig.update_layout(height=600, showlegend=False, title_text='Interactive Dashboard')
fig.show()
```

---

## 4. Statistical Plots and Distributions

Understanding distributions is fundamental for data analysis and machine learning.

### 4.1 Normal Distribution

```python
# Generate normal distribution
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 10000)

# Plot with Matplotlib
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(s, 50, density=True, alpha=0.7, color='blue')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - mu)**2 / (2 * sigma**2)),
         linewidth=2, color='red', label='PDF')
plt.title('Normal Distribution')
plt.legend()
plt.show()
```

### 4.2 Distribution Comparison

```python
# Using Seaborn for multiple distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
sns.histplot(data=tips, x='total_bill', kde=False, ax=axes[0, 0])
axes[0, 0].set_title('Histogram')

# KDE plot
sns.kdeplot(data=tips, x='total_bill', shade=True, ax=axes[0, 1])
axes[0, 1].set_title('KDE Plot')

# Distribution plot (deprecated in newer versions, use histplot with kde=True)
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram + KDE')

# ECDF (Empirical Cumulative Distribution Function)
sns.ecdfplot(data=tips, x='total_bill', ax=axes[1, 1])
axes[1, 1].set_title('ECDF')

plt.tight_layout()
plt.show()
```

### 4.3 Q-Q Plot (Quantile-Quantile)

```python
from scipy import stats

# Check if data follows normal distribution
fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(tips['total_bill'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Normal Distribution Check')
plt.show()
```

---

## 5. Exercise: Create a Visualization Gallery

**Goal:** Practice all three libraries by creating a variety of plots.

**Tasks:**

1. **Matplotlib:**
   - Create a multi-panel figure (2x3 grid) showing: line plot, scatter, bar chart, histogram, pie chart, and box plot
   - Use at least 3 different color schemes
   - Add proper titles, labels, and legends

2. **Seaborn:**
   - Use the `penguins` dataset: `sns.load_dataset('penguins')`
   - Create: distribution plot, count plot, box plot, violin plot, pair plot, and heatmap
   - Experiment with different themes and color palettes

3. **Plotly:**
   - Create an interactive dashboard with 4 subplots
   - Include at least one 3D visualization
   - Add hover tooltips and zooming capabilities

**Deliverables:**

- Python script(s) (`visualization_gallery.py`) with all plots
- Export key plots as PNG files

---

## 6. Mini Project: Exploratory Data Analysis (EDA) on Titanic Dataset

**Goal:** Perform comprehensive EDA on the Titanic dataset using all visualization techniques learned.

### 6.1 Project Requirements

**Dataset:** Titanic (available from Seaborn or Kaggle)

```python
# Load dataset
titanic = sns.load_dataset('titanic')
# OR
# titanic = pd.read_csv('titanic.csv')
```

### 6.2 Analysis Steps

**Step 1: Data Understanding**

```python
# Basic exploration
print(titanic.head())
print(titanic.info())
print(titanic.describe())
print(titanic.isnull().sum())
```

**Step 2: Data Cleaning**

```python
# Handle missing values
# - Age: Fill with median or use predictive imputation
# - Embarked: Fill with mode
# - Cabin: Drop or create 'has_cabin' binary feature

# Example
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic['has_cabin'] = titanic['cabin'].notna().astype(int)
```

**Step 3: Univariate Analysis**

```python
# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', kde=True, bins=30)
plt.title('Age Distribution of Passengers')
plt.show()

# Survival count
sns.countplot(data=titanic, x='survived', palette='Set2')
plt.title('Survival Count')
plt.show()

# Class distribution
sns.countplot(data=titanic, x='pclass', palette='viridis')
plt.title('Passenger Class Distribution')
plt.show()
```

**Step 4: Bivariate Analysis**

```python
# Survival by gender
sns.countplot(data=titanic, x='sex', hue='survived', palette='pastel')
plt.title('Survival Rate by Gender')
plt.show()

# Survival by class
sns.countplot(data=titanic, x='pclass', hue='survived', palette='Set1')
plt.title('Survival Rate by Class')
plt.show()

# Age vs Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived',
                size='pclass', sizes=(50, 200), alpha=0.6)
plt.title('Age vs Fare (colored by survival)')
plt.show()
```

**Step 5: Multivariate Analysis**

```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
correlation = titanic[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(titanic[['survived', 'pclass', 'age', 'fare', 'sex']],
             hue='survived', palette='husl')
plt.show()

# Survival by multiple factors
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=titanic, x='pclass', y='age', hue='survived', ax=axes[0])
axes[0].set_title('Age by Class and Survival')

sns.violinplot(data=titanic, x='sex', y='fare', hue='survived',
               split=True, ax=axes[1])
axes[1].set_title('Fare by Gender and Survival')

sns.countplot(data=titanic, x='embarked', hue='survived', ax=axes[2])
axes[2].set_title('Survival by Embarkation Port')

plt.tight_layout()
plt.show()
```

**Step 6: Interactive Visualizations**

```python
# Interactive survival analysis
fig = px.sunburst(titanic, path=['pclass', 'sex', 'survived'],
                  values='fare', title='Titanic Survival Hierarchy')
fig.show()

# Interactive scatter
fig = px.scatter(titanic, x='age', y='fare', color='survived',
                 size='pclass', hover_data=['sex', 'embarked'],
                 title='Interactive Age vs Fare Analysis')
fig.show()

# 3D visualization
fig = px.scatter_3d(titanic, x='age', y='fare', z='pclass',
                    color='survived', size='siblings_spouses_aboard',
                    title='3D Titanic Visualization')
fig.show()
```

**Step 7: Statistical Insights**

```python
# Survival rate by category
print("Overall Survival Rate:", titanic['survived'].mean())
print("\nSurvival Rate by Gender:")
print(titanic.groupby('sex')['survived'].mean())
print("\nSurvival Rate by Class:")
print(titanic.groupby('pclass')['survived'].mean())

# Age statistics by survival
print("\nAge Statistics by Survival:")
print(titanic.groupby('survived')['age'].describe())
```

### 6.3 Deliverables

1. **Python Script:** `Titanic_EDA.py` containing:
   - Data loading and cleaning
   - All visualizations (at least 15 different plots)
   - Statistical summaries
   - Key insights and findings (as comments)

2. **Insights Report:** A separate `insights.md` file with:
   - Top 5 findings from the analysis
   - Survival patterns discovered
   - Recommendations or hypotheses

3. **Visualizations:** Export at least 5 key plots as PNG files for presentation

4. **Optional:** Create an interactive HTML dashboard using Plotly

### 6.4 Expected Insights

You should discover patterns such as:

- Women had higher survival rates than men
- First-class passengers had better survival chances
- Age groups showed different survival patterns
- Fare prices correlated with survival
- Embarkation port might have influenced survival

---

## 7. Best Practices for Data Visualization

### 7.1 Design Principles

1. **Choose the Right Chart Type:**
   - Trends over time: Line charts
   - Comparisons: Bar charts
   - Distributions: Histograms, box plots
   - Relationships: Scatter plots
   - Proportions: Pie charts (use sparingly!)
   - Correlations: Heatmaps

2. **Use Color Effectively:**
   - Color-blind friendly palettes (e.g., 'viridis', 'plasma')
   - Consistent color schemes across related visualizations
   - Don't use more than 5-7 distinct colors

3. **Label Everything:**
   - Clear titles and axis labels
   - Units of measurement
   - Legends when using multiple categories
   - Annotations for key points

4. **Keep it Simple:**
   - Avoid chart junk (unnecessary decorations)
   - Remove gridlines if not needed
   - Focus on the data story

### 7.2 Common Pitfalls to Avoid

- ❌ 3D pie charts (distortion)
- ❌ Truncated y-axis (misleading)
- ❌ Too many colors
- ❌ Missing labels/legends
- ❌ Wrong chart type for data
- ✅ Clean, clear, informative visuals

### 7.3 Performance Tips

```python
# Don't create new figures unnecessarily
# Reuse figure objects

# Good
fig, ax = plt.subplots()
ax.plot(x, y)

# Also good for interactive
fig = go.Figure()
fig.add_trace(...)

# Clear figures when done
plt.close('all')
```

---

## 8. Next Steps

Congratulations! You've mastered data visualization in Python. You now know how to:

- Create static plots with Matplotlib
- Design beautiful statistical graphics with Seaborn
- Build interactive visualizations with Plotly
- Perform comprehensive EDA

**Coming Up Next:**

- Day 7-8: Machine Learning Foundations
- Scikit-learn basics
- Classification and regression
- Model evaluation

---

## Additional Resources

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Plotly Python Guide](https://plotly.com/python/)
- [From Data to Viz](https://www.data-to-viz.com/) - Chart selection guide
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

**Practice Datasets:**

- Iris dataset: `sns.load_dataset('iris')`
- Tips dataset: `sns.load_dataset('tips')`
- Titanic: `sns.load_dataset('titanic')`
- Gapminder: `px.data.gapminder()`

---

**Happy Visualizing! 📊✨**
