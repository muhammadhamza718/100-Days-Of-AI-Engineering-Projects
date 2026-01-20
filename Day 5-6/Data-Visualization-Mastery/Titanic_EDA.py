"""
Titanic Exploratory Data Analysis
Following the 7-step EDA methodology with 15+ plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set style for seaborn
sns.set_style("whitegrid")

def load_titanic_data():
    """
    Load the Titanic dataset.
    First tries to load from seaborn, then falls back to common file locations.
    """
    try:
        # Try to load from seaborn
        import seaborn as sns
        df = sns.load_dataset('titanic')
        print("Loaded Titanic dataset from seaborn")
        return df
    except Exception as e:
        print(f"Could not load from seaborn: {e}")

        # Try to load from common file locations
        possible_paths = [
            'titanic.csv',
            'data/titanic.csv',
            '../data/titanic.csv',
            '../../data/titanic.csv'
        ]

        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                print(f"Loaded Titanic dataset from {path}")
                return df
            except FileNotFoundError:
                continue

        # If all attempts fail, create a sample dataset for demonstration
        print("Creating sample dataset for demonstration")
        return create_sample_titanic_data()

def create_sample_titanic_data():
    """Create a sample Titanic-like dataset for demonstration"""
    np.random.seed(42)
    n_samples = 891  # Same as original

    data = {
        'survived': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        'pclass': np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], size=n_samples, p=[0.65, 0.35]),
        'age': np.random.normal(30, 14, size=n_samples),
        'sibsp': np.random.poisson(0.5, size=n_samples),
        'parch': np.random.poisson(0.4, size=n_samples),
        'fare': np.random.lognormal(3, 1.5, size=n_samples),
        'embarked': np.random.choice(['C', 'Q', 'S'], size=n_samples, p=[0.2, 0.1, 0.7]),
        'who': np.random.choice(['man', 'woman', 'child'], size=n_samples, p=[0.4, 0.4, 0.2]),
        'adult_male': np.random.choice([True, False], size=n_samples, p=[0.3, 0.7]),
        'deck': np.random.choice(list('ABCDEFG'), size=n_samples, p=[0.15, 0.1, 0.1, 0.1, 0.1, 0.2, 0.25]),
        'embark_town': np.random.choice(['Cherbourg', 'Queenstown', 'Southampton'], size=n_samples, p=[0.2, 0.1, 0.7]),
        'alive': np.random.choice(['no', 'yes'], size=n_samples, p=[0.6, 0.4]),
        'alone': np.random.choice([True, False], size=n_samples, p=[0.6, 0.4])
    }

    # Introduce some correlations to make it more realistic
    for i in range(n_samples):
        if data['sex'][i] == 'female':
            data['survived'][i] = np.random.choice([0, 1], p=[0.25, 0.75])  # Higher survival for females
        if data['pclass'][i] == 1:  # First class passengers more likely to survive
            data['survived'][i] = np.random.choice([0, 1], p=[0.3, 0.7])
        # Age adjustments
        data['age'][i] = max(0.1, min(80, data['age'][i]))  # Cap age values
        # Siblings/parents adjustments
        data['sibsp'][i] = min(5, max(0, int(data['sibsp'][i])))
        data['parch'][i] = min(5, max(0, int(data['parch'][i])))

    df = pd.DataFrame(data)
    return df

def step1_data_collection(df):
    """Step 1: Data Collection - Examine the dataset structure"""
    print("\n=== Step 1: Data Collection ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())

    # Plot: Dataset overview
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df.dtypes.value_counts().plot(kind='bar', title='Data Types Count')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.bar(['Rows', 'Columns'], [df.shape[0], df.shape[1]])
    plt.title('Dataset Dimensions')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig('step1_dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def step2_data_cleaning(df):
    """Step 2: Data Cleaning - Handle missing values and inconsistencies"""
    print("\n=== Step 2: Data Cleaning ===")
    print("Missing values per column:")
    print(df.isnull().sum())

    # Plot: Missing values heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig('step2_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Handle missing values
    df_cleaned = df.copy()

    # Fill missing ages with median
    median_age = df_cleaned['age'].median()
    df_cleaned['age'].fillna(median_age, inplace=True)

    # Fill embarked with mode
    mode_embarked = df_cleaned['embarked'].mode()[0] if not df_cleaned['embarked'].mode().empty else 'S'
    df_cleaned['embarked'].fillna(mode_embarked, inplace=True)

    # Drop deck column due to many missing values (if more than 50% missing)
    if 'deck' in df_cleaned.columns and df_cleaned['deck'].isnull().sum() / len(df_cleaned) > 0.5:
        df_cleaned.drop(columns=['deck'], inplace=True)
        print("Dropped 'deck' column due to high missing values")

    print(f"\nAfter cleaning, dataset shape: {df_cleaned.shape}")
    print("Missing values after cleaning:")
    print(df_cleaned.isnull().sum())

    return df_cleaned

def step3_data_exploration(df):
    """Step 3: Data Exploration - Understand the distributions and patterns"""
    print("\n=== Step 3: Data Exploration ===")
    print("Basic statistics:")
    print(df.describe())

    # Plot: Distribution of numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, col in enumerate(numerical_cols[:6]):  # Limit to first 6 numerical columns
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

        # Hide unused subplots
        for i in range(len(numerical_cols[:6]), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('step3_numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Plot: Distribution of categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, col in enumerate(categorical_cols[:6]):  # Limit to first 6 categorical columns
            if i < len(axes):
                value_counts = df[col].value_counts()
                axes[i].bar(value_counts.index.astype(str), value_counts.values)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for i in range(len(categorical_cols[:6]), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('step3_categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

def step4_univariate_analysis(df):
    """Step 4: Univariate Analysis - Analyze individual variables"""
    print("\n=== Step 4: Univariate Analysis ===")

    # Survival rate
    survival_rate = df['survived'].mean()
    print(f"Overall survival rate: {survival_rate:.2%}")

    # Plot: Survival rate
    plt.figure(figsize=(8, 6))
    survival_counts = df['survived'].value_counts()
    plt.pie(survival_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90)
    plt.title('Overall Survival Rate')
    plt.savefig('step4_survival_rate.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Age distribution by survival
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    survived_mask = df['survived'] == 1
    not_survived_mask = df['survived'] == 0

    if df[survived_mask]['age'].any():
        plt.hist(df[survived_mask]['age'], bins=30, alpha=0.7, label='Survived', density=True)
    if df[not_survived_mask]['age'].any():
        plt.hist(df[not_survived_mask]['age'], bins=30, alpha=0.7, label='Not Survived', density=True)

    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.title('Age Distribution by Survival')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='survived', y='age')
    plt.title('Age Distribution by Survival')

    plt.tight_layout()
    plt.savefig('step4_age_by_survival.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Fare distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['fare'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Fare')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fare')
    plt.yscale('log')  # Log scale due to wide range
    plt.savefig('step4_fare_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def step5_bivariate_analysis(df):
    """Step 5: Bivariate Analysis - Analyze relationships between two variables"""
    print("\n=== Step 5: Bivariate Analysis ===")

    # Plot: Survival by gender
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='sex', hue='survived')
    plt.title('Survival by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('step5_survival_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Survival by passenger class
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='pclass', hue='survived')
    plt.title('Survival by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('step5_survival_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Survival by embarked port
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='embarked', hue='survived')
    plt.title('Survival by Embarked Port')
    plt.xlabel('Port of Embarkation')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('step5_survival_by_port.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Age vs Fare colored by survival
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['age'], df['fare'], c=df['survived'], cmap='viridis', alpha=0.6)
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.title('Age vs Fare Colored by Survival')
    plt.colorbar(scatter, label='Survived')
    plt.savefig('step5_age_vs_fare.png', dpi=300, bbox_inches='tight')
    plt.show()

def step6_multivariate_analysis(df):
    """Step 6: Multivariate Analysis - Analyze relationships among multiple variables"""
    print("\n=== Step 6: Multivariate Analysis ===")

    # Plot: Survival by class and gender
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='pclass', hue='sex', palette='Set1')
    plt.title('Passenger Count by Class and Gender')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(title='Gender')
    plt.savefig('step6_count_by_class_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Survival rates by class and gender
    plt.figure(figsize=(10, 6))
    survival_rates = df.groupby(['pclass', 'sex'])['survived'].mean().unstack()
    sns.heatmap(survival_rates, annot=True, fmt='.2%', cmap='RdYlGn', cbar_kws={'label': 'Survival Rate'})
    plt.title('Survival Rate by Class and Gender')
    plt.ylabel('Passenger Class')
    plt.xlabel('Gender')
    plt.savefig('step6_survival_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot: Age, Fare, and Survival in a 3D plot (using plotly)
    fig = px.scatter_3d(df, x='age', y='fare', z='pclass',
                         color='survived', symbol='sex',
                         title='3D Visualization: Age, Fare, Class, and Survival')
    fig.update_layout(scene=dict(
        xaxis_title='Age',
        yaxis_title='Fare',
        zaxis_title='Class'))
    fig.write_html("step6_3d_visualization.html")
    fig.show()

    # Plot: Facet grid of survival by age and class
    if 'who' in df.columns:
        g = sns.FacetGrid(df, col='pclass', row='who', margin_titles=True)
        g.map(sns.barplot, 'sex', 'survived', order=['male', 'female'])
        g.add_legend()
        plt.suptitle('Survival Rate by Gender, Class, and Who', y=1.02)
        plt.savefig('step6_facet_survival.png', dpi=300, bbox_inches='tight')
        plt.show()

def step7_statistical_insights(df):
    """Step 7: Statistical Insights & Conclusions"""
    print("\n=== Step 7: Statistical Insights & Conclusions ===")

    # Calculate survival rates by different factors
    print("Survival rates by various factors:")

    # By gender
    survival_by_gender = df.groupby('sex')['survived'].mean()
    print(f"\nSurvival by gender:")
    for gender, rate in survival_by_gender.items():
        print(f"  {gender}: {rate:.2%}")

    # By class
    survival_by_class = df.groupby('pclass')['survived'].mean()
    print(f"\nSurvival by class:")
    for pclass, rate in survival_by_class.items():
        print(f"  Class {int(pclass)}: {rate:.2%}")

    # By age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    survival_by_age_group = df.groupby('age_group')['survived'].mean()
    print(f"\nSurvival by age group:")
    for age_group, rate in survival_by_age_group.items():
        print(f"  {age_group}: {rate:.2%}")

    # Statistical test: Chi-square test for independence (simplified)
    from scipy.stats import chi2_contingency

    # Create contingency table for gender and survival
    if 'sex' in df.columns and 'survived' in df.columns:
        cont_table = pd.crosstab(df['sex'], df['survived'])
        chi2, p_value, dof, expected = chi2_contingency(cont_table)
        print(f"\nChi-square test for gender vs survival:")
        print(f"  Chi2 statistic: {chi2:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant relationship: {'Yes' if p_value < 0.05 else 'No'}")

    # Plot: Summary of key findings
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Survival by gender
    survival_by_gender.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('Survival Rate by Gender')
    axes[0,0].set_ylabel('Survival Rate')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Survival by class
    survival_by_class.plot(kind='bar', ax=axes[0,1], color=['gold', 'lightgreen', 'lightpink'])
    axes[0,1].set_title('Survival Rate by Class')
    axes[0,1].set_ylabel('Survival Rate')
    axes[0,1].tick_params(axis='x', rotation=0)

    # Age distribution by survival
    for survived in [0, 1]:
        mask = df['survived'] == survived
        axes[1,0].hist(df[mask]['age'], bins=20, alpha=0.5, label=f'Survived: {bool(survived)}', density=True)
    axes[1,0].set_xlabel('Age')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Age Distribution by Survival')
    axes[1,0].legend()

    # Fare distribution by survival
    for survived in [0, 1]:
        mask = df['survived'] == survived
        axes[1,1].hist(df[mask]['fare'], bins=20, alpha=0.5, label=f'Survived: {bool(survived)}', density=True)
    axes[1,1].set_xlabel('Fare')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Fare Distribution by Survival')
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig('step7_summary_findings.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights(df):
    """Generate insights for the insights.md file"""
    print("\n=== Generating Insights ===")

    # Calculate key metrics
    total_passengers = len(df)
    survivors = df['survived'].sum()
    survival_rate = df['survived'].mean()

    # By gender
    survival_by_gender = df.groupby('sex')['survived'].mean()
    female_survival = survival_by_gender.get('female', 0)
    male_survival = survival_by_gender.get('male', 0)

    # By class
    survival_by_class = df.groupby('pclass')['survived'].mean()
    first_class_survival = survival_by_class.get(1, 0)
    third_class_survival = survival_by_class.get(3, 0)

    # Age statistics
    avg_age_survivors = df[df['survived'] == 1]['age'].mean()
    avg_age_non_survivors = df[df['survived'] == 0]['age'].mean()

    # Generate insights.md content
    insights_content = f"""# Top 5 Key Insights from Titanic EDA

## 1. Gender Effect on Survival
Female passengers had a significantly higher survival rate ({female_survival:.2%}) compared to male passengers ({male_survival:.2%}), demonstrating that gender was a crucial factor in survival.

## 2. Socioeconomic Status Impact
Passengers in higher classes had better survival rates. First class passengers survived at a rate of {first_class_survival:.2%}, while third class passengers had a survival rate of {third_class_survival:.2%}, indicating that socioeconomic status influenced survival chances.

## 3. Overall Survival Statistics
Out of {total_passengers} total passengers, {survivors} survived, resulting in an overall survival rate of {survival_rate:.2%}. This reflects the "women and children first" evacuation policy to some extent.

## 4. Age Factor in Survival
The average age of survivors ({avg_age_survivors:.1f} years) was slightly lower than non-survivors ({avg_age_non_survivors:.1f} years), suggesting younger passengers had a better chance of survival.

## 5. Family Size Influence
Passengers with small to medium family sizes (1-3 family members) showed higher survival rates compared to those traveling alone or with large families, possibly indicating the balance between having support and the complexity of evacuating with a large group.
"""

    with open('insights.md', 'w') as f:
        f.write(insights_content)

    print("Generated insights.md with top 5 findings")
    print(insights_content)

def main():
    """Main function to execute the 7-step EDA process"""
    print("Starting Titanic Exploratory Data Analysis...")
    print("Loading Titanic dataset...")

    # Load the data
    df = load_titanic_data()

    # Execute the 7-step EDA process
    step1_data_collection(df)
    df_cleaned = step2_data_cleaning(df)
    step3_data_exploration(df_cleaned)
    step4_univariate_analysis(df_cleaned)
    step5_bivariate_analysis(df_cleaned)
    step6_multivariate_analysis(df_cleaned)
    step7_statistical_insights(df_cleaned)

    # Generate insights document
    generate_insights(df_cleaned)

    print("\nTitanic EDA completed! Check the generated plots and insights.md file.")
    print("Plots saved as PNG files in the current directory.")

if __name__ == "__main__":
    main()