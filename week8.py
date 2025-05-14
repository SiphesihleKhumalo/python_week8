# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set visualization style
plt.style.use('ggplot')
sns.set_palette("husl")

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')


# Load the dataset
try:
    covid_df = pd.read_csv('owid-covid-data.csv')
    print("Dataset loaded successfully with shape:", covid_df.shape)
except FileNotFoundError:
    print("Error: File not found. Please download the dataset first.")
    # Alternative loading method if available
    # covid_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

# Display basic information
print("\nDataset columns:")
print(covid_df.columns.tolist())

print("\nFirst 5 rows:")
display(covid_df.head())

print("\nMissing values summary:")
print(covid_df.isnull().sum().sort_values(ascending=False)[:20])  # Show top 20 columns with missing values

print("\nData types:")
print(covid_df.dtypes.value_counts())


# Convert date column to datetime
covid_df['date'] = pd.to_datetime(covid_df['date'])

# Select countries of interest
countries = ['Kenya', 'United States', 'India', 'Brazil', 'Germany', 'South Africa']
filtered_df = covid_df[covid_df['location'].isin(countries)].copy()

# Handle missing values for key columns
key_columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
               'total_vaccinations', 'people_vaccinated', 'population']

# Forward fill for time-series data
filtered_df[key_columns] = filtered_df.groupby('location')[key_columns].fillna(method='ffill')

# Fill remaining NAs with 0 where appropriate
filtered_df[key_columns] = filtered_df[key_columns].fillna(0)

# Calculate derived metrics
filtered_df['death_rate'] = filtered_df['total_deaths'] / filtered_df['total_cases']
filtered_df['vaccination_rate'] = filtered_df['people_vaccinated'] / filtered_df['population']

# Filter to most recent 12 months for analysis
latest_date = filtered_df['date'].max()
start_date = latest_date - pd.DateOffset(months=12)
filtered_df = filtered_df[filtered_df['date'] >= start_date]

print("\nCleaned data shape:", filtered_df.shape)
print("Date range:", filtered_df['date'].min(), "to", filtered_df['date'].max())


# Create a summary dataframe for latest data
latest_data = filtered_df[filtered_df['date'] == latest_date]

# Plot total cases by country
plt.figure(figsize=(12, 6))
sns.barplot(data=latest_data.sort_values('total_cases', ascending=False),
            x='location', y='total_cases')
plt.title(f'Total COVID-19 Cases by Country (as of {latest_date.date()})')
plt.ylabel('Total Cases (millions)')
plt.xticks(rotation=45)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Plot new cases over time
plt.figure(figsize=(14, 7))
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases'], label=country)
plt.title('Daily New COVID-19 Cases (Last 12 Months)')
plt.ylabel('New Cases')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Death rate comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='location', y='death_rate')
plt.title('COVID-19 Death Rate Distribution by Country')
plt.ylabel('Death Rate (Deaths/Cases)')
plt.xticks(rotation=45)
plt.show()

# Plot vaccination progress
plt.figure(figsize=(14, 7))
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['vaccination_rate'], label=country)
plt.title('Vaccination Rate Over Time (Last 12 Months)')
plt.ylabel('Vaccination Rate (% Population)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Latest vaccination status
plt.figure(figsize=(10, 6))
sns.barplot(data=latest_data.sort_values('vaccination_rate', ascending=False),
            x='location', y='vaccination_rate')
plt.title(f'Vaccination Rate by Country (as of {latest_date.date()})')
plt.ylabel('Percentage of Population Vaccinated')
plt.xticks(rotation=45)
plt.show()


# Correlation heatmap
corr_columns = ['total_cases', 'total_deaths', 'people_vaccinated', 'population', 'gdp_per_capita']
corr_df = latest_data[corr_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of COVID-19 Metrics')
plt.show()

# Facet grid of cases vs deaths
g = sns.FacetGrid(filtered_df, col='location', col_wrap=3, height=4)
g.map(sns.scatterplot, 'total_cases', 'total_deaths', alpha=0.6)
g.set_titles("{col_name}")
g.set_axis_labels("Total Cases", "Total Deaths")
plt.suptitle('Total Cases vs Total Deaths by Country', y=1.05)
plt.show()

# Save cleaned data for future use
filtered_df.to_csv('cleaned_covid_data.csv', index=False)

# Generate a report (requires additional libraries)
try:
    from pandas_profiling import ProfileReport
    profile = ProfileReport(filtered_df, title="COVID-19 Data Profiling Report")
    profile.to_file("covid19_report.html")
except ImportError:
    print("Pandas profiling not installed. Install with: pip install pandas-profiling")