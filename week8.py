import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Task 1: Load and Explore the Dataset
# Using the Iris dataset
iris_data = load_iris()
iris = pd.DataFrame(data=iris_data['data'], columns=iris_data['feature_names'])
iris['species'] = pd.Categorical.from_codes(iris_data['target'], iris_data['target_names'])

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris.head())

# Explore the structure of the dataset
print("\nDataset information:")
iris.info()

# Check for missing values
print("\nMissing values:")
print(iris.isnull().sum())

# No missing values were found, but if there were, we could use:
# iris.fillna(iris.mean(), inplace=True)  # Fill missing values with the mean
# iris.dropna(inplace=True)             # Drop rows with missing values

# Task 2: Basic Data Analysis
# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(iris.describe())

# Perform groupings on the 'species' column and compute the mean of 'sepal length (cm)'
print("\nMean sepal length per species:")
print(iris.groupby('species')['sepal length (cm)'].mean())

# Identify patterns or interesting findings
# From the basic statistics, we can see the range of values for each numerical feature.
# From the grouped data, we can see how the mean sepal length varies across different species.
# For example, we can observe the mean sepal length for each species.
# Task 3: Data Visualization
# 1. Line chart showing trends over time (not applicable for this dataset, so I'll plot sepal length vs sample number)
plt.figure(figsize=(10, 5))
plt.plot(iris['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Trend')
plt.xlabel('Sample Number')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar chart showing the comparison of a numerical value across categories
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=iris)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram of a numerical column to understand its distribution
plt.figure(figsize=(8, 6))
sns.histplot(iris['sepal width (cm)'], kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot to visualize the relationship between two numerical columns
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris)
plt.title('Relationship between Sepal Length and Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()
