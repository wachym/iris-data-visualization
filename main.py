# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable Seaborn styling
sns.set(style="whitegrid")

# -----------------------
# Task 1: Load and Explore
# -----------------------

try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map(dict(enumerate(iris.target_names)))
    print("Data loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------
# Task 2: Basic Analysis
# -----------------------

print("\nDescriptive Statistics:")
print(df.describe())

print("\nAverage Features per Species:")
print(df.groupby('species').mean())

# -----------------------
# Task 3: Visualizations
# -----------------------

# 1. Line Chart (simulated trend)
df['index'] = df.index
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='index', y='sepal length (cm)', label='Sepal Length')
plt.title("Sepal Length Over Index (Fake Time)")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()

# 2. Bar Chart - Petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# 3. Histogram - Sepal width
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# 4. Scatter Plot - Sepal vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# -----------------------
# Observations
# -----------------------

print("\nObservations:")
print("- Setosa flowers have significantly shorter petal lengths.")
print("- Virginica generally has the longest petals and sepals.")
print("- Clear positive correlation between petal length and sepal length.")
print("- Sepal width varies more across samples, visible in histogram.")
