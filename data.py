import pandas as pd 
from sklearn import datasets
 
iris = datasets.load_iris()
 
df = pd.DataFrame(
    iris.data, 
    columns=iris.feature_names
    )
 
df['target'] = iris.target
 
# Map targets to target names
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}
 
df['target_names'] = df['target'].map(target_names)
print(df.head())


# Checking the data types of each column
print("Data Types:\n", df.dtypes)

# Checking for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Basic overview of the dataset
print("\nDataset Info:")
df.info()


# Handle missing values
# Fill numerical columns with their mean
df['sepal length (cm)'] = df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean())
df['sepal width (cm)'] = df['sepal width (cm)'].fillna(df['sepal width (cm)'].mean())

# Fill categorical column with the most frequent value (mode)
df['target_names'] = df['target_names'].fillna(df['target_names'].mode()[0])

# Verify no missing values remain
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Display the cleaned dataset
print("\nCleaned Dataset:\n", df)

# Save the cleaned dataset to a CSV file
df.to_csv('cleaned_iris_dataset.csv', index=False)
