import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the cleaned dataset from the CSV file
df_cleaned = pd.read_csv('cleaned_iris_dataset.csv')

print(df_cleaned.head())  # Inspect the data


df_cleaned['sepal length (cm)'].plot(kind= 'line', color= 'green', linestyle= '--', linewidth= 2)
plt.show()


df_cleaned.plot(kind='bar', x=df_cleaned['sepal width (cm)'])
plt.show()

df_cleaned.plot(kind='histogram', x=df_cleaned['sepa length (cm)'], y=df_cleaned['sepal width (cm)'])
plt.show()

df_cleaned.plot(kind='scatter plot', x=df_cleaned['sepa length (cm)'], y=df_cleaned['sepal width (cm)'])
plt.show()

x=df_cleaned['sepa length (cm)']
y=df_cleaned['sepal width (cm)']

plt.scatter(x, y)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Scatter plot of sepa length against sepal width (cm)')

plt.show()