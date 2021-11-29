# Exercise 4: Create categories of numeric data, i.e. label numeric data with categorical data. Open
# the cleaned data.csv and add categories according to Calories.
# If Calories <= Q1 → few
# If Calories > Q1 and Calories < Q3 → normal
# If Calories >= Q3 → high
# a) Insert a new column into the data frame to store the appropriate text.
# b) Calculate group mean Calories.

# Importing libraries
import pandas as pd

# read the modified data set ( the Calories category added manually )
df = pd.read_csv("data.csv")
print(df)

# Calculate means by groups
print(df.groupby(['Calories']).mean())
