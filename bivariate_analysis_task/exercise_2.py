# Exercise 2: Continue with Class 6 Exercise 4.
# Open data.csv, add a new column (categories for calories: few,
# normal or high). Apply label encoding / ordinal encoding / one-hot encoding to
# this new feature. Study correlation between duration and encoded calories features.


# Importing libraries
import pandas as pd
import numpy as np
from scipy import stats
# Label encoding categorical data
from sklearn.preprocessing import LabelEncoder
# One-hot encoding using scikit-learn
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

'''                    Label Encoding / correlation                  '''

#
df = pd.read_csv("data.csv")
# fetch the last colomn
stringCol = df.iloc[:, -1]
print(stringCol)

# applying label encoding to the last colomn
encoder = LabelEncoder()
encoder.fit(stringCol)
encoder.transform(stringCol)
# Replace cat_cal values with encoded labels
df["cat_calories"].replace(to_replace=df["cat_calories"].tolist(),
                           value=encoder.transform(stringCol),
                           inplace=True)
print(df.head())

# Visualizing data
df.plot()
plt.show()
# scatter plot for two attributes
df.plot(kind='scatter', x='Duration', y='cat_calories')
plt.scatter(x=df['Duration'], y=df['cat_calories'])
plt.show()
df["cat_calories"].plot(kind='hist')
plt.show()

# correlation matrix
sns.set(style='white', context='notebook', palette='deep')
# Correlation matrix
corrMatrix = df.corr()
print(corrMatrix)
# Visualizing correlation matrix
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Analysing correlation between Duration and Calories
sns.jointplot(x="Duration", y="cat_calories", data=df)
plt.show()
plt.scatter(x="Duration", y="cat_calories", data=df)
plt.show()

# Correlation coefficient
corr = np.corrcoef(df["Duration"], df["cat_calories"])[0, 1]
print(corr)
print("Correlation between Duration and cat_calories:", round(corr, 2))

# Significance of correlation coefficient
ttest, pval = stats.ttest_ind(df["Duration"], df["cat_calories"])
print("Independent t-test:", ttest, pval)

'''                    one hot encoding / correlation                  '''

df = pd.read_csv("data.csv")
# Instantiate the OneHotEncoder object
# The parameter drop = ‘first’ will handle dummy variable traps
onehotencoder = OneHotEncoder(sparse=False, handle_unknown='error',
                              drop='first')

# Perform one-hot encoding
onehotencoder_df = pd.DataFrame(onehotencoder.fit_transform(df[["cat_calories"]]))

# Merge one-hot encoding columns with dataframe
df = df.join(onehotencoder_df)
# drope the previous colom
df.drop(columns=['cat_calories'], inplace=True)
print(df.head())

sns.set(style='white', context='notebook', palette='deep')
# Correlation matrix
corrMatrix = df.corr()
print(corrMatrix)

corr = df.corr()
# Visualizing correlation matrix
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

'''                     Ordinal Encoding / correlation                  '''

df = pd.read_csv("data.csv")
# Create dictionary for mapping the ordinal numerical value
cat_cal_dict = {'few': 45, 'normal': 60, 'high': 75}
# Assign ordinal numerical value
df['cat_calories'] = df.cat_calories.map(cat_cal_dict)
print(df.head())

# Visualizing data
df.plot()
plt.show()
# scatter plot for two attributes
df.plot(kind='scatter', x='Duration', y='cat_calories')
plt.scatter(x=df['Duration'], y=df['cat_calories'])
plt.show()
df["cat_calories"].plot(kind='hist')
plt.show()

sns.set(style='white', context='notebook', palette='deep')
# Correlation matrix
corrMatrix = df.corr()
print(corrMatrix)

# Visualizing correlation matrix
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Analysing correlation between Duration and Calories
sns.jointplot(x="Duration", y="cat_calories", data=df)
plt.show()

plt.scatter(x="Duration", y="cat_calories", data=df)
plt.show()

# Correlation coefficient
corr = np.corrcoef(df["Duration"], df["cat_calories"])[0, 1]

print(corr)
print("Correlation between Duration and cat_calories:", round(corr, 2))

# Significance of correlation coefficient
ttest, pval = stats.ttest_ind(df["Duration"], df["cat_calories"])
print("Independent t-test:", ttest, pval)
