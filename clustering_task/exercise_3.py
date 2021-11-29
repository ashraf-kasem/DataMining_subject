# Clustering of shopping data

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster

# shopping dataset

shopping = pd.read_csv("shopping-data.csv")
# rab the values of the last 2 columns to a list x
x = shopping.iloc[:, [2, 3]].values
print(x)
shopping.info()
print(shopping[0:10])

# Frequency distribution of gender
# Make a crosstab and gender the count column
shopping_outcome = pd.crosstab(index=shopping["Genre"], columns="count")
print(shopping_outcome)

# Histograms of gender spending, outcome
sns.FacetGrid(shopping, hue="Genre", height=3).map(sns.distplot, "Annual Income (k$)").add_legend()
sns.FacetGrid(shopping, hue="Genre", height=3).map(sns.distplot, "Spending Score (1-100)").add_legend()
plt.show()

# Visualizing data distribution
# Scatter plot of features according to species
sns.set_style("whitegrid")
sns.pairplot(shopping, hue="Genre", height=3);
plt.show()

# Draw dendograms


X = shopping.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]]
# Try different types of linkage method
dist_sin = linkage(X, method="single")
plt.figure(figsize=(18, 6))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM", fontsize=18)

# Add horizontal line at a given distance
plt.axhline(y=8, c='grey', lw=1, linestyle='dashed')
plt.show()

# Apply the HAC algorithm
# Number of clusters : 15


cluster = AgglomerativeClustering(n_clusters=15, affinity='euclidean',
                                  linkage='single')
cluster.fit_predict(X)
print(cluster.labels_)

# Plot the clusters
# Plot Annual Income (k$) and Spending Score (1-100)
data = X.iloc[:, 0:2].values
plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
plt.show()

# Evaluation of hierarchical clustering

# Add hierarchical clustering result to data set

shopping_HAC = shopping.copy()

# fcluster : forms flat clusters from the hierarchical clustering
# defined by the given linkage matrix
shopping_HAC['K=15'] = fcluster(dist_sin, 15, criterion='maxclust')
print(shopping_HAC.head())


plt.figure(figsize=(24, 10))

plt.suptitle("Hierarchical Clustering Single Method ", fontsize=18)

plt.subplot(1, 2, 1)
plt.title("K = 15", fontsize=14)
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=shopping_HAC, hue="K=15")


plt.subplot(1, 2, 2)
plt.title("Species", fontsize=14)
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=shopping_HAC, hue="Genre")
plt.show()