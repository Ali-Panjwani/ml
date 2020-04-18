import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd

# Importing Data
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, -1].values

# Using the Dendogram to find optimal no of clusters
from scipy.cluster import hierarchy as ch
dendogram = ch.dendrogram(ch.linkage(X, method = 'ward'))
plt.title('The Dendogram Method')
plt.xlabel('Customer Clusters')
plt.ylabel('Euclidean Distance')
plt.show()


# # Applying the algorithm to the dataset
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_ac = ac.fit_predict(X)

# Visualizing The Clusters
plt.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], s = 10, c = 'yellow', label = 'Cluster 2')
plt.scatter(X[y_ac == 2, 0], X[y_ac == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_ac == 3, 0], X[y_ac == 3, 1], s = 10, c = 'orange', label = 'Cluster 4')
plt.scatter(X[y_ac == 4, 0], X[y_ac == 4, 1], s = 10, c = 'blue', label = 'Cluster 5')
plt.title('Clusters of Clients')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.show()

