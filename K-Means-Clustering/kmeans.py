import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Customers.csv')

# Using Annual Income(3) and Spending score(4) to track the data in the following command.
# (3,4 are the columns numbered from 0-4 i.e 5 columns)
#The iloc function in pandas is used to return the selected rows and columns.
# Here's the syntax.
# pandas_array.iloc[ [ <Row_Selection> ] ,[ <Column_Selection> ] ]
# .values is added to get the values in the selected fields.

X=dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting K-MEans to the dataset
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualize the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=50,c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=50,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=50,c='green',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=50,c='cyan',label='Cluster4')
# plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='Yellow',label='Centroids')

plt.title('Clusters of Outfits')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()


cluster_map = pd.DataFrame()
cluster_map['data_index'] = dataset.index.values
cluster_map['cluster'] = kmeans.labels_
print("============= C1 =============")
print(cluster_map[cluster_map.cluster == 0])
print("============= C2 =============")
print(cluster_map[cluster_map.cluster == 1])
print("============= C3 =============")
print(cluster_map[cluster_map.cluster == 2])
print("============= C4 =============")
print(cluster_map[cluster_map.cluster == 3])


















