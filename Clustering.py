#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Clustering:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.trained_model = None

    def Fit(self, estimator):
        self.trained_model = estimator.fit(self.train_x)

    def Predict(self, new_data):
        self.trained_model.predict(new_data)
    


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, data):
        self.data = data
        self.optimal_k = None
        self.model = None
        self.inertia_values = []

    def fit(self, max_k=10):
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data)
            self.inertia_values.append(kmeans.inertia_)

        # Plot the elbow curve to find optimal K
        self.plot_elbow_curve(max_k)

    def plot_elbow_curve(self, max_k):
        k_values = range(1, max_k + 1)
        plt.plot(k_values, self.inertia_values, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.show()

    def retrieve_inertia(self):
        if self.model:
            return self.model.inertia_
        else:
            raise ValueError("Fit the model first to retrieve inertia")

    def fit_with_optimal_k(self, k):
        self.optimal_k = k
        self.model = KMeans(n_clusters=k)
        self.model.fit(self.data)

# Usage example:
# Assuming 'data' contains your dataset
kmeans_cluster = KMeansClustering(data)
kmeans_cluster.fit(max_k=10)  # Try K values from 1 to 10 and plot the elbow curve

# Retrieve inertia for the KMeans estimator
inertia = kmeans_cluster.retrieve_inertia()
print(f"Inertia of the KMeans estimator: {inertia}")

# Fit KMeans with the optimal K (selected visually from the elbow curve)
optimal_k = 3  # Example: Choosing K=3 as the optimal value
kmeans_cluster.fit_with_optimal_k(optimal_k)


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

class AgglomerativeClusteringExample:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters
        self.labels = None

    def fit(self):
        agglomerative = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.labels = agglomerative.fit_predict(self.data)

# Usage example:
# Assuming 'data' contains your dataset and 'n_clusters' is the desired number of clusters
agglomerative_cluster = AgglomerativeClusteringExample(data, n_clusters=3)
agglomerative_cluster.fit()
# Retrieve the predicted labels
predicted_labels = agglomerative_cluster.labels


# In[ ]:


from sklearn.cluster import MeanShift

class MeanShiftClustering:
    def __init__(self, data):
        self.data = data
        self.labels = None

    def fit(self):
        meanshift = MeanShift()
        self.labels = meanshift.fit_predict(self.data)

# Usage example:
# Assuming 'data' contains your dataset
mean_shift_cluster = MeanShiftClustering(data)
mean_shift_cluster.fit()
# Retrieve the predicted labels
predicted_labels = mean_shift_cluster.labels

