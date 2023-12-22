#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


class Clustering:
    def __init__(self, train_x, train_y):
        """
        Initialize the Clustering model with training data.
        
        Parameters:
        train_x: Training features.
        train_y: Training labels (if applicable).
        """
        self.train_x = train_x
        self.train_y = train_y
        self.trained_model = None

    def fit(self, estimator):
        """
        Fit the clustering model.
        
        Parameters:
        estimator: The clustering estimator to be trained.
        """
        self.trained_model = estimator.fit(self.train_x)

    def predict(self, new_data):
        """
        Predict clusters for new data using the trained model.
        
        Parameters:
        new_data: New data to predict clusters for.
        
        """
        self.trained_model.predict(new_data)


# In[3]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, data):
        """
        Initialize the KMeansClustering model with data for clustering.
        
        Parameters:
        data: Input data for clustering.
        """
        self.data = data
        self.optimal_k = None
        self.model = None
        self.inertia_values = []

    def fit(self, max_k):
        """
        Fit the KMeans clustering model for different values of K and generate an elbow curve.
        
        Parameters:
        max_k (int): Maximum number of clusters (K) to consider.
        """
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters = k, random_state = 0)
            kmeans.fit(self.data)
            self.inertia_values.append(kmeans.inertia_)

        # Plot the elbow curve to find optimal K
        self.plot_elbow_curve(max_k)

    def plot_elbow_curve(self, max_k):
        """
        Plot the elbow curve to visualize inertia values for different values of K.
        
        Parameters:
        max_k (int): Maximum number of clusters (K) considered for plotting.
        """
        k_values = range(2, max_k + 1)
        plt.plot(k_values, self.inertia_values, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.show()

    def retrieve_inertia(self):
        """
        Retrieve the inertia value of the fitted KMeans model.
        """
        if self.model:
            return self.model.inertia_
        else:
            raise ValueError("Fit the model first to retrieve inertia")

    def fit_with_optimal_k(self, k):
         """
        Fit the KMeans model with the optimal number of clusters (K).
        
        Parameters:
        k (int): Number of clusters (K) to fit the model with.
        """
        self.optimal_k = k
        self.model = KMeans(n_clusters=k)
        self.model.fit(self.data)


# In[4]:


from sklearn.cluster import AgglomerativeClustering

class AgglomerativeClusteringExample:
    def __init__(self, data, n_clusters):
        """
        Initialize the AgglomerativeClusteringExample with data and number of clusters.
        
        Parameters:
        data: Input data for clustering.
        n_clusters: Number of clusters for agglomerative clustering.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.labels = None

    def fit(self):
        """
        Fit the Agglomerative Clustering model and obtain cluster labels.
        """
        agglomerative = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.labels = agglomerative.fit_predict(self.data)
    def plot_clusters(self):
        """
        Plot the clustered data points.
        """
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap="rainbow")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Agglomerative Clustering with {self.n_clusters} clusters')
        plt.show()


# In[5]:


from sklearn.cluster import MeanShift

class MeanShiftClustering:
    def __init__(self, data):
        """
        Initialize the MeanShiftClustering with data for clustering.
        
        Parameters:
        data: Input data for clustering.
        """
        self.data = data
        self.labels = None

    def fit(self):
        """
        Fit the MeanShift clustering model and obtain cluster labels.
        """
        meanshift = MeanShift()
        self.labels = meanshift.fit_predict(self.data)

