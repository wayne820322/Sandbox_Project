#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from Analyzer import Analyzer
from Classifier import Classifier, KNNClassifier, DecisionTree, RandomForest, SupportVectorClassifier, ArtificialNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    analyzer = Analyzer()
    analyzer.read_dataset('diamonds.csv')
    analyzer.describe()
    analyzer.drop_missing_data()
    columns_to_drop = ["Unnamed: 0"]
    analyzer.drop_columns(columns_to_drop)
    features_list = ["clarity", "color", "cut"]
    analyzer.encode_features(features_list)
    X = analyzer.retrieve_data().drop(["cut"], axis=1)
    y = analyzer.retrieve_data()["cut"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    # Logistic Regression
#     classifier = Classifier(X_train, y_train, X_test, y_test)
#     accuracy = classifier.score()
#     print(f"Accuracy: {accuracy:.2f}")

    # KNN-Classifier
#     knn_classifier = KNNClassifier(X_train, y_train, X_test, y_test)
#     optimal_k, accuracy = knn_classifier.train_knn(max_k=10)
#     print(f"Optimal k: {optimal_k}, Accuracy: {accuracy}")
    
    # Decision Tree
#     dt_classifier = DecisionTree(X_train, y_train, X_test, y_test)
#     criteria = ['gini', 'entropy']
#     optimal_criterion, accuracy = dt_classifier.train_decision_tree(criteria)
#     print(f"Optimal Criterion: {optimal_criterion}, Accuracy: {accuracy}")


    # Random Forest
#     rf_classifier = RandomForest(X_train, y_train, X_test, y_test)
#     criteria = ['gini', 'entropy']  
#     n_estimators_list = [10, 50, 100]
#     optimal_criterion, optimal_estimators, accuracy = rf_classifier.train_random_forest(criteria, n_estimators_list)
#     print(f"Optimal Criterion: {optimal_criterion}, Optimal Estimators: {optimal_estimators}, Accuracy: {accuracy}")

    # SVC
#     svc_classifier = SupportVectorClassifier(X_train, y_train, X_test, y_test)
#     accuracy = svc_classifier.train_svc()
#     print(f"Accuracy: {accuracy}")

    # ANN
#     ann_classifier = ArtificialNeuralNetwork(X_train, y_train, X_test, y_test)
#     conf_matrix = ann_classifier.train_ann()
#     print("Confusion Matrix:")
#     print(conf_matrix)

    
if __name__ == "__main__":
    main()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Regressor import Regressor, KNNRegressor, DecisionTree, RandomForest, SupportVectorRegressor, NeuralNetworkRegressor

def main():
    analyzer = Analyzer()
    analyzer.read_dataset('diamonds.csv')
    analyzer.describe()
    analyzer.drop_missing_data()
    columns_to_drop = ["Unnamed: 0"]
    analyzer.drop_columns(columns_to_drop)
    features_list = ["clarity", "color", "cut"]
    analyzer.encode_features(features_list)
    X = analyzer.retrieve_data().drop(["price"], axis=1)
    y = analyzer.retrieve_data()["price"]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=2)
    
    # Linear Regression
#     regressor = Regressor(train_x, train_y, test_x, test_y)

    # KNN-Regressor
#     knn_regressor = KNNRegressor(train_x, train_y, test_x, test_y)
#     max_k = 10 
#     scores = knn_regressor.train_knn(max_k)
#     # Print R-squared, MSE, and MAE scores
#     for metric, values in scores.items():
#         print(f"{metric} scores for different k values:")
#         for k, score in enumerate(values, start=1):
#             print(f"K={k}: {score}")

    # Decision Tree
#     decision_tree = DecisionTree(train_x, train_y, test_x, test_y)
#     criteria = ["mse", "friedman_mse", "mae"]  # List of splitting criteria to try
#     scores = decision_tree.train_decision_tree(criteria)
#     print("R-squared scores for different criteria:", scores)
#     print("Optimal criterion:", decision_tree.optimal_criterion)

    # Random Forest
#     random_forest = RandomForest(train_x, train_y, test_x, test_y)
#     criteria = ["mse"]  
#     estimators = [20, 100]
#     scores = random_forest.train_random_forest(criteria, estimators)
#     print("R-squared scores for different criteria and estimators:", scores)
#     print("Optimal criterion:", random_forest.optimal_criteria)
#     print("Optimal number of estimators:", random_forest.optimal_estimators)

    # SVC
#     sv_regressor = SupportVectorRegressor(train_x, train_y, test_x, test_y)
#     sv_regressor.train_svr()
#     scores = sv_regressor.evaluate()
#     print("Evaluation Scores:")
#     for metric, value in scores.items():
#         print(f"{metric}: {value}")

    # ANN
#     nn_regressor = NeuralNetworkRegressor(train_x, train_y, test_x, test_y)
#     nn_regressor.build_model()
#     nn_regressor.train_model(epochs=50, batch_size=32)
#     r2 = nn_regressor.evaluate()
#     print("R-squared score:", r2)

if __name__ == "__main__":
    main()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from Clustering import Clustering, KMeansClustering, AgglomerativeClusteringExample, MeanShiftClustering

def main():
    analyzer = Analyzer()
    analyzer.read_dataset('diamonds.csv')
    analyzer.drop_missing_data()
    analyzer.sample(0.1)
    columns_to_drop = ["Unnamed: 0"]
    analyzer.drop_columns(columns_to_drop)
    features_list = ["clarity", "color", "cut"]
    analyzer.encode_features(features_list)
    df = analyzer.retrieve_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
#     kmeans_cluster = KMeansClustering(X)
#     kmeans_cluster.fit(max_k=9)
#     optimal_k = 3
#     kmeans_cluster.fit_with_optimal_k(optimal_k)
#     inertia = kmeans_cluster.retrieve_inertia()
#     print(f"Inertia of the KMeans estimator: {inertia}")


#     agg_clustering = AgglomerativeClusteringExample(X, 3)
#     agg_clustering.fit()
#     agg_clustering.plot_clusters()
#     predicted_labels = agg_clustering.labels
    
    meanshift_cluster = MeanShiftClustering(X)
    clusters_mean_shift = meanshift_cluster.fit()
    predicted_labels = meanshift_cluster.labels
    plt.scatter(X[:, 0], X[:, 1], c = clusters_mean_shift, cmap="rainbow")
    plt.show()
    
    
   
if __name__ == "__main__":
    main()

