#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

class Regressor:
    def __init__(self, train_x, train_y, test_x, test_y) -> None:
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.__logical_regression_classifier = LogisticRegression(random_state=16)
        self.Fit(self.__logical_regression_classifier)
        prediction = self.Predict(self.__logical_regression_classifier)
        logical_regression_score = self.Score(prediction, self.test_y, r2=True)
    def Fit(self, estimator):
        estimator.fit(self.__train_x, self.__train_y)
    def Predict(self, estimator):
        estimator.fit(self.__train_x, self.__train_y)
    def Score(self, data, true_labels, r2=False, mse=False, mae=False, rmse=False):
        scores = {}

        if r2:
            scores['R-squared'] = r2_score(data, true_labels)
        if mse:
            scores['Mean Squared Error'] = mean_squared_error(data, true_labels)
        if mae:
            scores['Mean Absolute Error'] = mean_absolute_error(data, true_labels)
        if rmse:
            mse_value = mean_squared_error(data, true_labels)
            scores['Root Mean Squared Error'] = (mse_value ** 0.5)

        return scores


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np

class KNNRegressor:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_k = None

    def train_knn(self, max_k):
        scores = []
        best_score = 0.0

        for k in range(1, max_k + 1):
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.__train_x, self.__train_y)
            predictions = knn.predict(self.test_x)
            score = r2_score(self.test_y, predictions)
            scores.append(score)
            if score > best_score:
                best_score = score
                self.optimal_k = k

        return scores

# Assuming train_x, train_y, test_x, test_y are already defined
knn_regressor = KNNRegressor(train_x, train_y, test_x, test_y)
max_k_to_try = 10  # Change as needed

scores = knn_regressor.train_knn(max_k_to_try)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class DecisionTree:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_criterion = None

    def train_decision_tree(self, criteria):
        scores = {}
        best_score = 0.0

        for criterion in criteria:
            dt = DecisionTreeRegressor(criterion=criterion)
            dt.fit(self.__train_x, self.__train_y)
            predictions = dt.predict(self.test_x)
            score = r2_score(self.test_y, predictions)
            scores[criterion] = score

            if score > best_score:
                best_score = score
                self.optimal_criterion = criterion

        return scores

decision_tree = DecisionTree(train_x, train_y, test_x, test_y)
criteria_to_try = ['mse', 'friedman_mse', 'mae']  

scores = decision_tree.train_decision_tree(criteria_to_try)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class RandomForest:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_criteria = None
        self.optimal_estimators = None

    def train_random_forest(self, criteria, estimators):
        scores = {}
        best_score = 0.0

        for criterion in criteria:
            for estimator in estimators:
                rf = RandomForestRegressor(n_estimators=estimator, criterion=criterion)
                rf.fit(self.__train_x, self.__train_y)
                predictions = rf.predict(self.test_x)
                score = r2_score(self.test_y, predictions)
                scores[(criterion, estimator)] = score

                if score > best_score:
                    best_score = score
                    self.optimal_criteria = criterion
                    self.optimal_estimators = estimator

        return scores

random_forest = RandomForest(train_x, train_y, test_x, test_y)
criteria_to_try = ['mse', 'mae']  
estimators_to_try = [50, 100, 150] 

scores = random_forest.train_random_forest(criteria_to_try, estimators_to_try)


# In[ ]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

class SupportVectorRegressor:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.trained_model = None

    def train_svr(self):
        svr = SVR(kernel='rbf')  # Use 'rbf' kernel for non-linear regression
        svr.fit(self.__train_x, self.__train_y)
        self.trained_model = svr

    def evaluate(self):
        predictions = self.trained_model.predict(self.test_x)
        mse = mean_squared_error(self.test_y, predictions)
        return mse

# Assuming train_x, train_y, test_x, test_y are already defined
svr = SupportVectorRegressor(train_x, train_y, test_x, test_y)
svr.train_svr()
mse_score = svr.evaluate()


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

class NeuralNetworkRegressor:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = None

    def build_model(self):
        model = Sequential()
        # Add layers (customize as needed)
        model.add(Dense(64, activation='relu', input_shape=(self.train_x.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Output layer for regression

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, epochs=50, batch_size=32):
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self):
        predictions = self.model.predict(self.test_x)
        r2 = r2_score(self.test_y, predictions)
        return r2

nn_regressor = NeuralNetworkRegressor(train_x, train_y, test_x, test_y)
nn_regressor.build_model()
nn_regressor.train_model(epochs=100, batch_size=32)
r2_score = nn_regressor.evaluate()

