#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Regressor:
    def __init__(self, train_x, train_y, test_x, test_y) -> None:
        """
        Initialize the Regressor with training and testing data.
        
        Parameters:
        train_x: Training features.
        train_y: Training labels.
        test_x: Test features.
        test_y: Test labels.
        """
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.__linear_regression_classifier = LinearRegression()
        self.fit(self.__linear_regression_classifier)
        prediction = self.predict(self.test_x)
        linear_regression_score = self.score(prediction, self.test_y, r2=True)
        print(linear_regression_score)
        
    def fit(self, estimator):
        """
        Fit the given estimator on the training data.
        
        Parameters:
        estimator: The regression model to be trained.
        """
        estimator.fit(self.__train_x, self.__train_y)
        
    def predict(self, test_data):
        """
        Make predictions using the trained regression model.
        
        Parameters:
        test_data: Data to predict on.
        
        Returns:
        array-like: Predicted values.
        """
        return self.__linear_regression_classifier.predict(test_data)
        
    def score(self, data, true_labels, r2=False, mse=False, mae=False, rmse=False):
        """
        Calculate regression evaluation metrics.
        
        Parameters:
        data: Predicted values.
        true_labels: True labels.
        r2: Whether to calculate R-squared score.
        mse: Whether to calculate Mean Squared Error.
        mae: Whether to calculate Mean Absolute Error.
        rmse: Whether to calculate Root Mean Squared Error.
        
        Returns:
        dict: Dictionary containing requested scores.
        """
        scores = {}

        if r2:
            scores['R-squared'] = r2_score(true_labels, data)
        if mse:
            scores['Mean Squared Error'] = mean_squared_error(true_labels, data)
        if mae:
            scores['Mean Absolute Error'] = mean_absolute_error(true_labels, data)
        if rmse:
            mse_value = mean_squared_error(true_labels, data)
            scores['Root Mean Squared Error'] = (mse_value ** 0.5)

        return scores


# In[3]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class KNNRegressor:
    def __init__(self, train_x, train_y, test_x, test_y):
        """
        Initialize the KNNRegressor with training and testing data.
        
        Parameters:
        train_x: Training features.
        train_y: Training labels.
        test_x: Test features.
        test_y: Test labels.
        """
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_k = None

    def train_knn(self, max_k):
        """
        Train the KNN regressor for different values of k and calculate evaluation metrics.
        
        Parameters:
        max_k (int): Maximum value of k to consider.
        
        Returns:
        dict: Dictionary containing R-squared, Mean Squared Error, and Mean Absolute Error for different k values.
        """
        r2_scores = []
        mse_scores = []
        mae_scores = []
        best_r2_score = 0.0

        for k in range(1, max_k + 1):
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.__train_x, self.__train_y)
            predictions = knn.predict(self.test_x)

            r2 = r2_score(self.test_y, predictions)
            mse = mean_squared_error(self.test_y, predictions)
            mae = mean_absolute_error(self.test_y, predictions)

            r2_scores.append(r2)
            mse_scores.append(mse)
            mae_scores.append(mae)

            if r2 > best_r2_score:
                best_r2_score = r2
                self.optimal_k = k

        return {
            'R-squared': r2_scores,
            'Mean Squared Error': mse_scores,
            'Mean Absolute Error': mae_scores
        }


# In[6]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class DecisionTree:
    def __init__(self, train_x, train_y, test_x, test_y):
        """
        Initialize the DecisionTree with training and testing data.
        
        Parameters:
        train_x: Training features.
        train_y: Training labels.
        test_x: Test features.
        test_y: Test labels.
        """
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_criterion = None

    def train_decision_tree(self, criteria):
        """
        Train the DecisionTree regressor for different criteria and calculate R-squared scores.
        
        Parameters:
        criteria (list): List of criteria to consider.
        
        Returns:
        dict: Dictionary containing R-squared scores for different criteria.
        """
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


# In[4]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SupportVectorRegressor:
    
    def __init__(self, train_x, train_y, test_x, test_y):
        """
        Initialize the SupportVectorRegressor with training and testing data.
        
        Parameters:
        train_x: Training features.
        train_y: Training labels.
        test_x: Test features.
        test_y: Test labels.
        """
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.trained_model = None

    def train_svr(self):
        """
        Train the Support Vector Regressor.
        """
        svr = SVR(kernel='rbf')
        svr.fit(self.__train_x, self.__train_y)
        self.trained_model = svr

    def evaluate(self):
        """
        Evaluate the trained Support Vector Regressor on the test data.
        
        Returns:
        dict: Dictionary containing Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared score.
        """
        predictions = self.trained_model.predict(self.test_x)
        mse = mean_squared_error(self.test_y, predictions)
        mae = mean_absolute_error(self.test_y, predictions)
        r2 = r2_score(self.test_y, predictions)
        return {'MSE': mse, 'MAE': mae, 'R-squared': r2}


# In[5]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

class NeuralNetworkRegressor:
    def __init__(self, train_x, train_y, test_x, test_y):
        """
        Initialize the NeuralNetworkRegressor with training and testing data.
        
        Parameters:
        train_x: Training features.
        train_y: Training labels.
        test_x: Test features.
        test_y: Test labels.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = None

    def build_model(self):
        """
        Build the neural network model.
        """
        model = Sequential()
        # Add layers (customize as needed)
        model.add(Dense(64, activation='relu', input_shape=(self.train_x.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Output layer for regression

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, epochs=50, batch_size=32):
        """
        Train the neural network model.
        
        Parameters:
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        """
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=0)

    def evaluate(self):
        """
        Evaluate the trained neural network model on the test data.
        
        Returns:
        float: R-squared score.
        """
        predictions = self.model.predict(self.test_x)
        r2 = r2_score(self.test_y, predictions)
        return r2

