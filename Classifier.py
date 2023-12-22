#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Classifier:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        """
        Initialize the Classifier with training and testing data.
        
        Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        """
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
        self.__logistic_regression_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
        self.fit(self.__logistic_regression_classifier)
        prediction = self.predict(self.__logistic_regression_classifier)
        self.plot_confusionMatrix(prediction, self.__y_test)
        
    def fit(self, estimator):
        """
        Fit the given estimator on the training data.
        
        Parameters:
        estimator: The model to be trained.
        """
        estimator.fit(self.__X_train, self.__y_train)
        
    def predict(self, estimator):
        """
        Make predictions using the given estimator on the test data.
        
        Parameters:
        estimator: The trained model for prediction.
        
        Returns:
        Predicted labels.
        """
        return estimator.predict(self.__X_test)
    
    def score(self):
        """
        Calculate and return the accuracy score of the classifier.
        
        Returns:
        Accuracy score.
        """
        accuracy = self.__logistic_regression_classifier.score(self.__X_test, self.__y_test)
        return accuracy
    
    def plot_confusionMatrix(self, predictions, true_labels):
        """
        Plot the confusion matrix for predicted and true labels.
        
        Parameters:
        predictions: Predicted labels.
        true_labels: True labels.
        """
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


# In[6]:


from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        """
        Initialize the KNNClassifier with training and testing data.
        
        Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        """
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
        self.optimal_k = None
        self.optimal_score = 0.0

    def train_knn(self, max_k):
        """
        Train the KNN model for different values of k and find the optimal k.
        
        Parameters:
        max_k (int): Maximum value of k to consider.
        
        Returns:
        tuple: Optimal k value, Optimal score achieved.
        """
        for k in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.__X_train, self.__y_train)
            score = knn.score(self.__X_test, self.__y_test)
            if score > self.optimal_score:
                self.optimal_score = score
                self.optimal_k = k

        return self.optimal_k, self.optimal_score

    def get_optimal_knn(self):
        """
        Get the optimal KNN model using the previously found optimal k.
        
        Returns:
        KNeighborsClassifier: Optimal KNN model.
        """
        return KNeighborsClassifier(n_neighbors=self.optimal_k)


# In[7]:


from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        """
        Initialize the DecisionTree classifier with training and testing data.
        
        Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        """
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
        self.optimal_criterion = None
        self.optimal_score = 0.0

    def train_decision_tree(self, criteria):
        """
        Train the DecisionTree model for different criteria and find the optimal criterion.
        
        Parameters:
        criteria (list): List of criteria to consider.
        
        Returns:
        tuple: Optimal criterion, Optimal score achieved.
        """
        for criterion in criteria:
            decision_tree = DecisionTreeClassifier(criterion=criterion, random_state=42)
            decision_tree.fit(self.__X_train, self.__y_train)
            score = decision_tree.score(self.__X_test, self.__y_test)
            if score > self.optimal_score:
                self.optimal_score = score
                self.optimal_criterion = criterion

        return self.optimal_criterion, self.optimal_score


# In[8]:


from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        """
        Initialize the RandomForest classifier with training and testing data.
        
        Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        """
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
        self.optimal_criterion = None
        self.optimal_estimators = None
        self.optimal_score = 0.0

    def train_random_forest(self, criteria, n_estimators_list):
        """
        Train the RandomForest model for different criteria and number of estimators, and find the optimal parameters.
        
        Parameters:
        criteria (list): List of criteria to consider.
        n_estimators_list (list): List of number of estimators to consider.
        
        Returns:
        tuple: Optimal criterion, Optimal number of estimators, Optimal score achieved.
        """
        for criterion in criteria:
            for n_estimators in n_estimators_list:
                rf_classifier = RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    random_state=42
                )
                rf_classifier.fit(self.__X_train, self.__y_train)
                score = rf_classifier.score(self.__X_test, self.__y_test)
                if score > self.optimal_score:
                    self.optimal_score = score
                    self.optimal_criterion = criterion
                    self.optimal_estimators = n_estimators

        return self.optimal_criterion, self.optimal_estimators, self.optimal_score


# In[9]:


from sklearn.svm import SVC

class SupportVectorClassifier:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        """
        Initialize the SupportVectorClassifier with training and testing data.
        
        Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        """
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_test = X_test
        self.__y_test = y_test
        self.optimal_score = 0.0

    def train_svc(self):
        """
        Train the Support Vector Classifier and calculate the accuracy score and confusion matrix.
        
        Returns:
        tuple: Optimal score achieved, Confusion matrix.
        """
        svc_classifier = SVC(kernel='rbf', random_state=42)
        svc_classifier.fit(self.__X_train, self.__y_train)
        score = svc_classifier.score(self.__X_test, self.__y_test)
        self.optimal_score = score
        predictions = svc_classifier.predict(self.__X_test)
        self.conf_matrix = confusion_matrix(self.__y_test, predictions)

        return self.optimal_score, self.conf_matrix


# In[10]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class ArtificialNeuralNetwork:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        """
        Initialize the Artificial Neural Network with training and testing data.
        
        Parameters:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        """
        self.__scaler = StandardScaler()
        self.__X_train = self.__scaler.fit_transform(X_train)
        self.__y_train = y_train
        self.__X_test = self.__scaler.transform(X_test)
        self.__y_test = y_test
        self.optimal_score = 0.0
        self.conf_matrix = None

    def train_ann(self):
        """
        Train the Artificial Neural Network and compute the confusion matrix.
        
        Returns:
        array-like: Confusion matrix.
        """
        model = Sequential()
        input_layer = Dense(units=6, activation='relu', kernel_initializer='uniform')
        model.add(input_layer)
        hidden_layer = Dense(units=6, activation='relu', kernel_initializer='uniform')
        model.add(hidden_layer)
        output_layer = Dense(units=1, activation='sigmoid', kernel_initializer='uniform')
        model.add(output_layer)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.fit(self.__X_train, self.__y_train, batch_size=10, epochs=10)
        y_pred_prob = model.predict(self.__X_test)

        # Convert probabilities to class labels based on a threshold (0.5 for binary classification)
        y_pred = (y_pred_prob > 0.5).astype(int)

        self.conf_matrix = confusion_matrix(self.__y_test, y_pred)

        return self.conf_matrix

