#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self, train_x, train_y, test_x, test_y) -> None:
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.__logical_regression_classifier = LogisticRegression(random_state=16)
        self.Fit(self.__logical_regression_classifier)
        prediction = self.Predict(self.__logical_regression_classifier)
        self.Plot_ConfusionMatrix(prediction, test_y)
    def Fit(self, estimator):
        estimator.fit(self.__train_x, self.__train_y)
    def Predict(self, estimator):
        return estimator.predict(self.test_x)
    def Score(self, data, true_labels):
        prediction = self.__logical_regression_classifier.predict(self.test_x)
        accuracy = self.__logical_regression_classifier.score(self.test_x, self.test_y)
        return accuracy
    def Plot_ConfusionMatrix(self, predictions, true_labels):
        cm = confusion_matrix(true_labels, prediction)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_k = None
        self.optimal_score = 0.0

    def train_knn(self, max_k):
        for k in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.__train_x, self.__train_y)
            score = knn.score(self.test_x, self.test_y)
            if score > self.optimal_score:
                self.optimal_score = score
                self.optimal_k = k

        return self.optimal_k, self.optimal_score


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_k = None
        self.optimal_score = 0.0

    def train_knn(self, max_k):
        for k in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.__train_x, self.__train_y)
            score = knn.score(self.test_x, self.test_y)
            if score > self.optimal_score:
                self.optimal_score = score
                self.optimal_k = k

        return self.optimal_k, self.optimal_score

knn_classifier = KNNClassifier(train_x, train_y, test_x, test_y)
optimal_k, accuracy = knn_classifier.train_knn(max_k=10)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_criterion = None
        self.optimal_score = 0.0

    def train_decision_tree(self, criteria):
        for criterion in criteria:
            decision_tree = DecisionTreeClassifier(criterion=criterion, random_state=42)
            decision_tree.fit(self.__train_x, self.__train_y)
            score = decision_tree.score(self.test_x, self.test_y)
            if score > self.optimal_score:
                self.optimal_score = score
                self.optimal_criterion = criterion

        return self.optimal_criterion, self.optimal_score

decision_tree_model = DecisionTree(train_x, train_y, test_x, test_y)
criteria_to_test = ['gini', 'entropy']
optimal_criterion, accuracy = decision_tree_model.train_decision_tree(criteria_to_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_criterion = None
        self.optimal_estimators = None
        self.optimal_score = 0.0

    def train_random_forest(self, criteria, n_estimators_list):
        for criterion in criteria:
            for n_estimators in n_estimators_list:
                rf_classifier = RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    random_state=42
                )
                rf_classifier.fit(self.__train_x, self.__train_y)
                score = rf_classifier.score(self.test_x, self.test_y)
                if score > self.optimal_score:
                    self.optimal_score = score
                    self.optimal_criterion = criterion
                    self.optimal_estimators = n_estimators

        return self.optimal_criterion, self.optimal_estimators, self.optimal_score

random_forest_model = RandomForest(train_x, train_y, test_x, test_y)
criteria_to_test = ['gini', 'entropy']
n_estimators_values = [100, 200, 300]  # Adjust the list of values as needed
optimal_criterion, optimal_estimators, accuracy = random_forest_model.train_random_forest(criteria_to_test, n_estimators_values)


# In[ ]:


from sklearn.svm import SVC

class SupportVectorClassifier:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.optimal_score = 0.0

    def train_svc(self):
        svc_classifier = SVC(kernel='rbf', random_state=42)  # Using RBF kernel as an example
        svc_classifier.fit(self.__train_x, self.__train_y)
        score = svc_classifier.score(self.test_x, self.test_y)
        self.optimal_score = score

        return self.optimal_score

svc_model = SupportVectorClassifier(train_x, train_y, test_x, test_y)
accuracy = svc_model.train_svc()


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

class ArtificialNeuralNetwork:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.__scaler = StandardScaler()
        self.__train_x = self.__scaler.fit_transform(train_x)
        self.__train_y = train_y
        self.__test_x = self.__scaler.transform(test_x)
        self.__test_y = test_y
        self.optimal_score = 0.0

    def train_ann(self, neurons, learning_rate, activation):
        model = Sequential()
        model.add(Dense(neurons, input_dim=self.__train_x.shape[1], activation=activation))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(self.__train_x, self.__train_y, epochs=10, batch_size=32, verbose=0)

        _, accuracy = model.evaluate(self.__test_x, self.__test_y)
        self.optimal_score = accuracy

        return self.optimal_score

# Assuming train_x, train_y, test_x, test_y are already defined
ann_model = ArtificialNeuralNetwork(train_x, train_y, test_x, test_y)
neurons = 64  # Number of neurons in the hidden layer
learning_rate = 0.001
activation = 'relu'  # Activation function for the hidden layer
accuracy = ann_model.train_ann(neurons, learning_rate, activation)

