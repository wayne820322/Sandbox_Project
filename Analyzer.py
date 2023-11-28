#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Analyzer:
    def __init__(self) -> None:
        self.__data = None
    def read_dataset(self, fileName):
        self.__data  = pd.read_csv(fileName)
    def describe(self):
        print('numeric data description:')
        print(self.__data.describe())
        print('non numeric data description:')
        print(self.__data.describe(include='object'))
    def drop_missing_data(self):
        self.__data = self.__data.dropna()
    def plot_histograms_numeric(self, columnName):
        self.__data.hist(column=columnName, bins=50)
        plt.show()
    def retrieve_data(self):
        return self.__data
    def drop_columns(self, attribute_list):
        self.__data.drop(columns=attribute_list, inplace=True)
    def encode_features(self, columns_list):
        # Assuming categorical attributes need to be encoded using one-hot encoding
        self.__data = pd.get_dummies(self.__data, columns=columns_list)
    def encode_label(self, label):
        self.__data = pd.get_dummies(self.__data, columns=label)
    def shuffle(self):
        self.__data = self.__data.sample(frac=1).reset_index(drop=True)
    def sample(self, reduction_factor):
        self.__data = self.__data.sample(frac=reduction_factor)
    def plot_correlationMatrix(self):
        correlation_matrix = self.__data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
    def plot_pairPlot(self):
        categorical_columns = self.__data.select_dtypes(include='object').columns
        for col in categorical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.__data[col], kde=False)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
    def plot_histograms_categorical(self):
        categorical_columns = self.__data.select_dtypes(include='object').columns
        for col in categorical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.__data[col], kde=False)
            plt.show()
    def plot_boxPlot(self):
        numeric_columns = self.__data.select_dtypes(include='number').columns
        for col in numeric_columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=self.__data[col])
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.show()

