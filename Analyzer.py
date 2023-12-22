#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

class Analyzer:
    def __init__(self):
        self.__data = None
        
    def read_dataset(self, fileName):
        """
        Read dataset from a CSV file.
        
        Parameters:
        fileName (str): Name of the CSV file.
        """
        self.__data  = pd.read_csv(fileName)    
    
    def describe(self):
        """
        Display descriptive statistics of the dataset.
        """
        self.__data.describe()
        
    def drop_missing_data(self):
        """
        Remove rows with missing data.
        """
        self.__data = self.__data.dropna()
        
    def retrieve_data(self):
        """
        Retrieve the dataset.
        
        Returns:
        DataFrame: The dataset.
        """
        return self.__data 
    
    def drop_columns(self, attribute_list):
        """
        Remove columns in the dataset.
        """
        self.__data = self.__data.drop(columns=attribute_list, axis=1)
        
    def encode_features(self, features_list):
        """
        Encode specified features using OrdinalEncoder.

        Parameters:
        features_list (list): List of feature columns to encode.
        """
        enc = OrdinalEncoder()
        df_ordinal = self.__data.copy()
        df_ordinal[features_list] = enc.fit_transform(self.__data[features_list])
        self.__data = df_ordinal
        
    def encode_label(self, label):
        """
        Encode a categorical label column into dummy variables using one-hot encoding.

        Parameters:
        label (str): Name of the categorical label column.
        """
        self.__data = pd.get_dummies(self.__data, columns = label)
        
    def shuffle(self):
        """
        Shuffle the rows of the dataset.
        """
        self.__data = self.__data.sample(frac=1).reset_index(drop=True)
        
    def sample(self, reduction_factor):
        """
        Randomly sample a fraction of the dataset.

        Parameters:
        reduction_factor (float): Fraction of data to retain (0-1).
        """
        self.__data = self.__data.sample(frac=reduction_factor)
        
    def plot_correlationMatrix(self):
        """
        Plot the correlation matrix heatmap.
        """
        correlation_matrix = self.__data.corr()
        plt.figure(figsize=(14, 8))
        sns.heatmap(correlation_matrix, annot=True)
        plt.show()
        
    def plot_pairPlot(self):
        """
        Plot pairplots for numerical columns.
        """
        numerical_columns = self.__data.select_dtypes(include='number').columns
        sns.pairplot(self.__data[numerical_columns])
        plt.show()
            
    def plot_histograms_numeric(self, columnName):
        """
        Plot histograms for a specific numeric column.
        
        Parameters:
        columnName (str): Name of the numeric column.
        """
        self.__data.hist(column = columnName, bins = 4)
        plt.show()
        
    def plot_histograms_categorical(self):
        """
        Plot histograms for categorical columns.
        """
        categorical_columns = self.__data.select_dtypes(include='number').columns
        for col in categorical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.__data[col], kde=False)
            plt.show()
            
    def plot_boxPlot(self):
        """
        Plot boxplots for numerical columns.
        """
        numeric_columns = self.__data.select_dtypes(include='number').columns
        for col in numeric_columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=self.__data[col])
            plt.xlabel(col)
            plt.show()

