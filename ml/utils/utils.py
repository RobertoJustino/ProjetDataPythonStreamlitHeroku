import pandas as pd
import numpy as np
import streamlit as st
from nltk.corpus import stopwords


class DataHandler:
    """
        Get data from sources
    """
    def __init__(self):
        self.csvfile1 = None
        self.csvfile2 = None
        self.gouped_data = None

    def getData(self):
            return self.csvfile1

    def setData(self, data: pd.DataFrame):
            self.csvfile1 = data

df = DataHandler()
df.setData(pd.read_csv('labels.csv'))
df = df.getData()
df.info()


class FeatureRecipe:
    """
    Feature processing class
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.continuous = None
        self.categorical = None
        self.discrete = None
        self.datetime = None

    def setContinuous(self, data: pd.DataFrame):
        self.continuous = data

    def setCategorical(self, data: pd.DataFrame):
        self.categorical = data

    def setDiscrete(self, data: pd.DataFrame):
        self.discrete = data

    def getCategorical(self):
        return self.categorical

    def tweetClean(self):

        tweetClean = self.data['tweet'].str.replace('\W', ' ')
        stop = stopwords.words('english')
        pat = r'\b(?:{})\b'.format('|'.join(stop))
        self.data['tweetClean'] = tweetClean.str.replace(pat, '')
        tweetClean = self.data['tweetClean']

        return tweetClean






class FeatureExtractor:
    """
    Feature Extractor class
    """
    def __init__(self, data: pd.DataFrame, flist: list):
        """
            Input : pandas.DataFrame, feature list to drop
            Output : X_train, X_test, y_train, y_test according to 		    sklearn.model_selection.train_test_split
        """
        self.data = data
        self.X = flist
        self.y = None

    def getData(self):
    	return self.data

    def getX(self):
    	return self.X

    def getY(self):
    	return self.y

    def setX(self, X: list):
    	self.X =  X

    def setY(self, y: list):
    	self.y = y





class ModelBuilder:
    """
        Class for train and print results of ml model
    """
    def __init__(self, model_path: str = None, save: bool = None):
        pass
    def __repr__(self):
        pass
    def train(self, X, Y):
        pass
    def predict_test(self, X) -> np.ndarray:
        pass
    def predict_from_dump(self, X) -> np.ndarray:
        pass
    def save_model(self, path:str):
        #with the format : ‘model_{}_{}’.format(date)
        pass
    def print_accuracy(self):
        pass
    def load_model(self):
        try:
            #load model
            pass
        except:
            pass
