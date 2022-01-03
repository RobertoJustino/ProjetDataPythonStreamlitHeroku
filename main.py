from ml.utils import utils
from ml.utils.utils import DataHandler
from ml.utils.utils import FeatureRecipe
from ml.utils.utils import FeatureExtractor
import pandas as pd
import numpy as np
import streamlit as st
from nltk.corpus import stopwords

'''
def DataManager(d:DataHandler=None, fr: FeatureRecipe=None, fe:FeatureExtractor=None):
    """
        Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)
    """

    pass
#on appelera la fonction DataManager() de la facon suivante :
X_train, X_test, y_train, y_test = DataManager()
'''

df = DataHandler()
df.setData(pd.read_csv('labels.csv'))
df = df.getData()
df.info()

fr = FeatureRecipe(df)
fr.tweetClean()


number = st.number_input("Number of Rows to View")
st.dataframe(df.head(int(number)))

fr = FeatureRecipe(df)
fr.setCategorical(df['tweetClean'])
fr = fr.getCategorical()

X = fr
X
y = df['class']
y


fe = FeatureExtractor(df, X)
