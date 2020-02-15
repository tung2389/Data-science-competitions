# import tensorflow as tf
# from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
import os
from preprocess import preprocess
import json

train_data = pd.read_csv(os.getcwd() + "/NLP with Disaster Tweets/data/train.csv",sep=",")
test_data = pd.read_csv(os.getcwd() + "/NLP with Disaster Tweets/data/test.csv",sep=",")

X_train = np.array(train_data.drop(['id', 'target'], axis='columns'))
Y_train = np.array(train_data['target'])
X_test = np.array(train_data.drop(['id', 'target'], axis='columns'))

dictionary = {'<PAD>': 0, 'UNK': 1}

def create_and_save_dictionary():
    i = 2
    numwords = 0
    for data in X_train:
        word_array = preprocess(data[2])
        for word in word_array:
            word = word.lower()
            if not word in dictionary:
                dictionary[word] = i
                i = i + 1
                numwords = numwords + 1
    save_dictionary(dictionary)

def save_dictionary(dict):
    with open(os.getcwd() + '/NLP with Disaster Tweets/dictionary.pkl', 'wb') as fileData:
        pickle.dump(dict, fileData, pickle.HIGHEST_PROTOCOL)
        
def load_dictionary():
    with open(os.getcwd() + '/NLP with Disaster Tweets/dictionary.pkl', 'rb') as fileData:
        return pickle.load(fileData)

def saveDictAsJson(dict):
    with open(os.getcwd() + '/NLP with Disaster Tweets/dictionary.json', 'w') as fileData:
        return json.dump(dict, fileData)
        
create_and_save_dictionary()
# print(load_dictionary())

