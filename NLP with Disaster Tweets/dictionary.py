import pickle
import os
import pandas as pd
from preprocess import preprocess

dictionary = {'<PAD>': 0, 'UNK': 1}
keywordDict = {'UNK':0}
locationDict = {'UNK':0}

num_words = 10000

def createTextdictionary(X_train):
    i = 2
    count = {}
    for data in X_train:
        word_array = preprocess(data)
        for word in word_array:
                if not word in count:
                    count[word] = 1
                else:
                    count[word] = count[word] + 1
    count = sorted(count.items(), key=lambda item: item[1], reverse=True)
    for word,value in count:
        if not word in dictionary:
            dictionary[word] = i
            i = i + 1
        if i >= num_words + 2: # Only take top 15000 words with most occurrence.
            break
    save_dictionary(dictionary, 'dictionary.pkl')

def save_dictionary(dict, fileName):
    with open(os.getcwd() + '/NLP with Disaster Tweets/dictionary/' + fileName, 'wb') as fileData:
        pickle.dump(dict, fileData, pickle.HIGHEST_PROTOCOL)

def saveDictAsJson(dict):
    with open(os.getcwd() + '/NLP with Disaster Tweets/dictionary/dictionary.json', 'w') as fileData:
        return json.dump(dict, fileData)

def load_dictionary(fileName):
    with open(os.getcwd() + '/NLP with Disaster Tweets/dictionary/' + fileName, 'rb') as fileData:
        return pickle.load(fileData)

def createKeyDict(data):
    i = 1
    for keyword in data:
        if pd.notnull(keyword):
            if not keyword in keywordDict:
                keywordDict[keyword] = i
                i = i + 1
    save_dictionary(keywordDict, 'keyDict.pkl')

def createLocationDict(data):
    i = 1
    count = {}
    for location in data:
        if pd.notnull(location):
            if not location in count:
                count[location] = 1
            else:
                count[location] = count[location] + 1
    count = sorted(count.items(), key=lambda item: item[1], reverse=True)
    for location,value in count:
        if not location in locationDict and value >= 3: # Only save location whose number of occurrence is greater or equal to 3
            locationDict[location] = i
            i = i + 1
        elif value < 3:
            break
    save_dictionary(locationDict, 'locDict.pkl')

    
