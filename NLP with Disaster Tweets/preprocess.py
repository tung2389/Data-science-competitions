import numpy as np
from tensorflow import keras

specialCharacters = [",", ".", "(", ")", ":", '"', ";", "!","?","@","/","\\"]
count = 0
def preprocess(text):
    text = deleteUrls(text) # Based on the training data, I think that the urls are irrelevant and they should have no impact on the prediction of the model
    text = text.lower()
    for char in specialCharacters:
        text = text.replace(char, "") # Eliminate special characters
    text = text.strip().split(" ") # Split words into array
    return text

def deleteUrls(text):
    global count
    if(haveUrl(text)):
        urlsArray = findUrls(text)
        count = count + len(urlsArray)
        for url in urlsArray:
            text = text.replace(url, "")
    return text

def haveUrl(text):
    if text.find('http://') != -1 or text.find('https://') != -1:
        return True
    else:
        return False

def findUrls(text):
    urls = []
    start1 = 0 # Position start searching for 'http'
    start2 = 0 # Position start searching for 'https'
    while True:
        http = True # If find 'http' string, htpt = True
        pos = text.find('http://', start1)
        if(pos == -1):
            pos = text.find('https://', start2)
            http = False 
        if(pos != -1):
            temp = "" # temp holding value of an url
            for char in range(pos, len(text)):
                if text[char] == " " or text[char] == '\n':
                    break
                temp = temp + text[char]
            urls.append(temp)
            if(http):
                start1 = pos + len(temp)
            else:
                start2 = pos + len(temp)
        else:
            break
    return urls

def encodeData(data, dictionary, maxLen):
    for i in range(0, len(data)):
        data[i] = preprocess(data[i])

    for i in range(0, len(data)):
        text = data[i]
        for j in range(0, len(text)):
            word = text[j]
            if word in dictionary:
                data[i][j] = dictionary[word]
            else:
                data[i][j] = dictionary['UNK']
    data = keras.preprocessing.sequence.pad_sequences(data, 
                                                    value=dictionary["<PAD>"], 
                                                    padding="post", 
                                                    maxlen=maxLen)
    return data

def encodeKeyAndLoc(data, dictionary):
    for i in range(0, len(data)):
        element = data[i]
        if element in dictionary:
            data[i] = dictionary[element]
        else:
            data[i] = dictionary['UNK']
    return data