from tensorflow import keras
import pandas as pd
import numpy as np
import os
from dictionary import load_dictionary
from preprocess import encodeData, encodeKeyAndLoc

maxLen = 150

test_data = pd.read_csv(os.getcwd() + "/NLP with Disaster Tweets/data/test.csv",sep=",")
ids = np.array(test_data['id'])
ids = ids.reshape(len(ids),1)
X_test = np.array(test_data)

textDict = load_dictionary('dictionary.pkl')
keyDict = load_dictionary('keyDict.pkl')
locDict = load_dictionary('locDict.pkl')

textInput = np.asarray(encodeData(X_test[:,3], textDict, maxLen), dtype=np.float32)
keyInput = encodeKeyAndLoc(X_test[:,1], keyDict)
locInput = encodeKeyAndLoc(X_test[:,2], locDict)

keyInput = np.expand_dims(np.array(keyInput), axis=1)
locInput = np.expand_dims(np.array(locInput), axis=1)
keyAndLocInput = np.asarray(np.concatenate((keyInput, locInput), axis=1), dtype=np.float32)

model = keras.models.load_model(os.getcwd() + '/NLP with Disaster Tweets/model.h5')

predictions = model.predict([textInput, keyAndLocInput])

for i in range(0, len(predictions)):
    prediction = predictions[i]
    if(prediction < 0.5):
        predictions[i] = 0
    else:
        predictions[i] = 1

predictions = predictions.astype(int)

df = pd.DataFrame(np.concatenate((ids, predictions), axis=1), columns=['id', 'target'])
df.to_csv(os.getcwd() + '/NLP with Disaster Tweets/submission/submission.csv', index=False)
