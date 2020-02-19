import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import os
from utils import analyzeDataLength
from dictionary import load_dictionary, createTextdictionary, createKeyDict, createLocationDict
from preprocess import encodeData, encodeKeyAndLoc

tf.random.set_seed(42)
np.random.seed(42)

train_data = pd.read_csv(os.getcwd() + "/NLP with Disaster Tweets/data/train.csv",sep=",")
test_data = pd.read_csv(os.getcwd() + "/NLP with Disaster Tweets/data/test.csv",sep=",")

# Shuffle data
train_data = train_data.sample(len(train_data), random_state=42)

X_train = np.array(train_data.drop(['id', 'target'], axis='columns'))
Y_train = np.array(train_data['target'])
X_test = np.array(test_data.drop(['id'], axis='columns'))

num_words = 20095
num_key_words = 221
num_locations = 257

maxLen = 150

# createTextdictionary(X_train[:,2])
# createKeyDict(X_train[:,0])
# createLocationDict(X_train[:,1])

# analyzeDataLength(X_train[:,2])

textDict = load_dictionary('dictionary.pkl')
keyDict = load_dictionary('keyDict.pkl')
locDict = load_dictionary('locDict.pkl')

textInput = np.asarray(encodeData(X_train[:,2], textDict, maxLen), dtype=np.float32)
keyInput = encodeKeyAndLoc(X_train[:,0], keyDict)
locInput = encodeKeyAndLoc(X_train[:,1], locDict)

keyInput = np.expand_dims(np.array(keyInput), axis=1)
locInput = np.expand_dims(np.array(locInput), axis=1)
keyAndLocInput = np.asarray(np.concatenate((keyInput, locInput), axis=1), dtype=np.float32)

# print(textInput)
# print(keyAndLocInput)
# num_words + 1 because the 0 position is 'UNK' words

text_input_layer = Input(shape=(maxLen,), name = 'text_input')
embedding_layer_1 = Embedding(input_dim = num_words + 1, output_dim = 120, input_length = maxLen)(text_input_layer)
lstm_layer = LSTM(100)(embedding_layer_1)
lstm_layer = Dropout(0.4)(lstm_layer)

other_input_layer = Input(shape=(2,), name = 'other_input')
embedding_layer_2 = Embedding(input_dim = num_key_words + num_locations + 2, output_dim = 40, input_length = 2)(other_input_layer)
flatten_embedding = Flatten()(embedding_layer_2)
concatenated_layer = keras.layers.concatenate([lstm_layer, flatten_embedding])

next_layer = Dense(64, activation='relu')(concatenated_layer)
next_layer = Dense(64, activation='relu')(next_layer)

output_layer = Dense(1, activation='sigmoid', name='output')(next_layer)

model = Model(inputs=[text_input_layer, other_input_layer], outputs=[output_layer])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["acc"])
model_checkpoint = keras.callbacks.ModelCheckpoint(
    os.getcwd() + "/NLP with Disaster Tweets/model.h5", 
    save_best_only=True,
    monitor="val_acc",
    mode="max")
# model.summary()
model.fit([textInput, keyAndLocInput], [Y_train],
         epochs = 120, 
         batch_size = 1000, 
         shuffle = True, 
         validation_split=0.15,
         callbacks=[model_checkpoint])

# model.save(os.getcwd() + "/NLP with Disaster Tweets/model.h5")