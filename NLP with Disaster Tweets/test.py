# from tensorflow import keras
# from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten

# maxlen = 250
# num_word = 80000
# num_word_keyword_location = 300

# text_input = Input(shape=(maxlen,), name = 'text_input')
# embedding_layer_1 = Embedding(input_dim = num_word, output_dim = 80, input_length = maxlen)(text_input)
# lstm_layer = LSTM(60)(embedding_layer_1)

# other_input = Input(shape=(2,), name = 'other_input')
# embedding_layer_2 = Embedding(input_dim = num_word_keyword_location, output_dim = 10, input_length = 2)(other_input)
# flatten_layer = Flatten()(embedding_layer_2)

# c = keras.layers.concatenate([lstm_layer, flatten_layer])
# print(c.shape)
import pandas as pd
import numpy as np
df = pd.DataFrame(np.array([[1,2,3],[3,4,5]]), columns=['id','target'])