import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    X_len = len(series) - window_size
    for i in range(X_len):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    n_units = 5
    
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(window_size, 1), activation='tanh'))
    model.add(Dense(1, activation='linear'))
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import string
    punctuation = ['!', ',', '.', ':', ';', '?']
    letters = string.ascii_lowercase
    chars = set(text)
    atypical_chars = [c for c in chars if c not in punctuation and c not in letters]
    print("atypical characters: ", atypical_chars)
    for c in atypical_chars:
        text = text.replace(c, ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    inputs_len = len(text) - window_size
    for i in range(0, inputs_len, step_size):
        inputs.append(text[i: i + window_size])
        outputs.append(text[i + window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    n_units = 200

    model = Sequential()
    model.add(LSTM(n_units, input_shape=(window_size, num_chars), activation='tanh'))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
              
