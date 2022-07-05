import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

word_index = data.get_word_index() # Only gives tuples

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<Start>"] = 1
word_index["<UMK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value,key) for (key, value) in word_index.items()]) #Reverses Dic key values

print(len(test_data[0]), len(test_data[1])) # Doesnt work for our Model, need to be same length for Input Neurons
# So lets use Padding Tag to set definiete length for Dataset, we could use the longest Review
# But we can also set an arbitrary number. Lets say 250

#Pad data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

print(len(test_data[0]), len(test_data[1])) # Doesnt work for our Model, need to be same length for Input Neurons


def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text]) #Return all of keys we want (human readable words)





