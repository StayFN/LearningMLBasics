import tensorflow as tf
import tensorflow.keras as keras #Fixes code suggestions
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

# print(len(test_data[0]), len(test_data[1])) # Doesnt work for our Model, need to be same length for Input Neurons
# So lets use Padding Tag to set definiete length for Dataset, we could use the longest Review
# But we can also set an arbitrary number. Lets say 250

#Pad data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

print(len(test_data[0]), len(test_data[1])) # Doesnt work for our Model, need to be same length for Input Neurons


def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text]) #Return all of keys we want (human readable words)

#Model

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
# Embedding layer tries to group words in a similair way (finds word vectors) 16 dimensional here (10000 Word vectors)
model.add(keras.layers.GlobalAveragePooling1D())
# Takes whatever dimension our Data is in and puts it in another Dimension
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
# We want final output whether Review is good or review is bad so 1 output Neuron either 0 or 1 or somewhere between
# To give a probability between that (we can accomplish that using a sigmoid activation function)
model.summary() # Prints summary of model

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#Binary Crossentropy two options for output neuron 1/0 sigmoid function means between 0 and 1 loss funciton
# Will calculate how much of a diference 0.2 is from 0 for example.

x_val = train_data[:10000] #Gonna use val to validate models and tweak params
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
#BatchSize = how many review we load in at once, bc we load it all into memory

results = model.evaluate(test_data, test_labels)

print(results)

