import tensorflow as tf
import tensorflow.keras as keras #Fixes code suggestions
import numpy as np


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

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

# print(len(test_data[0]), len(test_data[1])) # Doesnt work for our Model, need to be same length for Input Neurons


def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text]) #Return all of keys we want (human readable words)

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded
#Model

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
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

#Saving Model

model.save("model.h5") # h5 extension for saved model in keras/tensorflow, saves model in binary data

#Load model
model = keras.models.load_model("model.h5")
"""Realworld_Review_IMDB_In_Time_10of10Stars.txt"""
#Handles Lion King Review well, but more complicated like the IMDB Review of Just-in-Time not so well.
#Probably because embeddings are too simple. Maybe with more sophisticated like Bert Embeddings it will be better
#Test model on real world data
with open("test.txt", encoding="utf-8") as file:
    for line in file.readlines():
        #This should work better with stopword library and nltk or smth. but easy way in tutorial w/o library so i do it too
        #If we Embed Words we only want words, bc we split by spaces. So we get "Word" and not "Word." or "Word,"
        nline = line.replace(",", "").replace(".", "").replace("(","").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                                maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

