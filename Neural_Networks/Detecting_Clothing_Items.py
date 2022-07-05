import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


"""plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

print(train_images[7])
 """
# np_array of Black/White Values from 0-255

train_images = train_images/255.0 # To get Values between 0 and 1 we divide by max val of 255
test_images = test_images/255.0
# Now we need to flatten Data to input in NN [28x28] --> [784] ([[x,y,z],[x,...],..] --> [x,y,z,x,y,z,...]

model = keras.Sequential([ # Sequentiel == defining them in order
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax") # Softma will pick values for each neuron, so they add up to 1. So its probability basically
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)# epochs essentially means how many times the model is gonna see this information
#It randomply pikcs images and labels correspondoing to each other and feed that trough the NN
#So how many epochs you decide is how many times you see the same image. Bc order images come in will affect how how
#Parameters and things in network are tweaked, maybe seeing 10 images that are pants if it sees a few pants and few shirt
# and few sandals. essentially it gives same images in a different order. Hopefully increases accuracy
# However not always does that, have to play with it.

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"tested Acc: {test_acc}")

# Use the model

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

print(class_names[np.argmax(prediction[0])])




