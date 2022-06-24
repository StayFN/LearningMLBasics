import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
# New DF returns it and drops from dataframe at same time
# X Contains input (all rows except the one we want to predict
Y = np.array(data[predict])
# Y Contains prediction data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)  # Split into

""" #Training Model with different train test splits to get best one
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)  # Split into
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train) #Fit this data to find a best fit line and stores this line in linear
    acc = linear.score(x_test, y_test) #Accuracy of model based on test data
    print(acc)

    if acc > best:
        with open("studentmodel.pickle", "wb") as file: #Saves Model as pickle file
            pickle.dump(linear, file)
"""

pickle_in = open("studentmodel.pickle", "rb") # Loads Pickle file and assigns it to linear model
linear = pickle.load(pickle_in)
print("Coefficient:  \n", linear.coef_) # (line in 5 dimensional spaces (mx+b in multidemensional (m-coefficients))
# mx + zy + cy, ... m, z ,y, ... are those 5 values
# the bigger the coefficent the more weight each attribute has
print("Intercept:  \n", linear.intercept_) # Point at which axis meets 0

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "failures"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()