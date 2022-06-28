import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()

data = scale(digits.data) # Scales digits down into a range between -1 and 1

y = digits.target

k = 10
samples, features = data.shape


def bench_k_means(estimator, name, data):
    pass


