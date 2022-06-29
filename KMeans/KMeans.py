import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()

data = scale(digits.data)  # Scales digits down into a range between -1 and 1

y = digits.target

k = 10  # Fancypants way: len(np.unique(y))
samples, features = data.shape  # (1000,728) f.e.

# How does it Work? Assign K random points (centroids) on graph. Points closest to K are mapped to Point Kx.
# Then take Average of coordinates of those points x1+x2+x3+..+xn/n and y1+y2+y3+y4..yn/n to get average
# This gives us next coordinate for the centroid
# Draw straight line trough midpoint of centroids and reassign points.
# Repeat process (start with average) until we get no changes between data points. -> Points are clustered in K clusters

# Scoring functions form Sklearn


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


clf = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300)
bench_k_means(clf, "1", data)


