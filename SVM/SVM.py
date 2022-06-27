import sklearn
from sklearn import svm
from sklearn import datasets

cancer = datasets.load_breast_cancer()

print(f"Features: {cancer.feature_names}")
print(f"Labels: {cancer.target_names}")

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

print(x_train, y_train)

