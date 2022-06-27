import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(f"Features: {cancer.feature_names}")
#print(f"Labels: {cancer.target_names}")

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

#print(x_train, y_train)
classes = ["malignant", "benign"]

# SVM Trying to classify data by spltting it with a line (in 2d) if that's not possible we can add a dimension with a
# Kernel function f(X1,X2) -> X3 that returns a 3rd dimension coordinate. Thus we could divide it via a Plane
# If that doesnt work we can also do even higher dimensions. You typically don't create a Kernel, you use an existing
# One. F.e. X1^2+X2^2 would be a simple kernel function. That would result in X3. f(1,2) = 1^2+2^2 = 5. So (1,2,5)
# Super simpflified high-level explanation, normally more complicated.

# Soft margin: Allow for a few points to exist within the margin on both sides of splitting hyperplane in order to
# get a better classifier.

"""
kernels = ["linear", "poly", "rbf", "sigmoid"] # Trying out different Kernels
for i in range(len(kernels)):
    clf = svm.SVC(kernel=kernels[i])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy {kernels[i]}-kernel: {acc}")
"""

clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(f"Linear SVM C=1 Acc: {acc}")

clf = svm.SVC(kernel="linear", C=2) #Trying different soft margin (default is also 1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(f"Linear SVM C=2 Acc:{acc}")

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(f"KNN n=9 Acc:{acc}")

#Typically KNN doesn't work that well with huge dimensions (we have like 30 here) svm Better