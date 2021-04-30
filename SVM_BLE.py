import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

names = ['rssi1', 'rssi2', 'rssi_wifi', '25_area', '10_area', '2_area', 'trajectory', '3_area']
data_set = pd.read_csv("C:/Thesis/data/default_ble.csv", names=names)
print(data_set.head())

x = data_set.iloc[:, :-5].values
y = data_set.iloc[:, 5].values

i = 0
acc = 0
while i < 1000:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    # Specifies the kernel type to be used in the algorithm.
    # It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
    # If none is given, ‘rbf’ will be used.
    # If a callable is given it is used to pre-compute the kernel matrix from data matrices;
    # that matrix should be an array of shape (n_samples, n_samples).

    cls = svm.SVC(kernel="precomputed")
    cls.fit(x_train, y_train)
    pred = cls.predict(x_test)
    acc += metrics.accuracy_score(y_test, y_pred=pred)
    i += 1

# print(metrics.classification_report(y_test,y_pred=pred))
print("Ortalama:", (acc / 1000) * 100)
