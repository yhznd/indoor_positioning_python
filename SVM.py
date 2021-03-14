import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

names = ['rssi1', 'rssi2', 'rssi3','class']
data_set = pd.read_csv("C:/Thesis/data/default_last (1).csv", names=names)
print(data_set.head())

x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.10)

cls=svm.SVC(kernel="linear")
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
print("accuracy",metrics.accuracy_score(y_test,y_pred=pred))
print("precision",metrics.precision_score(y_test,y_pred=pred, average='micro'))
print("recall score",metrics.precision_score(y_test,y_pred=pred, average='micro'))
print(metrics.classification_report(y_test,y_pred=pred))