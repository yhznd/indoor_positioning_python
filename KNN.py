import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

names = ['rssi1', 'rssi2', 'rssi3', 'class']
data_set = pd.read_csv("C:/Thesis/data/result-excel.csv", names=names)
print(data_set.head())

x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classfier = KNeighborsClassifier(n_neighbors=7)
classfier.fit(x_train, y_train)

y_pred = classfier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(acc)