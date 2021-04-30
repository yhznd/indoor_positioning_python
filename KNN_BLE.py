import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score

names = ['rssi1', 'rssi2', 'rssi_wifi','25_area','10_area','2_area','trajectory','3_area']
data_set = pd.read_csv("C:/Thesis/data/default_ble.csv", names=names)
print(data_set.head())

x = data_set.iloc[:, :-5].values
y = data_set.iloc[:, 5].values

i=0
acc=0
while i<1000:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    #p int, default=2
    #Power parameter for the Minkowski metric.
    # When p = 1, this is equivalent to using manhattan_distance (l1),
    # When p = 2, this is equivalent to using euclidean_distance (l2),
    # For arbitrary p, this is equivalent to using minkowski_distance (l_p)

    classfier = KNeighborsClassifier(p=2)
    classfier.fit(x_train, y_train)

    y_pred = classfier.predict(x_test)
    acc += accuracy_score(y_test, y_pred)
    i+=1


#print(classification_report(y_test, y_pred))
print("Ortalama:", (acc/1000)*100)
