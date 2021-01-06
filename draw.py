import matplotlib.pyplot as plt
import json
from datetime import datetime

i = 0;
with open('C:/default.json') as json_file:
    data = json.load(json_file)

sortedArray = sorted(
    data['Location'],
    key=lambda x: datetime.strptime(x['createdAt'], "%d-%m-%Y %H:%M:%S"))

for p in sortedArray:
    x = p['locationX']
    y = p['locationY']
    plt.scatter(x, y, c='blue')
    plt.text(x, y, i)
    i = i + 1;

plt.show()
