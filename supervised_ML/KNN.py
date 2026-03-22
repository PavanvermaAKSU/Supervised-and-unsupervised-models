import numpy as np
from sklearn.neighbors import KNeighborsClassifier

n = int(input("Enter number of data points: "))

X = []
y = []

for i in range(n):
    x_val = float(input("Enter feature value: "))
    label = int(input("Enter class label: "))
    X.append([x_val])
    y.append(label)

X = np.array(X)
y = np.array(y)

k = int(input("Enter value of K: "))

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)

test = float(input("Enter value to classify: "))
prediction = model.predict([[test]])

print("Predicted Class:", prediction)