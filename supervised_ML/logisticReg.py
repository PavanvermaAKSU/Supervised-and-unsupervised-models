import numpy as np
from sklearn.linear_model import LogisticRegression

n = int(input("Enter number of data points: "))

X = []
y = []

for i in range(n):
    x_val = float(input("Enter feature value: "))
    label = int(input("Enter class (0 or 1): "))
    X.append([x_val])
    y.append(label)

X = np.array(X)
y = np.array(y)

model = LogisticRegression()
model.fit(X, y)

test = float(input("Enter value to classify: "))
prediction = model.predict([[test]])

print("Predicted Class:", prediction)