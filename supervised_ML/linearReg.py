import numpy as np
from sklearn.linear_model import LinearRegression

n = int(input("Enter number of data points: "))

X = []
y = []

for i in range(n):
    x_val = float(input("Enter X value: "))
    y_val = float(input("Enter Y value: "))
    X.append([x_val])
    y.append(y_val)

X = np.array(X)
y = np.array(y)

model = LinearRegression()
model.fit(X, y)

test = float(input("Enter value to predict: "))
prediction = model.predict([[test]])

print("Predicted Output:", prediction)