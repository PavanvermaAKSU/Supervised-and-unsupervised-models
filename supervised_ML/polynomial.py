import numpy as np
from sklearn.preprocessing import PolynomialFeatures
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

degree = int(input("Enter polynomial degree: "))

poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

test = float(input("Enter value to predict: "))
test_poly = poly.transform([[test]])

prediction = model.predict(test_poly)

print("Predicted Output:", prediction)