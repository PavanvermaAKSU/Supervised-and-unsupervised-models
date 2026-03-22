import numpy as np
from sklearn.cluster import KMeans

n = int(input("Enter number of data points: "))

data = []
for i in range(n):
    x = float(input("Enter x value: "))
    y = float(input("Enter y value: "))
    data.append([x, y])

data = np.array(data)

k = int(input("Enter number of clusters: "))

model = KMeans(n_clusters=k)
model.fit(data)

labels = model.predict(data)

print("Cluster Labels:", labels)
print("Cluster Centers:")
print(model.cluster_centers_)