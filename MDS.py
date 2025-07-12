import math
import matplotlib.pyplot as plt
import numpy as np

coordinates = [[37.5665, 126.9780], [7.0051, 110.4381], [35.6762, 139.6503],
               [35.0116, 135.7681], [26.6476, 106.6302], [3.1390, 101.6869],
               [39.9042, 116.4074], [31.2304, 121.4737], [30.2741, 120.1152],
               [23.1291, 113.2644]]

distance_matrix = []
distance_squared_matrix = []
for city in coordinates:
    dist_list = []
    dist_squared_list = []
    for other_city in coordinates:
        x_dis = abs(other_city[0] - city[0])
        y_dis = abs(other_city[1] - city[1])
        dist = math.sqrt(x_dis*x_dis + y_dis*y_dis)
        dist_squared = x_dis*x_dis + y_dis*y_dis
        dist_list.append(dist)
        dist_squared_list.append(dist_squared)
    distance_matrix.append(dist_list)
    distance_squared_matrix.append(dist_squared_list)
print(distance_matrix)

J = np.identity(10) - np.ones(10)/10
B = - np.matmul(np.matmul(J, distance_squared_matrix), J)/2

eigenvalue, eigenvector = np.linalg.eig(B)
eigenvalue_list = []
for i in eigenvalue:
    eigenvalue_list.append(i)
temp_list = eigenvalue_list.copy()
temp_list.remove(max(eigenvalue_list))
largest_index = eigenvalue_list.index(max(eigenvalue_list))
second_index = eigenvalue_list.index(max(temp_list))
print(eigenvector[largest_index])
print(eigenvector[second_index])

largest_eigval = [[math.sqrt(max(eigenvalue_list)), 0], [0, math.sqrt(max(temp_list))]]
X = np.dot(largest_eigval, [eigenvector[largest_index], eigenvector[second_index]])
X = X.T
print(X)

plt.scatter(-X[:, 0], -X[:, 1], s=25, c='black', alpha=0.5)
plt.show()
