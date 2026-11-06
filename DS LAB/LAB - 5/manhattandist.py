import numpy as np
from scipy.spatial import distance

pointA = np.array([2,4,6])
pointB = np.array([5,1,9])

euclidean_distance = distance.euclidean(pointA, pointB)
print("Euclidean Distance :", euclidean_distance)

similarity_euclidean = 1 / (1 + euclidean_distance)
print("Euclidean Similarity :", similarity_euclidean)

manhattan_distance = distance.cityblock(pointA, pointB)
print("Manhattan Distance :", manhattan_distance)

similarity_manhattan = 1 / (1 + manhattan_distance)
print("Manhattan Similarity :", similarity_manhattan)