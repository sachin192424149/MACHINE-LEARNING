import math
from collections import Counter

# Calculate Euclidean distance
def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)

# KNN Algorithm
def knn(train_data, train_labels, test_point, k):
    distances = []

    # Calculate distance from test point to all training points
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_point)
        distances.append((dist, train_labels[i]))

    # Sort distances
    distances.sort(key=lambda x: x[0])

    # Select k nearest neighbors
    k_neighbors = distances[:k]

    # Get class labels
    labels = [label for _, label in k_neighbors]

    # Majority vote
    prediction = Counter(labels).most_common(1)[0][0]
    return prediction

# Sample Training Data
train_data = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7]
]

train_labels = ['A', 'A', 'A', 'B', 'B']

# Test Data Point
test_point = [3, 2]
k = 3

# Predict class
result = knn(train_data, train_labels, test_point, k)

print("Predicted Class:", result)
