import math
from collections import Counter

# -------------------------------
# Euclidean Distance Function
# -------------------------------
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

# -------------------------------
# KNN Algorithm
# -------------------------------
def knn(train_data, train_labels, test_point, k):
    distances = []

    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_point)
        distances.append((dist, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    labels = [label for _, label in k_nearest]
    return Counter(labels).most_common(1)[0][0]

# -------------------------------
# Sample Iris Dataset
# Features: [Sepal Length, Sepal Width, Petal Length, Petal Width]
# -------------------------------
train_data = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
    [5.6, 2.9, 3.6, 1.3],
    [6.7, 3.1, 4.7, 1.5]
]

train_labels = [
    "Setosa",
    "Setosa",
    "Virginica",
    "Virginica",
    "Versicolor",
    "Versicolor"
]

# -------------------------------
# Test Sample
# -------------------------------
test_flower = [5.8, 2.7, 4.1, 1.0]
k = 3

# Prediction
result = knn(train_data, train_labels, test_flower, k)

print("Predicted Iris Flower Class:", result)
