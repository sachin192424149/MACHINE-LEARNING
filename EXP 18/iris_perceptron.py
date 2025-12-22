import numpy as np

# Sample Iris Dataset (only 2 classes for simplicity: Setosa and Versicolor)
X = np.array([
    [5.1, 3.5],
    [4.9, 3.0],
    [5.6, 2.9],
    [5.8, 2.7]
])
y = np.array([1, 1, -1, -1])  # 1=Setosa, -1=Versicolor

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0
learning_rate = 0.1
epochs = 10

# Training Perceptron
for _ in range(epochs):
    for xi, target in zip(X, y):
        update = learning_rate * (target - np.sign(np.dot(w, xi) + b))
        w += update * xi
        b += update

# Prediction
def predict(x):
    return 1 if np.dot(w, x) + b >= 0 else -1

test_sample = np.array([5.4, 3.0])
result = predict(test_sample)
print("Predicted Iris Class:", "Setosa" if result==1 else "Versicolor")
