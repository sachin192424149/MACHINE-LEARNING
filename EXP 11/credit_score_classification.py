import math

# Sample dataset
X = [
    [30000, 25, 5000],
    [60000, 45, 10000],
    [45000, 35, 7000],
    [80000, 50, 20000],
    [20000, 23, 3000]
]

# Labels: 0 = Bad, 1 = Good
y = [0, 1, 1, 1, 0]

weights = [0.0, 0.0, 0.0]
bias = 0.0
learning_rate = 0.000001
epochs = 500

# Safe sigmoid function
def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))

# Training
for _ in range(epochs):
    for i in range(len(X)):
        z = sum(weights[j] * X[i][j] for j in range(len(weights))) + bias
        prediction = sigmoid(z)
        error = y[i] - prediction

        for j in range(len(weights)):
            weights[j] += learning_rate * error * X[i][j]
        bias += learning_rate * error

# Prediction
def predict(sample):
    z = sum(weights[j] * sample[j] for j in range(len(weights))) + bias
    return 1 if sigmoid(z) >= 0.5 else 0

test_customer = [50000, 30, 8000]

result = predict(test_customer)
print("Credit Score:", "Good" if result == 1 else "Bad")
