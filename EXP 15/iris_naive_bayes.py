import math
from collections import defaultdict

# -------------------------------
# Sample Iris Dataset (simplified)
# Features: [Sepal Length, Sepal Width, Petal Length, Petal Width]
# Labels: 'Setosa', 'Versicolor', 'Virginica'
# -------------------------------
X = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
    [5.6, 2.9, 3.6, 1.3],
    [6.7, 3.1, 4.7, 1.5]
]
y = ["Setosa", "Setosa", "Virginica", "Virginica", "Versicolor", "Versicolor"]

# -------------------------------
# Gaussian Naive Bayes Classifier
# -------------------------------
class GaussianNB:
    def fit(self, X, y):
        self.classes = set(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        
        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.mean[c] = [sum(feature)/len(feature) for feature in zip(*X_c)]
            self.var[c] = [sum((f - m)**2 for f, m in zip(feature, self.mean[c])) / len(feature)
                           for feature in zip(*X_c)]
            self.prior[c] = len(X_c) / len(X)
    
    def gaussian_prob(self, x, mean, var):
        eps = 1e-6  # to avoid division by zero
        return (1 / math.sqrt(2 * math.pi * var + eps)) * math.exp(-((x - mean) ** 2) / (2 * var + eps))
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            probs = {}
            for c in self.classes:
                prob = self.prior[c]
                for i in range(len(x)):
                    prob *= self.gaussian_prob(x[i], self.mean[c][i], self.var[c][i])
                probs[c] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions

# -------------------------------
# Train and Predict
# -------------------------------
nb = GaussianNB()
nb.fit(X, y)

# Test sample
X_test = [[5.8, 2.7, 4.1, 1.0]]
y_pred = nb.predict(X_test)

print("Predicted Iris Flower Class:", y_pred[0])
