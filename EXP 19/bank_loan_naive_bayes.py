import math
from collections import defaultdict

# Sample dataset: [Income(1000s), Credit Score(0-100)]
X = [
    [30, 70],
    [50, 80],
    [40, 65],
    [20, 50],
    [60, 90]
]
y = ["No", "Yes", "Yes", "No", "Yes"]

# Gaussian Naive Bayes
class GaussianNB:
    def fit(self, X, y):
        self.classes = set(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        
        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i]==c]
            self.mean[c] = [sum(f)/len(f) for f in zip(*X_c)]
            self.var[c] = [sum((f-m)**2 for f,m in zip(feature,self.mean[c]))/len(feature)
                           for feature in zip(*X_c)]
            self.prior[c] = len(X_c)/len(X)
    
    def gaussian_prob(self, x, mean, var):
        eps = 1e-6
        return (1 / math.sqrt(2*math.pi*var + eps)) * math.exp(-((x-mean)**2)/(2*var+eps))
    
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

# Train and predict
nb = GaussianNB()
nb.fit(X, y)

test_sample = [[45, 75]]
result = nb.predict(test_sample)
print("Predicted Loan Approval:", result[0])
