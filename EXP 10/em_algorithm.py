import math

# Sample 1-D data
data = [1.2, 1.8, 2.0, 5.0, 6.0, 6.5]

# Number of clusters
K = 2
n = len(data)

# Initial parameters
means = [2.0, 6.0]
variances = [1.0, 1.0]
weights = [0.5, 0.5]

# Gaussian probability density function
def gaussian(x, mean, var):
    return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mean) ** 2) / (2 * var))

# EM Algorithm
for iteration in range(5):

    # ---------- E-Step ----------
    responsibilities = []
    for x in data:
        probs = []
        for k in range(K):
            probs.append(weights[k] * gaussian(x, means[k], variances[k]))
        total = sum(probs)
        responsibilities.append([p / total for p in probs])

    # ---------- M-Step ----------
    for k in range(K):
        Nk = sum(responsibilities[i][k] for i in range(n))

        means[k] = sum(responsibilities[i][k] * data[i] for i in range(n)) / Nk
        variances[k] = sum(
            responsibilities[i][k] * (data[i] - means[k]) ** 2
            for i in range(n)
        ) / Nk
        weights[k] = Nk / n

# Output
print("Final Means:", means)
print("Final Variances:", variances)
print("Final Weights:", weights)
