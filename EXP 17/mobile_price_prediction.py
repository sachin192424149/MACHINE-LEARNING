# Mobile Price Prediction using Linear Regression (From Scratch)

# Sample dataset: [RAM(GB), Battery(mAh), Weight(g)]
X = [
    [4, 3000, 150],
    [6, 3500, 160],
    [8, 4000, 170],
    [3, 2800, 145],
    [2, 2500, 140]
]
# Target: Price (in thousand rupees)
Y = [15, 20, 25, 12, 10]

n = len(X)

# Using only single feature for simplicity (RAM)
x_single = [row[0] for row in X]

sum_x = sum(x_single)
sum_y = sum(Y)
sum_xy = sum(x_single[i] * Y[i] for i in range(n))
sum_x2 = sum(x_single[i] ** 2 for i in range(n))

m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / n

print("Slope (m):", m)
print("Intercept (b):", b)

# Predict price for new mobile with 7 GB RAM
ram_new = 7
predicted_price = m * ram_new + b
print("Predicted Mobile Price for 7GB RAM:", predicted_price, "thousand rupees")
