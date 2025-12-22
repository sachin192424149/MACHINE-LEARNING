# House Price Prediction using Linear Regression (From Scratch)

# Sample dataset
# Feature: House Size (in square feet)
# Target: House Price (in lakhs)
X = [800, 1000, 1200, 1500, 1800]
Y = [40, 50, 60, 75, 90]

n = len(X)

# Calculate required sums
sum_x = sum(X)
sum_y = sum(Y)
sum_xy = sum(X[i] * Y[i] for i in range(n))
sum_x2 = sum(X[i] ** 2 for i in range(n))

# Calculate slope (m) and intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / n

print("Slope (m):", m)
print("Intercept (b):", b)

# Predict house price for new size
house_size = 1400
predicted_price = m * house_size + b

print("Predicted House Price for", house_size, "sq.ft is:", predicted_price, "lakhs")
