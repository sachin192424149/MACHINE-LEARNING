# Car Price Prediction using Linear Regression (From Scratch)

# Sample dataset
# Feature: Car Age (years)
# Target: Car Price (in lakhs)
X = [1, 2, 3, 4, 5]
Y = [9, 8, 7, 6, 5]

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

# Predict car price
car_age = 6
predicted_price = m * car_age + b

print("Predicted Car Price for age", car_age, "years is:", predicted_price, "lakhs")
