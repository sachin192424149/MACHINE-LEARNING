# Linear Regression using Least Squares Method

def linear_regression(x, y):
    n = len(x)

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] ** 2 for i in range(n))

    # Calculate slope (m) and intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    return m, b

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Train the model
m, b = linear_regression(x, y)

print("Slope (m):", m)
print("Intercept (b):", b)

# Predict value
x_new = 6
y_pred = m * x_new + b
print("Predicted value for x =", x_new, "is", y_pred)
