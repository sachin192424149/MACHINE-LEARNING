import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 8, 10])

# ----- Linear Regression -----
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

linear_mse = mean_squared_error(y, y_linear_pred)

# ----- Polynomial Regression (Degree 2) -----
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

poly_mse = mean_squared_error(y, y_poly_pred)

# Results
print("Linear Regression MSE:", linear_mse)
print("Polynomial Regression MSE:", poly_mse)
