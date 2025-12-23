# future_sales_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Create a CSV file named 'sales_data.csv' with columns: Advertising, Sales
data = pd.read_csv("sales_data.csv")

# Independent and Dependent variables
X = data[['Advertising']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict future sales
future_advertising = np.array([[200]])  # Example future advertising budget
future_sales = model.predict(future_advertising)

print("Predicted Future Sales for Advertising = 200:", future_sales[0])

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X), linewidth=2)
plt.xlabel("Advertising Budget")
plt.ylabel("Sales")
plt.title("Future Sales Prediction using Linear Regression")
plt.show()
