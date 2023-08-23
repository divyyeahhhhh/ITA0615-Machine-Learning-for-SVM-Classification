import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
# Create a polynomial features object
poly_features = PolynomialFeatures(degree=2)
# Transform the data
x_poly = poly_features.fit_transform(np.array(x).reshape((-1, 1)))
# Fit the model
model = LinearRegression()
model.fit(x_poly, y)
# Plot the data and the fitted line
plt.scatter(x, y)
plt.plot(x, model.predict(x_poly), color='red')
plt.show()
