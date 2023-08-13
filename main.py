import matplotlib.pyplot as plt
import numpy as np

# The given x and y value (training set)
x_train = np.array(range(10))
y_train = np.array([25, 85, 279, 356, 659, 895, 1025, 1413, 1610, 1919])

# importing the LinearRegression class
from lr import LinearRegression
# create an object instance
lr = LinearRegression(x_train, y_train)
# fit the x and y value (entire data set)
y_hat = lr.fit()
# extract the final value of the coefficients b and w
print(lr.b)
print(lr.w)
# Predict new input
print(lr.predict(5.5))
# Plot the fit line, and cost function
lr.plotter(y_hat)
