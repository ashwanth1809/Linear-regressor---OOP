import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """w is slope, b is intercept, alpha is learning rate, num_iters is iterations for grad descent."""

    def __init__(self, x, y):
        self.w = 0
        self.b = 0
        self.total_cost = 0
        self.x = x
        self.y = y
        self.num_iters = 10000
        self.alpha = 0.01
        self.total_cost_list = np.zeros(self.num_iters)

        self.w, self.b, self.total_cost_list = self.gradient_descent()

    def plotter(self, y_fit):
        """This method is used for plotting raw data, fit line, cost function."""
        figure, axis = plt.subplots(1, 3)
        axis[0].scatter(self.x, self.y, marker='x', c='red')
        axis[0].plot(self.x, y_fit, c='black')
        axis[1].plot(range(10000), self.total_cost_list)
        axis[1].set_xlim(-500, 500)
        axis[2].plot(range(10000), self.total_cost_list)
        plt.show()

    def fit(self):
        """This method is used to fit the line using simple linear regression."""
        m = self.x.shape[0]
        f = np.zeros(m)
        for i in range(m):
            f[i] = self.x[i] * self.w + self.b
        return f

    def cost_function_calculator(self):
        """This method is used to calculate the cost function"""
        m = self.x.shape[0]
        cost = 0
        for i in range(m):
            f = self.x[i] * self.w + self.b
            cost += (f - self.y[i]) ** 2
        self.total_cost = 1 / (2 * m) * cost
        return self.total_cost

    def compute_gradient(self):
        """This method computes the gradient descent, given a value of w and b."""
        m = self.x.shape[0]
        dj_dw = 0
        dj_db = 0
        for i in range(m):
            f = self.x[i] * self.w + self.b
            dj_dw += (f - self.y[i]) * self.x[i]
            dj_db += f - self.y[i]
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def gradient_descent(self):
        """This method updates w and b by computing the gradient and cost function each time."""
        self.total_cost = np.zeros(self.num_iters)
        for i in range(self.num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = self.compute_gradient()
            self.total_cost_list[i] = self.cost_function_calculator()

            # Update Parameters using equation (3) above
            self.b -= self.alpha * dj_db
            self.w -= self.alpha * dj_dw

        return self.w, self.b, self.total_cost_list

    def predict(self, x_new):
        """This method is used to predict the the dependent variable (y_hat), given a new input value (x)"""
        y_new = self.w * x_new + self.b

        return y_new
