# Define the LinearRegression class
import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the model with learning rate (lr) and the number of iterations for gradient descent.
        The weights and bias are initialized as None and will be set during training.
        """
        self.lr = lr  # Learning rate
        self.n_iters = n_iters  # Number of iterations
        self.weights = None  # Placeholder for weights (coefficients)
        self.bias = None  # Placeholder for bias (intercept)

    def fit(self, X, y):
        """
        Fit the linear regression model to the data using gradient descent.
        X: Input features (numpy array), shape (n_samples, n_features)
        y: Target values (numpy array), shape (n_samples,)
        """
        n_samples, n_features = X.shape  # Get number of samples and features
        self.weights = np.zeros(n_features)  # Initialize weights with zeros
        self.bias = 0  # Initialize bias to zero

        # Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias  # Linear model prediction
            dw = (1 / n_samples) * (
                np.dot(X.T, (y_predicted - y))
            )  # Compute gradient w.r.t. weights
            db = (1 / n_samples) * np.sum(
                y_predicted - y
            )  # Compute gradient w.r.t. bias

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Make predictions using the learned linear model.
        X: Input features (numpy array), shape (n_samples, n_features)
        Returns: Predicted values (numpy array), shape (n_samples,)
        """
        y_predicted = np.dot(X, self.weights) + self.bias  # Linear model prediction
        return y_predicted
