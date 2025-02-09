import numpy as np


class GDRegression:
    def __init__(self, iterations: int, alpha: float):
        self.iterations = iterations
        self.alpha = alpha
        self.W = None
        self.B = None
        self.history = []


    def initialize_params(self, X: np.ndarray):
        """ Initializes the parameters for the model. """

        self.W = np.zeros((X.shape[1], 1))
        self.B = np.zeros((1, 1))
    

    def make_prediction(self, X: np.ndarray) -> np.ndarray:
        """ Makes prediction on given dataset. """

        Y_hat = X @ self.W + self.B

        return Y_hat


    def compute_cost(self, Y_hat: np.ndarray, Y: np.ndarray) -> np.float64:
        """ Computes the MSE cost. """

        cost = np.mean((Y_hat - Y)**2)

        return cost

    
    def compute_gradient(self, X: np.ndarray, Y_hat: np.ndarray, Y: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray]:
        """ Computes the gradients for the parameters update. """

        m = X.__len__()
        dW = (2 / m) * (X.T @ (Y_hat - Y))
        dB = (2 / m) * np.sum((Y_hat - Y), keepdims=True)
        
        return dW, dB
    

    def update_params(self, dW: np.ndarray, dB: np.ndarray):
        """ Updates the parameters having provided the gradients. """

        self.W -= self.alpha * dW
        self.B -= self.alpha * dB


    def fit(self, X: np.ndarray, Y: np.ndarray):
        """ Train the Regression model using gradient descent. """

        self.initialize_params(X)

        for i in range(self.iterations):
            print(f"Iteration: {i+1}/{self.iterations}")

            Y_hat = self.make_prediction(X)
            
            cost = self.compute_cost(Y_hat, Y)
            print(f"Cost: {cost}\n")

            dW, dB = self.compute_gradient(X, Y_hat, Y)
            self.update_params(dW, dB)

            self.history.append((i+1, cost))
