import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class NeuralNet:
    def __init__(self,
                 hidden_layers=[20],
                 hidden_activations=['relu'],
                 output_activation='sigmoid',
                 learning_rate=0.01,
                 num_iter=30000,
                 normalize=True,
                 regularization=None,
                 lambda_reg=0.01,
                 early_stopping=False,
                 patience=1000,
                 batch_size=None):

        self.hidden_layers = hidden_layers
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.normalize = normalize
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size

    # --- Активаційні функції ---
    def _sigmoid(self, Z): return 1 / (1 + np.exp(-Z))
    def _relu(self, Z): return np.maximum(0, Z)
    def _tanh(self, Z): return np.tanh(Z)
    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def _activation(self, Z, kind):
        return {
            'sigmoid': self._sigmoid,
            'relu': self._relu,
            'tanh': self._tanh,
            'softmax': self._softmax
        }[kind](Z)

    def _activation_derivative(self, A, kind):
        if kind == 'sigmoid': return A * (1 - A)
        if kind == 'relu': return (A > 0).astype(float)
        if kind == 'tanh': return 1 - A ** 2
        return None  # softmax derivative handled separately

    # --- Ініціалізація вагів ---
    def _initialize_parameters(self, layer_dims):
        self.parameters = {}
        for l in range(1, len(layer_dims)):
            self.parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            self.parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    def _normalize(self, X, mean=None, std=None):
        n = X.shape[0]
        if mean is None:
            mean = np.mean(X, axis=1).reshape((n, 1))
        if std is None:
            std = np.std(X, axis=1).reshape((n, 1))

        epsilon = 1e-8
        std = np.where(std < epsilon, 1.0, std)
        return (X - mean) / std, mean, std

    # --- Forward pass ---
    def _forward(self, X):
        cache = {'A0': X}
        A = X
        L = len(self.hidden_layers) + 1
        all_activations = self.hidden_activations + [self.output_activation]

        for l in range(1, L + 1):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = W @ A + b
            A = self._activation(Z, all_activations[l - 1])
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        return A, cache

    # --- Backward pass ---
    def _backward(self, X, Y, cache):
        grads = {}
        m = X.shape[1]
        L = len(self.hidden_layers) + 1
        all_activations = self.hidden_activations + [self.output_activation]

        A_final = cache[f"A{L}"]
        dA = A_final - Y if self.output_activation == 'softmax' else (A_final - Y)

        for l in reversed(range(1, L + 1)):
            A_prev = cache[f"A{l - 1}"]
            Z = cache[f"Z{l}"]
            A = cache[f"A{l}"]

            if self.output_activation == 'softmax' and l == L:
                dZ = dA
            else:
                dZ = dA * self._activation_derivative(A, all_activations[l - 1])

            grads[f"dW{l}"] = (1 / m) * dZ @ A_prev.T
            grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if self.regularization == 'l2':
                grads[f"dW{l}"] += self.lambda_reg * self.parameters[f"W{l}"] / m
            elif self.regularization == 'l1':
                grads[f"dW{l}"] += self.lambda_reg * np.sign(self.parameters[f"W{l}"]) / m

            dA = self.parameters[f"W{l}"].T @ dZ

        return grads

    # --- Update parameters ---
    def _update_parameters(self, grads):
        L = len(self.hidden_layers) + 1
        for l in range(1, L + 1):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def _compute_cost(self, A, Y):
        m = Y.shape[1]
        if self.output_activation == 'softmax':
            logprobs = np.sum(Y * np.log(A + 1e-8), axis=0)
        else:
            logprobs = Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)
        cost = -np.mean(logprobs)
        return cost

    def fit(self, X_vert, Y_vert, print_cost=True):
        X, Y = X_vert.T, Y_vert.T
        if self.normalize:
            X, self._mean, self._std = self._normalize(X)

        input_dim = X.shape[0]
        output_dim = Y.shape[0] if len(Y.shape) > 1 else 1
        layer_dims = [input_dim] + self.hidden_layers + [output_dim]

        self._initialize_parameters(layer_dims)

        best_cost = float('inf')
        patience_counter = 0
        costs = []

        for i in range(self.num_iter):
            if self.batch_size:
                X, Y = shuffle(X.T, Y.T)
                X, Y = X.T, Y.T
                for start in range(0, X.shape[1], self.batch_size):
                    end = start + self.batch_size
                    A, cache = self._forward(X[:, start:end])
                    grads = self._backward(X[:, start:end], Y[:, start:end], cache)
                    self._update_parameters(grads)
            else:
                A, cache = self._forward(X)
                grads = self._backward(X, Y, cache)
                self._update_parameters(grads)

            if i % 100 == 0:
                A, _ = self._forward(X)
                cost = self._compute_cost(A, Y)
                costs.append(cost)
                if print_cost:
                    print(f"{i}-th iteration: cost = {cost}")

                if self.early_stopping:
                    if cost < best_cost:
                        best_cost = cost
                        patience_counter = 0
                    else:
                        patience_counter += 100
                        if patience_counter >= self.patience:
                            print("Early stopping triggered.")
                            break

        if print_cost:
            plt.plot(costs)
            plt.title("Cost over iterations")
            plt.xlabel("Iteration (per 100)")
            plt.ylabel("Cost")
            plt.show()

    def predict(self, X_vert):
        X = X_vert.T
        if self.normalize:
            X, self._mean, self._std = self._normalize(X, self._mean, self._std)
        A, _ = self._forward(X)
        return np.argmax(A, axis=0) if self.output_activation == 'softmax' else (A > 0.5).astype(int).flatten()

    def predict_proba(self, X_vert):
        X = X_vert.T
        if self.normalize:
            X, _, _ = self._normalize(X, self._mean, self._std)
        A, _ = self._forward(X)
        return A.T
