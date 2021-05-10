import numpy as np
import matplotlib.pyplot as plt

# 定義一個類別及所需的變數
class LinearRegression:
    def __init__(self, num_iteration=100, learning_rate=1e-1, feature_scaling=True):
        self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        self.feature_scaling = feature_scaling
        self.M = 0
        self.S = 1
        self.W = None
        self.cost_history = np.empty(num_iteration)

    # Gradient Descent
    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        m, n = X.shape

        if self.feature_scaling:
            X = self.normalize(X)

        X = np.hstack((np.ones((m, 1)), X))
        y = y.reshape(y.shape[0], 1)
        self.W = np.zeros((n + 1, 1))

        for i in range(self.num_iteration):
            y_hat = X.dot(self.W)
            cost = self.cost_function(y_hat, y, m)
            self.cost_history[i] = cost
            self.gradient_descent(X, y_hat, y, m)


    # fit函式
    def normalize(self, X):
        self.M = np.mean(X, axis=0)
        self.S = np.max(X, axis=0) - np.min(X, axis=0)
        return (X - self.M) / self.S


    def cost_function(self, y_hat, y, m):
        return 1 / (2 * m) * np.sum((y_hat - y) ** 2)


    def compute_gradient(self, X, y_hat, y, m):
        return 1 / m * np.sum((y_hat - y) * X, axis=0).reshape(-1, 1)


    def gradient_descent(self, X, y_hat, y, m):
        self.W -= self.learning_rate * self.compute_gradient(X, y_hat, y, m)


    # prediction
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        m, n = X.shape
        if self.normalize:
            X = (X - self.M) / self.S
        X = np.hstack((np.ones((m, 1)), X))
        y_hat = X.dot(self.W)
        return y_hat


    # normal equation
    def normal_equation(self, X, y):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))
        y = y.reshape(y.shape[0], 1)
        self.W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


# Generate Univariate dataset
def generate_univariate_dataset():
    X = np.random.uniform(low=0, high=20, size=(30, ))
    Y = np.array([ x + np.random.uniform(low=-2, high=2) for x in X ])
    return {'X': X, 'Y': Y}

# Plot the figure then show
def plot_and_show(dataset, model=None, cost=False):
    if model and cost:
        cost = model.cost_history
        pred = model.predict(dataset['X'])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(dataset['X'], dataset['Y'])
        ax1.plot(dataset['X'], pred, color='r')
        ax2.plot(range(cost.shape[0]), cost)
        plt.show()
        
    elif model and not cost:
        pred = model.predict(dataset['X'])

        fig, ax = plt.subplots()
        ax.scatter(dataset['X'], dataset['Y'])
        ax.plot(dataset['X'], pred, color='r')
        plt.show()
        
    else:
        fig, ax = plt.subplots()
        ax.scatter(dataset['X'], dataset['Y'])
        plt.show()
        

def main():

    # generate dataset
    dataset = generate_univariate_dataset()
    plot_and_show(dataset)

    # no feature scaling
    model = LinearRegression(learning_rate=1e-3, feature_scaling=False)
    model.fit(dataset['X'], dataset['Y'])
    plot_and_show(dataset, model, cost=True)

    # feature scaling
    model = LinearRegression(learning_rate=1e-0, feature_scaling=True)
    model.fit(dataset['X'], dataset['Y'])
    plot_and_show(dataset, model, cost=True)

    # normal equation
    model = LinearRegression(feature_scaling=False)
    model.normal_equation(dataset['X'], dataset['Y'])
    plot_and_show(dataset, model)



main()