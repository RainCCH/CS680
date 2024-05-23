from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SpamBase:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
    
    def load_website_data(self):
        self.data = fetch_ucirepo(id=94)
        self.X = pd.DataFrame(data=self.data.data.features , columns=self.data.data.feature_names)
        self.y = pd.DataFrame(data=self.data.data.targets , columns=self.data.data.target_names)
        # Change data to numpy array
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy().flatten()
        print("Spambase dataset loaded. X:", self.X.shape, "y:", self.y.shape)
    
    def load_data(self):
        # Load data from csv files and change X's axis
        self.X = pd.read_csv('./spambase/spambase_X.csv').to_numpy().T
        self.y = pd.read_csv('./spambase/spambase_y.csv', header=None, names=['label']).to_numpy().flatten()
        print("Spambase dataset loaded. X:", self.X.shape, "y:", self.y.shape)

class Perceptron:
    def __init__(self, max_pass=500):
        self.max_pass = max_pass
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Initialize list to store the number of mistakes in each pass
        mistakes = []
        # Train perceptron
        # Apply tqdm to show progress bar
        for _ in tqdm(range(self.max_pass), desc="Training Progress"):
            mistake_count = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y[idx] * linear_output <= 0:
                    self.weights += y[idx] * x_i
                    self.bias += y[idx]
                    mistake_count += 1
            mistakes.append(mistake_count)
        return mistakes

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
    # Activation function
    def activation_func(self, x):
        return np.where(x >= 0, 1, -1)

if __name__ == "__main__":
    # Load dataset
    spam = SpamBase()
    spam.load_data()

    # Train perceptron
    p = Perceptron(max_pass=500)
    mistakes = p.fit(spam.X, spam.y)
    
    # Print weights and bias
    # print("Weights:", p.weights)
    # print("Bias:", p.bias)
    
    # Plot the number of mistakes vs. number of passes
    plt.plot(range(1, 501), mistakes)
    plt.xlabel('Number of Passes')
    plt.ylabel('Number of Mistakes')
    plt.title('Perceptron Mistakes vs. Passes on Spambase Dataset')
    plt.show()