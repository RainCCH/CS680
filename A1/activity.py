import numpy as np
from tqdm import tqdm
from itertools import combinations
import argparse

class Activity:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.load_data()

    def ReadX(self, path):
        print(f'>>> Reading data from: {path} ...')
        with open(path) as f:
            # only one line that includes everything
            file = f.readlines()

        print(f'#instances: {len(file)}') # 7352 for training set, 2947 for test set

        X_all = []
        for instance in file:
            f = filter(None, instance.split(' '))
            instance_filterd = list(f)
            instance_cleaned = [float(attr.strip()) for attr in instance_filterd]
            X_all.append(instance_cleaned)
        X_all = np.array(X_all)
        print('>>> Reading finished! Data are converted to numpy array.')
        print(f'shape of X: {X_all.shape} ==> each instance has {X_all.shape[1]} attributes.')

        return X_all

    def ReadY(self, path):
        print(f'>>> Reading data from: {path} ...')
        with open(path) as f:
            # only one line that includes everything
            file = f.readlines()
            print(f'#instances: {len(file)}')  # 7352 for training set, 2947 for test set

        y_all = [int(label.strip()) for label in file]
        y_all = np.array(y_all)
        print('>>> Reading finished! Data are converted to numpy array.')
        print(f'shape of y: {y_all.shape}')
        return y_all
    
    def load_data(self):
        self.X_train = self.ReadX('UCI HAR Dataset/train/X_train.txt')
        self.y_train = self.ReadY('UCI HAR Dataset/train/y_train.txt')
        self.X_test = self.ReadX('UCI HAR Dataset/test/X_test.txt')
        self.y_test = self.ReadY('UCI HAR Dataset/test/y_test.txt')
        print('Training set: X:', self.X_train.shape, 'y:', self.y_train.shape)
        print('Test set: X:', self.X_test.shape, 'y:', self.y_test.shape)
        print('>>> Data loaded successfully!')

# Q1.2
class MultiClass_OvA_Perceptron:
    # A one-vs-all perceptron classifier
    def __init__(self, max_pass=500):
        self.max_pass = max_pass
        self.weights = None
        self.bias = None
        self.classes = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y -= 1
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)
        for _ in tqdm(range(self.max_pass), desc="Training Progress"):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights.T) + self.bias
                y_pred = np.argmax(linear_output)
                if y_pred != y[idx]:
                    self.weights[y[idx]] += x_i
                    self.weights[y_pred] -= x_i
                    self.bias[y[idx]] += 1
                    self.bias[y_pred] -= 1
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights.T) + self.bias
        y_predicted = np.argmax(linear_output, axis=1)+1
        return y_predicted
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
# Q1.3
class MultiClass_OvO_Perceptron:
    # A one-vs-one perceptron classifier
    def __init__(self, max_pass=500):
        self.max_pass = max_pass
        self.classifiers = {}
    
    def fit(self, X, y):
        y -= 1
        self.classes = np.unique(y)
        self.class_pairs = list(combinations(self.classes, 2))
        for (class1, class2) in tqdm(self.class_pairs, desc="Training Classifiers"):
            # Select data points belonging to the two classes
            idx = np.where((y == class1) | (y == class2))
            X_pair = X[idx]
            y_pair = y[idx]
            y_pair = np.where(y_pair == class1, 1, -1)
            # Initialize the weights and bias
            weights = np.zeros(X.shape[1])
            bias = 0
            # Train the binary perceptron
            for _ in range(self.max_pass):
                for i, x_i in enumerate(X_pair):
                    if y_pair[i] * (np.dot(x_i, weights) + bias) <= 0:
                        weights += y_pair[i] * x_i
                        bias += y_pair[i]
            self.classifiers[(class1, class2)] = (weights, bias)
    
    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes)))
        for (class1, class2), (weights, bias) in self.classifiers.items():
            linear_output = np.dot(X, weights) + bias
            y_pred = np.where(linear_output >= 0, class1, class2)
            for i, pred in enumerate(y_pred):
                votes[i, pred] += 1
        y_predicted = np.argmax(votes, axis=1)+1
        return y_predicted
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

# Q1.6
class MulticlassPerceptron:
    def __init__(self):
        self.num_classes = None
        self.weights = None

    def fit(self, X, y, epochs=500):
        y -= 1
        self.num_classes = len(np.unique(y))
        self.weights = np.zeros((self.num_classes, X.shape[1]))
        for _ in tqdm(range(epochs)):
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]
                y_hat = np.argmax(np.dot(self.weights, x_i))
                if y_hat != y_i:
                    self.weights[y_hat] -= x_i
                    self.weights[y_i] += x_i

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat = np.argmax(np.dot(self.weights, x))+1
            predictions.append(y_hat)
        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose a classifier to run.')
    parser.add_argument('--classifier', choices=['OvA', 'OvO', 'Multi'], help='Specify which classifier to run: OvA, OvO, Multi', default='OvA')
    args = parser.parse_args()
    if args.classifier == 'OvA':
        p = MultiClass_OvA_Perceptron(max_pass=500)
        print('Running One-vs-All Perceptron Classifier ...')
    elif args.classifier == 'OvO':
        p = MultiClass_OvO_Perceptron(max_pass=500)
        print('Running One-vs-One Perceptron Classifier ...')
    else:
        p = MulticlassPerceptron()
        print('Running Multi-class Perceptron Classifier ...')
    a = Activity()
    p.fit(a.X_train, a.y_train)
    y_pred = p.predict(a.X_test)
    acc = p.accuracy(a.y_test, y_pred)
    y_pred_train = p.predict(a.X_train)
    acc_train = p.accuracy(a.y_train, y_pred_train-1)
    print(f'Test set accuracy: {acc}')
    print(f'Training set accuracy: {acc_train}')