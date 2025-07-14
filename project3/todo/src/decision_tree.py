"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np


class Node:
    def __init__(self, predicted):
        self.predicted = predicted
        self.feature_idx = -1
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.root = None
        self.feature_importance = None

    def fit(self, X, y):
        self.feature_importance = [0] * X.shape[1]
        self.root, _ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if X.size == 0:
            return None, 0
        values, counts = np.unique(y, return_counts=True)
        node = Node(predicted=values[np.argmax(counts)])
        if entropy(y) == 0 or (self.max_depth is not None) and (depth >= self.max_depth):
            return node, 0
        feature_idx, threshold = find_best_split(X, y)
        importance = entropy(y) * len(y)
        if feature_idx != -1:
            Xleft, Xright, yleft, yright = split_dataset(X, y, threshold=threshold, feature_idx=feature_idx)
            node.feature_idx = feature_idx
            node.threshold = threshold
            node.left, left_import = self._grow_tree(Xleft, yleft, depth + 1)
            node.right, right_import = self._grow_tree(Xright, yright, depth + 1)
            importance -= (left_import + right_import)
        self.feature_importance[feature_idx] += importance
        return node, importance

    def predict(self, X):
        ypred = []
        for datapoint in X:
            node = self.root
            while node.left is not None and node.right is not None:
                if node.feature_idx == -1:
                    break
                node = node.left if datapoint[node.feature_idx] <= node.threshold else node.right
            ypred.append(node.predicted)
        return ypred

    def _predict_tree(self, tree_node):
        if tree_node is None:
            return
        print(tree_node.feature_idx, tree_node.threshold)
        self._predict_tree(tree_node.left)
        self._predict_tree(tree_node.right)


# Split dataset based on a feature and threshold
def split_dataset(X, y, threshold, feature_idx=None):
    splitpoint = X[:, feature_idx] <= threshold if feature_idx is not None else X <= threshold
    Xleft, yleft = X[splitpoint], y[splitpoint]
    Xright, yright = X[~splitpoint], y[~splitpoint]
    return Xleft, Xright, yleft, yright


# Find the best split for the dataset
def find_best_split(X, y):
    feature_idx = -1
    threshold = None
    min_impure = 1
    for i in range(X.shape[1]):
        x = np.array(X[:, i])
        for j in np.unique(x):
            _, _, yleft, yright = split_dataset(x, y, threshold=j)
            impure_avg = (entropy(yleft) * (len(yleft) / len(y))) + (entropy(yright) * (len(yright) / len(y)))
            if impure_avg < min_impure:
                feature_idx = i
                threshold = j
                min_impure = impure_avg
    return feature_idx, threshold


def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -sum(p * np.log2(p) for p in probs)


def gini(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - sum(p * p for p in probs)
