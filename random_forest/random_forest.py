from decision_tree.id3_decision_tree import DecisionTreeID3
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees = 10, max_depth = 10, min_sample_split = 2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_feature
        self.trees = []


    def fit(self, X, y):

        X = X.values if hasattr(X, "values") else np.array(X)
        y = y.values if hasattr(y, "values") else np.array(y)

        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTreeID3(max_depth=self.max_depth, min_samples_split=self.min_sample_split, n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]



    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
