import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, category=None, left=None, right=None, *, value=None):
        self.feature  = feature
        self.category = category
        self.left     = left
        self.right    = right
        self.value    = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeID3:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None): 
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth
        self.n_features        = n_features
        self.root              = None

    def fit(self, X, y):
        # garante ndarray e 2D
        X = X.values if hasattr(X, 'values') else np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # y pode ser texto; mantemos como array
        y = y.values if hasattr(y, 'values') else np.array(y)

        # define quantas features usar
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        # inicia construção recursiva
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Critérios de parada:
        if (n_samples < self.min_samples_split or depth >= self.max_depth or n_labels == 1):
            # nó folha com classe majoritária
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # amostra atributos aleatoriamente (para random forest ou choque computacional)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # encontra melhor atributo e categoria para dividir
        best_feat, best_cat, best_gain = None, None, -np.inf
        for feat in feat_idxs:
            col = X[:, feat]
            for cat in np.unique(col):
                gain = self._information_gain(col, y, cat)
                # ignora splits sem ganho ou que dividam vazio
                if gain > best_gain:
                    best_gain, best_feat, best_cat = gain, feat, cat

        # se nenhum ganho positivo, virar folha
        if best_gain <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # partição final nos índices
        left_idxs, right_idxs = self._split(X[:, best_feat], best_cat)
        # constrói subárvores
        left  = self._grow_tree(X[left_idxs, :],  y[left_idxs],  depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(feature=best_feat, category=best_cat, left=left, right=right)

    def _information_gain(self, X_col, y, category):
        # entropia antes
        parent_entropy = self._entropy(y)
        # particiona
        left_idxs, right_idxs = self._split(X_col, category)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        # entropia dos filhos
        n = len(y)
        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])
        child_entropy = (len(left_idxs)/n)*e_l + (len(right_idxs)/n)*e_r
        return parent_entropy - child_entropy
    
    def _most_common_label(self, y):
        # Retorna a classe mais frequente em y
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _entropy(self, y):
        # calcula entropia usando Counter (aceita strings)
        counter = Counter(y)
        n = len(y)
        ps = [cnt/n for cnt in counter.values()]
        return -sum(p * np.log(p) for p in ps if p>0)

    def _split(self, X_col, category):
        # split categórico: igual vs diferente
        left_idxs  = np.argwhere(X_col == category).flatten()
        right_idxs = np.argwhere(X_col != category).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        X = X.values if hasattr(X, 'values') else np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        # segue ramo de igualdade ou não
        if x[node.feature] == node.category:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
