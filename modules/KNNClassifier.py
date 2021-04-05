import numpy as np
import pandas as pd
from sklearn import model_selection, metrics, preprocessing


def euclidean_metric(x, y, _=None):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_metric(x, y, _=None):
    return np.sum(np.abs(x - y))


def minkowski_metric(x, y, q=3):
    return np.power(np.sum(np.power(np.abs(x - y), q)), 1 / q)


class KNNClassifier:
    """Классификатор, использующий метод к ближайших соседей

    Parameters
    ----------
    n_neighbors : int, default = 5
        Количество соседей, используемых для классификации нового объекта

    metric : {'euclidean', 'manhattan', 'minkowski'}, default = euclidean
        Метрика, используемая для подсчёта расстояний между объектами

    q : int, default = 3
        Параметр, используемый для возведения в степень в метрике Минковского

    """

    METRICS = {
        'euclidean': euclidean_metric,
        'manhattan': manhattan_metric,
        'minkowski': minkowski_metric
    }

    def __init__(self, n_neighbors=5, metric='euclidean', q=3):
        self.n_neighbors = n_neighbors
        self.metric = self.METRICS[metric]
        self.q = q
        self.samples_x = None
        self.classes = None
        self.samples_y = None

    def fit(self, x, y):
        self.samples_x = x
        self.samples_y = y
        self.classes = set(y)

    def predict(self, x):
        y_pred = []
        for test_sample in x:
            weights = []
            distances = []
            for index, train_sample in enumerate(self.samples_x):
                distances.append(self.metric(test_sample, train_sample, self.q))
                weights.append(1/distances[index])
            indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            neighbors = self.samples_y[indices]
            weights = np.array(weights)[indices]
            y_pred.append(np.argmax(np.bincount(neighbors, weights, len(self.classes))))
        return y_pred
