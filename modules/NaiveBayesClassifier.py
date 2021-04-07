import numpy as np


def gaussian_probability(x, mu, sigma_squared):
    exponent = np.exp(-((x - mu) ** 2 / (2 * sigma_squared)))
    return (1 / (np.sqrt(2 * np.pi * sigma_squared))) * exponent


class NBClassifier:
    """Наивный байесовский классификатор



    """
    def __init__(self):
        self.probabilities = None
        self.means = None
        self.stds = None
        self.classes = None
        self.samples = None

    def fit(self, x, y):
        self.classes = np.unique(y)
        self.samples = [list() for _ in self.classes]
        for index, cls in enumerate(y):
            self.samples[cls].append(x[index])
        self.means = [np.mean(sample, axis=0) for sample in self.samples]
        self.stds = [np.std(sample, axis=0) for sample in self.samples]
        self.probabilities = [len(sample)/len(x) for sample in self.samples]

    def predict(self, x):
        y_pred = []
        for i in range(len(x)):
            p = {}

            for cls in self.classes:
                p[cls] = self.probabilities[cls]

                for index, param in enumerate(x[i]):
                    p[cls] *= gaussian_probability(param, self.means[cls][index], self.stds[cls][index])
            y_pred.append(np.argmax([i for i in p.values()]))
        return y_pred
