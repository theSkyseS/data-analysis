import numpy as np


def linear_kernel(weights, x, _=None, __=None):
    return np.dot(x, weights)


def polynomial_kernel(weights, x, degree=3, bias=0):
    return np.power(np.dot(x, weights) + bias, degree)


def radial_kernel(weights, x, gamma=1, _=None):
    return np.exp(-gamma * np.sum(np.square(weights - x)))


KERNELS = {
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'radial': radial_kernel
}


class SVMClassifier:
    """Классификатор, основанный на методе опорных векторов
    Parameters
    ==========
    eta : float, default = 0.01
        Скорость градиентного спуска
    alpha : float, default = 1.0
        Коэффициент регуляризации
    kernel : {'linear', 'polynomial', 'radial'}
        Ядро преобразования
    p : int
        Степень для полиномиального ядра и множитель для радиального
    """
    def __init__(self, eta=0.01, alpha=1.0, kernel='linear', p=3):
        self.eta = eta
        self.alpha = alpha
        self.kernel = KERNELS[kernel]
        self.weights = None
        self.class_count = 0
        self.p = p

    def stochastic_gradient_step(self, x, y):
        scores = self.kernel(self.weights, x, self.p)
        correct_score = scores[np.arange(x.shape[0]), y]
        margins = np.maximum(0, scores - correct_score[:, np.newaxis] + 1.0)
        margins[np.arange(x.shape[0]), y] = 0

        incorrect_samples = np.zeros_like(margins)
        incorrect_samples[margins > 0] = 1
        count = np.sum(incorrect_samples, axis=1)
        incorrect_samples[np.arange(x.shape[0]), y] = -count

        weight_changes = np.dot(x.T, incorrect_samples)
        weight_changes /= x.shape[0]
        weight_changes += 2 * self.alpha * self.weights

        return weight_changes

    def fit(self, x, y, min_weight_dist=1e-8, max_iter=1e4, batch_size=5):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.class_count = len(np.unique(y))
        self.weights = np.ones((x.shape[1], self.class_count))
        weight_dist = np.array([np.inf, np.inf])
        iter_num = 0
        while weight_dist.any() > min_weight_dist and iter_num < max_iter:
            random_ind = np.random.randint(x.shape[0] - batch_size)
            weight_changes = self.stochastic_gradient_step(x[random_ind:random_ind + batch_size],
                                                           y[random_ind:random_ind + batch_size])
            self.weights += -self.eta * weight_changes
            weight_dist = []
            for weight_change in weight_changes:
                weight_dist.append(np.linalg.norm(weight_change))
            weight_dist = np.array(weight_dist)
            iter_num += 1

    def predict_probabilities(self, x):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        scores = self.kernel(self.weights, x, self.p)
        return scores

    def predict(self, x):
        scores = self.predict_probabilities(x)
        return np.argmax(scores, axis=1)
