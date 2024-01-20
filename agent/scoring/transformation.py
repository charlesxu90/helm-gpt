# Adapted from https://github.com/MolecularAI/Reinvent/blob/reinvent.v.2.0/scoring/score_transformations.py
import numpy as np


class TransformFunction:
    def __init__(self, trans_type: str, low, high, params: dict = None):
        """
            trans_type: type of transformation function, including: 'sigmoid', 'rsigmoid'(reverse sigmoid), 'dsigmoid' (double
                    sigmoid)
            low: definition of low score
            high: definition of high score
            params: dict of params for transformation: sigmoid/rsigmoid: k; dsigmoid: k1, k2. Larger k, quicker change.
        """
        self.transformation_list = {'sigmoid': self.sigmoid_trans, 'rsigmoid': self.reverse_sigmoid_trans,
                                    'dsigmoid': self.double_sigmoid}
        try:
            self.trans_fn = self.transformation_list[trans_type]
        except:
            raise Exception(f"transformation type {trans_type} not found! Options: 'sigmoid', 'rsigmoid' and 'dsigmoid'.")
        self.low, self.high = low, high
        self.params = params

    def __call__(self, scores: np.array):
        """
            scores: np.array of scores
        """
        return self.trans_fn(scores)

    def sigmoid_trans(self, scores: np.array) -> np.array:
        _k = self.params['k'] if self.params is not None and 'k' in self.params.keys() else 2.

        def _sigmoid_fn(score, low, high, k) -> float:
            return 1 / (1 + 10 ** (- k * (score - (high + low) / 2) / (high - low)))

        transformed = [_sigmoid_fn(score, self.low, self.high, _k) for score in scores]
        return np.array(transformed, dtype=np.float32)

    def reverse_sigmoid_trans(self, scores: np.array) -> np.array:
        _k = self.params['k'] if self.params is not None and 'k' in self.params.keys() else 2.

        def _reverse_sigmoid_fn(score, low, high, k) -> float:
            return 1 / (1 + 10 ** (k * (score - (high + low) / 2) / (high - low)))

        transformed = [_reverse_sigmoid_fn(score, self.low, self.high, _k) for score in scores]
        return np.array(transformed, dtype=np.float32)

    def double_sigmoid(self, scores: np.array) -> np.array:
        _k1, _k2 = 1.5, 1.5  # large k, quicker change
        if self.params is not None and all(key in self.params for key in ('k1', 'k2')):
            _k1, _k2 = self.params['k1'], self.params['k2']

        def _double_sigmoid_fn(score, low, high, k1, k2):
            return 1/(1 + 10 ** (- k1 * (score - low))) - 1/(1 + 10 ** (k2 * (high - score)))

        transformed = [_double_sigmoid_fn(score, self.low, self.high, _k1, _k2) for score in scores]
        return np.array(transformed, dtype=np.float32)


def unittest():
    scores = np.arange(-2, 2.25, 0.25)
    print(scores)

    trans_fn = TransformFunction('sigmoid', -2, 2, params={'k': 1.})
    scores_tfd = trans_fn(scores)
    print(scores_tfd)

    trans_fn = TransformFunction('sigmoid', -2, 2, params={'k': 2.})
    scores_tfd = trans_fn(scores)
    print(scores_tfd)

    trans_fn = TransformFunction('rsigmoid', 2, -2, params={'k': 2.5})
    scores_tfd = trans_fn(scores)
    print(scores_tfd)

    trans_fn = TransformFunction('dsigmoid', -1, 1, params={'k1': 1.5, 'k2': 1.5})
    scores_tfd = trans_fn(scores)
    print(scores_tfd)


if __name__ == '__main__':
    unittest()
