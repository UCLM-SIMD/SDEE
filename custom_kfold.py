import numpy as np


class CustomKFold:
    def __init__(self, n_splits=10, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        for i in range(self.n_splits):
            # an iteration ith in every ten iterations is included in fold ith
            test_indices = indices[i: len(X): self.n_splits]
            train_indices = [idx for idx in indices if idx not in test_indices]
            yield train_indices, test_indices
