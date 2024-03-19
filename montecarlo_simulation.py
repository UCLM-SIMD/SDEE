from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class MCSimulation(BaseEstimator, RegressorMixin):
    def __init__(self, percentile=95, simulations=1000):
        self.percentile = percentile
        self.simulations = simulations
        self.velocities_distribution = None

    def fit(self, X, y=None):
        """
        Given a dataset X of features of tasks, and a target variable y, that is the difference between the velocity
        planned and the velocity executed, we need to simulate 10000 iterations (each row is an existing iteration),
        by picking random velocities. Then we return the percentil 95 of the simulated velocities.
        """
        velocities_distribution = []
        for sim in range(self.simulations):
            # pick random velocities from y:
            velocities = np.random.choice(y, X.shape[0])
            velocities_distribution.extend(velocities)
        self.velocities_distribution = np.array(velocities_distribution)

    def predict(self, X):
        """
        Given a dataset X of features of tasks, we return the percentil 95 of the simulated velocities.
        """
        percentile = np.percentile(self.velocities_distribution, self.percentile)
        return [percentile for _ in range(X.shape[0])]
