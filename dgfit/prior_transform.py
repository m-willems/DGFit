import numpy as np
from scipy.stats import rv_continuous

class LogUniformLin(rv_continuous):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

        super().__init__(
            a=np.log10(lower),
            b=np.log10(upper),
            name="linear_uniform_log10"
        )

    def _ppf(self, q):
        x = self.lower + q * (self.upper - self.lower)
        return x

    def _pdf(self, y):
        return np.full_like(
            y, 1.0 / (self.upper - self.lower)
        )

    def _cdf(self, z):
        return (z - self.lower) / (self.upper - self.lower)