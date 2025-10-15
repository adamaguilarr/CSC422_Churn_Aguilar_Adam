from __future__ import annotations
import numpy as np

class MajorityClassBaseline:
    def fit(self, X, y):
        # store majority class
        values, counts = np.unique(y, return_counts=True)
        self.majority_ = values[np.argmax(counts)]
        return self
    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.majority_, dtype=int)
