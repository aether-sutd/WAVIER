from datasets.core import ExperimentDataset
import numpy as np

class TaskClassDataset(ExperimentDataset):
    configs = ["A1vA2", "A1vA3", "A1vA4", "A2vA3", "A2vA4", "A3vA4", "A1A2vA3A4", "all"]

    def get_training_config(self, X, y, config: str):
        # A1 vs A2
        if config == "A1vA2":
            X = X[y < 2]
            y = y[y < 2]
            return X, y, np.full(47, 2)
        # A1 vs A3
        if config == "A1vA3":
            X = X[(y != 1) & (y != 3)]
            y = (y[((y != 1) & (y != 3))] == 2).astype(int)
            return X, y, np.full(47, 2)
        # A1 vs A4
        if config == "A1vA4":
            X = X[(y != 1) & (y != 2)]
            y = (y[(y != 1) & (y != 2)] == 3).astype(int)
            return X, y, np.full(47, 2)
        # A2 vs A3
        if config == "A2vA3":
            X = X[(y != 0) & (y != 3)]
            y = y[(y != 0) & (y != 3)]-1
            return X, y, np.full(47, 2)
        # A2 vs A4
        if config == "A2vA4":
            X = X[(y != 0) & (y != 2)]
            y = (y[(y != 0) & (y != 2)] == 3).astype(int)
            return X, y, np.full(47, 2)
        # A3 vs A4
        if config == "A3vA4":
            X = X[(y != 0) & (y != 1)]
            y = y[(y != 0) & (y != 1)]-2
            return X, y, np.full(47, 2)
        
        # A1A2 vs A3A4
        if config == "A1A2vA3A4":
            y = (y >= 2).astype(int)
            return X, y, np.full(47, 4)
        
        # all
        return X, y, np.full(47, 4)
