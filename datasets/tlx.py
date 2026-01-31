from datasets.core import ExperimentDataset, ExperimentData

import numpy as np
class TLXClassDataset(ExperimentDataset):
    configs = ["C1vC2", "C1vC3", "C2vC3", "C1C2vC3", "C1vC2C3", "all"]

    def get_training_config(self, X, y, config: str):
        samples_per_subj = np.full(47, 4)

        # C1 vs C2
        if config == "C1vC2":
            reshaped = (y != 2).reshape(-1, 4)
            samples_per_subj =  np.sum(reshaped, axis=1)
            
            X = X[y != 2]
            y = y[y != 2]
        
            return X, y, samples_per_subj
        
        # C1 vs C3
        if config == "C1vC3":
            reshaped = (y != 1).reshape(-1, 4)
            samples_per_subj =  np.sum(reshaped, axis=1)
            
            X = X[y != 1]
            y = (y[y != 1] > 0).astype(int)
            
            return X, y, samples_per_subj
        # C2 vs C3
        if config == "C2vC3":
            reshaped = (y != 0).reshape(-1, 4)
            samples_per_subj =  np.sum(reshaped, axis=1)
            
            X = X[y != 0]
            y = y[y != 0] - 1
            
            return X, y, samples_per_subj
        # C1 vs C2 vs C3
        if config == "all":
            return X, y, samples_per_subj
        # C1, C2 vs C3
        if config == "C1C2vC3":
            y = (y == 2).astype(int)
            return X, y, samples_per_subj
        # C1 vs C2, C3
        if config == "C1vC2C3":
            y = (y != 0).astype(int)
            return X, y, samples_per_subj

        # all
        return X, y, samples_per_subj
    
    def __iter__(self):
        for columns, label in self.extractor.get_feature_set():
            if self.use_nslr and "gaze" not in label:
                continue
            
            print("analying", label)
            
            X = self.features[columns].to_numpy()
            y = self.targets
            
            for cfg in self.configs:
                print("config", cfg)
                X_conf, y_conf, samples_per_subj = self.get_training_config(X, y, cfg)

                if len(np.unique(y_conf)) < 2:
                    print("config", cfg, "only has unique", np.unique(y_conf), "not valid")
                    continue
                
                yield ExperimentData(
                    X=X_conf, y=y_conf,
                    columns=columns,
                    label=label,
                    config=cfg,
                    samples_per_subj=samples_per_subj
                )

