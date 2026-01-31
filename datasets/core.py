from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from inputs.task import Task
from features import FeatureExtractor
from labels import FeatureLabeller


@dataclass
class ExperimentData:
    X: npt.NDArray
    y: npt.NDArray
    columns: list[str]
    label: str
    config: str
    samples_per_subj: npt.NDArray = None
    


class ExperimentDataset:
    configs = [""]

    def __init__(self,
                 tasks: list[Task],
                 extractor: FeatureExtractor,
                 labeller: FeatureLabeller,
                 use_nslr: bool = False):
        self.tasks = tasks
        self.extractor = extractor
        self.labeller = labeller
        self.use_nslr = use_nslr
        
        self.objective = labeller.objective
        self.features = extractor.load_ds(tasks)
        self.targets = labeller.load_ds(tasks)
    
    def get_training_config(self, X, y, config: str):
        return X, y, np.full(47, 4)
    
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
