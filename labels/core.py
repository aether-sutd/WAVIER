from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Literal
import numpy as np
import numpy.typing as npt
from inputs.task import Task


T = TypeVar("T")

class FeatureLabeller(ABC, Generic[T]):
    def __init__(self, objective: Literal["reg", "cls"]):
        self.objective = objective
    
    @abstractmethod
    def extract_label(self, task: Task) -> T:
        pass
    
    def __call__(self, task: Task) -> T:
        return self.extract_label(task)

    def load_ds(self, tasks: list[Task]) -> npt.NDArray:
        y = []
        for t in tasks:
            y.append(self(t))
        
        return np.array(y)
