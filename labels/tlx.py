from typing import Literal
from inputs.task import Task
from labels.core import FeatureLabeller

class TLXLabeller(FeatureLabeller[float]):
    def __init__(self, objective: Literal["reg", "cls"],
                 property: Literal[
                     "mean",
                     "mental", "physical", "temporal",
                     "performance", "effort", "frustration"
                 ] = "mean"):
        super().__init__(objective)
        self.objective = objective
        self.property = property
    
    def _convert_to_cls(self, value: float) -> float:
        return 0 if value < 30 else (1 if value < 50 else 2)
    
    def extract_label(self, task: Task) -> float:
        if self.property == "mean":
            result = task.mean_tlx
        else:
            result = task.tlx[self.property]
        
        if self.objective == "cls":
            return self._convert_to_cls(result)
        return result
