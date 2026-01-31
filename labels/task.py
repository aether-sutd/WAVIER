from inputs.task import Task
from labels.core import FeatureLabeller

class TaskLabeller(FeatureLabeller[float]):
    def __init__(self):
        super().__init__("cls")

    def extract_label(self, task: Task) -> float:
        return task.task_id
