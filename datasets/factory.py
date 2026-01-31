from typing import Literal
from datasets.core import ExperimentDataset
from datasets.tlx import TLXClassDataset
from datasets.task import TaskClassDataset


class ExperimentDatasetFactory:
    def load(self,
             task: Literal['task', 'tlx'],
             objective: Literal['reg', 'cls']):
        if task == "task":
            if objective == 'reg':
                raise ValueError(f"task 'task' has no regression support")
            return TaskClassDataset
        elif task == "tlx":
            return TLXClassDataset if objective == "cls" else ExperimentDataset
        else:
            raise ValueError(f"task {repr(task)} not in ['task', 'tlx']")
