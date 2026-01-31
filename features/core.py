from abc import ABC, abstractmethod
from itertools import chain, combinations
from typing import Any
import pandas as pd
from inputs.task import Task

class FeatureGroup(ABC):
    """
    Class to group features together.
    This can be used to organize features into logical groups.
    """
    
    def __init__(self, name: str, features: list[str], temporal: bool = False):
        self.name = name
        self.features = features

    def __repr__(self):
        return f"FeatureGroup(name={self.name}, features={self.features})"
    
    @abstractmethod
    def extract(self, task: Task, **kwargs) -> dict:
        """
        Extract features from the given task.

        :param task: Input Task data
        :return: Extracted features as a dictionary.
        """
        pass

class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    All feature extractors should inherit from this class and implement the `extract` method.
    """
    
    def __init__(self, groups: list[FeatureGroup] = []):
        self.groups = {
            group.name: group for group in groups
        }
        self.archive: dict[tuple[int, int], dict[str, Any]] = {}  # Cache for extracted features'
    
    @property
    def available_groups(self) -> list[str]:
        return [i for i in self.groups]
    
    def __call__(self, task: Task, groups: list[str] | None = None, **kwargs) -> dict[str, Any]:
        """
        Allows the feature extractor to be called like a function.
        """
        
        if groups is None:
            groups = self.available_groups
        
        # check if the group list is valid, i.e. corresponds to self.groups.name
        
        if not all(group in self.groups.keys() for group in groups):
            raise ValueError("Invalid group names provided. Please check the available groups.")
        
        archive = self.archive.get((task.participant_id, task.task_id), None)
        
        if archive is None:
            archive = self.archive[(task.participant_id, task.task_id)] = {}
        
        data = {}
        
        for group in groups:
            if group in archive:
                data.update(archive[group])
            else:
                group_data = self.groups[group].extract(task, **kwargs)
                data.update(group_data)
                self.archive[(task.participant_id, task.task_id)][group] = group_data
        
        return data

    def get_feature_set(self, groups: list[str] | None = None, min_groups: int = 1, use_columns: bool = True) -> list[tuple[list[str], str]]:
        if groups is None:
            groups = self.available_groups
        
        configs = []
        
        for i in range(min_groups, len(groups)+1):
            for subgroups in combinations(groups, i):
                label = "+".join(subgroups)
                
                if use_columns:
                    columns = self.get_columns(list(subgroups))
                    configs.append((columns, label))
                else:
                    configs.append((list(subgroups), label))
        
        return configs

    def get_columns(self, groups: list[str] | None = None) -> list[str]:
        """
        Get all columns for the given groups.
        If no groups are provided, returns all columns from all groups.
        """
        if groups is None:
            groups = self.available_groups
        
        columns = list(chain(*[
            self.groups[label].features for label in groups
        ]))
        
        return columns

    def load_ds(self, tasks: list[Task]) -> pd.DataFrame:
        X = []
        for t in tasks:
            X.append(self(t))
        
        return pd.DataFrame(X)
