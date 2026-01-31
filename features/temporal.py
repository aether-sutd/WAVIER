from typing import Literal
from inputs.task import Task
from features.core import (
    FeatureGroup,
    FeatureExtractor
)


class TemporalGazeFeatureGroup(FeatureGroup):
    def __init__(self, method: Literal["ivt", "nslr"] = "ivt",
                velocity_threshold: float = 35,
                min_fixation_duration: float = 0.055):
        super().__init__("gaze", [
            "durations",
            "velocities",
            "labels"
        ], temporal=False)
        self.method = method
        self.velocity_threshold = velocity_threshold
        self.min_fixation_duration = min_fixation_duration
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        labels, durations, velocities, total_time = task._temporal_gaze(
            use_nslr=(self.method == "nslr"),
            velocity_threshold=self.velocity_threshold,
            min_fixation_duration=self.min_fixation_duration
        )
        
        data["durations"] = durations.tolist()
        data["velocities"] = velocities.tolist()
        data["labels"] = labels.tolist()
        
        return data

class TemporalPupilFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("pupil", [
            "lpd", "rpd"
        ], temporal=True)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        data["lpd"] = task.pupil.lpd.tolist()
        data["rpd"] = task.pupil.rpd.tolist()
        
        return data

class TemporalRIPAFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("ripa", [
            "ripa"
        ], temporal=True)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        ripa = task._temporal_ripa(window_size=121, step_size=1, pad=True)
        
        data["ripa"] = ripa.tolist()
        
        return data

class TemporalGazeFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 method: Literal["ivt", "nslr"] = "ivt",
                 velocity_threshold: float = 35,
                 min_fixation_duration: float = 0.055):
        super().__init__(groups=[
            TemporalGazeFeatureGroup(
                method=method,
                velocity_threshold=velocity_threshold,
                min_fixation_duration=min_fixation_duration
            )
        ])

class TemporalPupilFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(groups=[
            TemporalPupilFeatureGroup(),
            TemporalRIPAFeatureGroup()
        ])
