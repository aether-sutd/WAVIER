from dataclasses import dataclass
from typing import Literal
import numpy as np

from inputs.task import Task
from features.stats import safe_stats

from features.core import (
    FeatureGroup,
    FeatureExtractor
)


@dataclass
class FixationFeatures:
    """Class for keeping track of Fixation Data"""
    fixation_durations: np.ndarray
    fixation_frequency: float
    
    def get_features(self):
        fix_dur_mean, fix_dur_cv, fix_dur_skew, fix_dur_kurt = safe_stats(self.fixation_durations)
        
        return {
            'fixation_frequency': self.fixation_frequency,
            'fixation_duration_mean': fix_dur_mean,
            'fixation_duration_CV': fix_dur_cv,
            'fixation_duration_skewness': fix_dur_skew,
            'fixation_duration_kurtosis': fix_dur_kurt
        }
    
@dataclass
class SaccadeFeatures:
    """Class for keeping track of Saccade Data"""
    saccade_durations: np.ndarray
    saccade_velocities: np.ndarray
    saccade_frequency: float
    
    def get_features(self):
        sac_dur_mean, sac_dur_cv, sac_dur_skew, sac_dur_kurt = safe_stats(self.saccade_durations)
        sac_vel_mean, sac_vel_cv, sac_vel_skew, sac_vel_kurt = safe_stats(self.saccade_velocities)
        
        return {
            'saccade_frequency': self.saccade_frequency,
            'saccade_duration_mean': sac_dur_mean,
            'saccade_duration_CV': sac_dur_cv,
            'saccade_duration_skewness': sac_dur_skew,
            'saccade_duration_kurtosis': sac_dur_kurt,
            'saccade_velocity_mean': sac_vel_mean,
            'saccade_velocity_CV': sac_vel_cv,
            'saccade_velocity_skewness': sac_vel_skew,
            'saccade_velocity_kurtosis': sac_vel_kurt
        }

@dataclass
class DiameterFeatures:
    """Class for keeping track of Diameter Data"""
    diameter: np.ndarray
    name: str
    
    def get_features(self):
        d_mean, d_cv, d_skew, d_kurt = safe_stats(self.diameter)
        
        return {
            f'{self.name}_mean': d_mean,
            f'{self.name}_CV': d_cv,
            f'{self.name}_skewness': d_skew,
            f'{self.name}_kurtosis': d_kurt
        }

@dataclass
class RIPAFeatures:
    """Class for keeping track of RIPA Data"""
    ripa: np.ndarray
    
    def get_features(self):
        ripa_mean, ripa_cv, ripa_skew, ripa_kurt = safe_stats(self.ripa)
        
        return {
            f'ripa_mean': ripa_mean,
            f'ripa_CV': ripa_cv,
            f'ripa_skewness': ripa_skew,
            f'ripa_kurtosis': ripa_kurt
        }

class SummaryGazeFeatureGroup(FeatureGroup):
    def __init__(self, method: Literal["ivt", "nslr"] = "ivt",
                velocity_threshold: float = 35,
                min_fixation_duration: float = 0.055):
        super().__init__("gaze", [
            "fixation_frequency",
            "fixation_duration_mean",
            "fixation_duration_CV",
            "fixation_duration_skewness",
            "fixation_duration_kurtosis",
            "saccade_frequency",
            "saccade_duration_mean",
            "saccade_duration_CV",
            "saccade_duration_skewness",
            "saccade_duration_kurtosis",
            "saccade_velocity_mean",
            "saccade_velocity_CV",
            "saccade_velocity_skewness",
            "saccade_velocity_kurtosis"
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
        
        fixation_durations = durations[labels == 1]
        fixation_frequency = len(fixation_durations) / total_time
        fixations = FixationFeatures(
            fixation_durations=fixation_durations,
            fixation_frequency=fixation_frequency
        )
        
        data.update(fixations.get_features())
        
        saccade_durations = durations[labels == 0]
        saccade_velocities = velocities[labels == 0]
        saccade_frequency = len(saccade_durations) / total_time
        saccades = SaccadeFeatures(
            saccade_durations=saccade_durations,
            saccade_velocities=saccade_velocities,
            saccade_frequency=saccade_frequency
        )
        
        data.update(saccades.get_features())
        
        return data

class SummaryPupilFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("pupil", [
            "left_diameter_mean",
            "left_diameter_CV",
            "left_diameter_skewness",
            "left_diameter_kurtosis",
            "right_diameter_mean",
            "right_diameter_CV",
            "right_diameter_skewness",
            "right_diameter_kurtosis"
        ], temporal=False)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        lpd = DiameterFeatures(task.pupil.lpd, "left_diameter")
        rpd = DiameterFeatures(task.pupil.rpd, "right_diameter")
        data.update(lpd.get_features())
        data.update(rpd.get_features())
        
        return data

class SummaryCOLETPupilFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("colet_pupil", [
            "diameter_mean",
            "diameter_CV",
            "diameter_skewness",
            "diameter_kurtosis"
        ], temporal=False)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        d = DiameterFeatures(np.hstack((task.pupil.lpd, task.pupil.rpd)), "diameter")
        data.update(d.get_features())
        
        return data

class IPAFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("ipa", [
            "ipa"
        ], temporal=False)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        ipa = task.pupil.ipa(
            wavelet="sym16",
            periodization="per",
            level=2,
            thresh_mode="hard"
        )
        data["ipa"] = ipa
        
        return data

class LHIPAFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("lhipa", [
            "lhipa"
        ], temporal=False)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        lhipa = task.pupil.lhipa(
            wavelet="sym16",
            periodization="per",
            thresh_mode="less"
        )
        data["lhipa"] = lhipa
        
        return data

class SummaryRIPAFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__("ripa", [
            "ripa_mean",
            "ripa_CV",
            "ripa_skewness",
            "ripa_kurtosis"
        ], temporal=False)
    
    def extract(self, task: Task, **kwargs) -> dict:
        data = {}
        
        ripa = task._temporal_ripa(window_size=120, step_size=60, pad=False)
        ripa_features = RIPAFeatures(ripa)
        data.update(ripa_features.get_features())
        
        return data

class SummaryFeatureExtractor(FeatureExtractor):
    def __init__(self,
                use_nslr: bool = False,
                velocity_threshold: float = 35,
                min_fixation_duration: float = 0.055):
        super().__init__(groups=[
            SummaryGazeFeatureGroup(
                method="nslr" if use_nslr else "ivt",
                velocity_threshold=velocity_threshold,
                min_fixation_duration=min_fixation_duration
            ),
            SummaryPupilFeatureGroup(),
            SummaryCOLETPupilFeatureGroup(),
            IPAFeatureGroup(),
            LHIPAFeatureGroup(),
            SummaryRIPAFeatureGroup()
        ])
