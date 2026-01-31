from pathlib import Path
import pandas as pd
import numpy as np
from functools import cache

from inputs.pupil import Pupil


class EseedTask:
    """
    Task class for ESEED dataset that mimics the interface of the CoLET Task class
    but works with ESEED's folder structure and data format.
    """
    
    def __init__(self, 
                 participant_id: int, 
                 video_id: int,  # Note: using video_id instead of task_id
                 data_dir=None):
        self.participant_id = participant_id
        self.video_id = video_id  # ESEED uses video_id instead of task_id
        self.task_id = video_id  # Add task_id alias for compatibility with extractors
        
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "eseed"
        else:
            data_dir = Path(data_dir)
        
        # ESEED folder structure: participant_01/video_01/
        self.path = data_dir / f"participant_{participant_id:02d}" / f"video_{video_id:02d}"
        
        assert self.path.exists(), f"Path {self.path} does not exist."
        assert (self.path / "pupil.csv").exists(), f"File {self.path / 'pupil.csv'} does not exist."
        
        # Load pupil data using custom ESEED loader
        self.pupil = self._load_eseed_pupil()
        
        # No TLX labels for ESEED - set defaults
        self.tlx = {}
        self.tlx_lst = []
        self.mean_tlx = 0
        self.label = 0  # Default label since ESEED has no labels
    
    def _load_eseed_pupil(self) -> Pupil:
        """
        Load pupil data from ESEED CSV and convert to Pupil object format.
        Uses the same logic as CoLET's Pupil.from_pupil_capture method.
        """
        pupil_csv_path = self.path / "pupil.csv"
        df = pd.read_csv(pupil_csv_path)
        
        # Follow the same logic as CoLET's from_pupil_capture method
        use_3d = True  # Default to 3D like CoLET
        
        # Filter by eye_id and method (same as CoLET)
        right_eye = df[df.eye_id == 0]
        right_eye = right_eye[right_eye["method"] == ("3d c++" if use_3d else "2d c++")]
        left_eye = df[df.eye_id == 1]
        left_eye = left_eye[left_eye["method"] == ("3d c++" if use_3d else "2d c++")]
        
        # Take minimum samples (same as CoLET)
        samples = min(len(right_eye), len(left_eye))
        right_eye = right_eye.iloc[:samples]
        left_eye = left_eye.iloc[:samples]
        
        # Average timestamps (same as CoLET)
        timestamp = (right_eye["pupil_timestamp"].to_numpy() + left_eye["pupil_timestamp"].to_numpy()) / 2
        
        # Use correct diameter column (same as CoLET)
        column = "diameter" + ("_3d" if use_3d else "")
        
        # Create Pupil object with correct eye assignment (same as CoLET: left=1, right=0)
        return Pupil(t=timestamp, lpd=left_eye[column].to_numpy(), rpd=right_eye[column].to_numpy())
    
    @cache
    def _temporal_ripa(self,
                       window_size: int = 121,
                       step_size: int = 1,
                       pad: bool = False) -> np.ndarray:
        """
        Compute RIPA features for compatibility with TemporalPupilFeatureExtractor.
        """
        ripa = self.pupil.ripa(
            window_size=window_size,
            step_size=step_size
        )
        
        if pad:
            ripa = np.concatenate([
                np.full(window_size // 2, ripa[0]), 
                ripa, 
                np.full(window_size // 2, ripa[-1])
            ])
        
        return ripa


def get_eseed_tasks(max_participants=48, max_videos=10, verbose: bool = True) -> list[EseedTask]:
    """
    Load all available ESEED tasks, similar to get_tasks() for CoLET.
    """
    tasks = []
    for participant_id in range(1, max_participants + 1):
        for video_id in range(1, max_videos + 1):
            try:
                if verbose:
                    print(f"Loading participant {participant_id}, video {video_id}")
                task = EseedTask(participant_id, video_id)
                tasks.append(task)
            except (FileNotFoundError, AssertionError):
                # Skip if files don't exist
                continue
            except Exception as e:
                if verbose:
                    print(f"Error loading participant {participant_id}, video {video_id}: {e}")
                continue
    
    if verbose:
        print(f"Successfully loaded {len(tasks)} ESEED tasks")
    return tasks