from pathlib import Path
import json
from functools import cache
import numpy as np

from inputs.gaze import Gaze2D
from inputs.pupil import Pupil


TLX_LABELS = ["mental", "physical", "temporal", "performance", "effort", "frustration"]

class Task:
    def __init__(self,
                 participant_id: int,
                 task_id: int,
                 data_dir = None,
                 screen_deg_x: float = 30,
                 screen_deg_y: float = 17):
        self.participant_id = participant_id
        self.task_id = task_id
        
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "colet"
        else:
            data_dir = Path(data_dir)
        
        self.path = data_dir / f"participant{participant_id}" / f"task{task_id}"
        
        assert not str(self.path).endswith(".zip"), f"Path {self.path} is a zip file and should be ignored."
        assert self.path.exists(), f"Path {self.path} does not exist."
        assert (self.path / "gaze.csv").exists(), f"File {self.path / 'gaze.csv'} does not exist."
        assert (self.path / "pupil.csv").exists(), f"File {self.path / 'pupil.csv'} does not exist."
        assert (self.path / "annotation.json").exists(), f"File {self.path / 'annotation.json'} does not exist."
        
        self.gaze = Gaze2D.from_pupil_capture(str(self.path / "gaze.csv"))
        self.pupil = Pupil.from_pupil_capture(str(self.path / "pupil.csv"))
        self.tlx = self._get_tlx()
        
        self.tlx_lst = [self.tlx[label] for label in TLX_LABELS]
        self.mean_tlx = sum(self.tlx_lst) / len(self.tlx_lst)
        
        self.label = 0 if self.mean_tlx < 30 else (1 if self.mean_tlx < 50 else 2)
        
        self.screen_deg_x = screen_deg_x
        self.screen_deg_y = screen_deg_y
    
    def _get_tlx(self) -> dict[str, float]:
        tlx_path = str(self.path / "annotation.json")
        with open(tlx_path, "r") as f:
            tlx = json.load(f)
        tlx = {label: tlx[label] for label in TLX_LABELS}
        return tlx
    
    @cache
    def _temporal_gaze(self,
                       use_nslr: bool = False,
                       velocity_threshold: float = 35,
                       min_fixation_duration: float = 0.055):
        if use_nslr:
            labels, durations, velocities, total_time = self.gaze.nslr_hmm(
                screen_deg_x=self.screen_deg_x,
                screen_deg_y=self.screen_deg_y
            )
        else:
            labels, durations, velocities, total_time = self.gaze.ivt(
                screen_deg_x=self.screen_deg_x,
                screen_deg_y=self.screen_deg_y,
                velocity_threshold=velocity_threshold,
                min_fixation_duration=min_fixation_duration
            )
        
        return labels, durations, velocities, total_time
    
    @cache
    def _temporal_ripa(self,
                       window_size: int = 121,
                       step_size: int = 1,
                       pad: bool = False) -> np.ndarray:
        ripa = self.pupil.ripa(
            window_size=window_size,
            step_size=step_size
        )
        
        if pad:
            ripa = np.concat([
                np.full(window_size // 2, ripa[0]), 
                ripa, 
                np.full(window_size // 2, ripa[-1])
            ])
        
        return ripa


def get_tasks(verbose: bool = True) -> list[Task]:
    tasks: list[Task] = []
    for participant in range(47):
        for task in range(4):
            if verbose:
                print(f"Loading participant {participant}, task {task}")
            t = Task(participant, task)
            tasks.append(t)
    return tasks