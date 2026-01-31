from itertools import groupby
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.signal import medfilt

from inputs.nslr import nslr_fit_gaze, nslr_classify_segments

try:
    from nslr import fit_gaze as nslr_fit_gaze # type: ignore
    print("Using fast nslr")
except ImportError:
    print("Using slow nslr")


class Gaze2D:
    def __init__(self, t: npt.ArrayLike, 
                 x: npt.ArrayLike, 
                 y: npt.ArrayLike):
        self._gaze2d = pd.DataFrame({
            "t": t,
            "x": x,
            "y": y
        })

    # ======================= GETTERS =======================

    @property
    def t(self) -> npt.NDArray:
        return self._gaze2d["t"].to_numpy()
    
    @property
    def x(self) -> npt.NDArray:
        return self._gaze2d["x"].to_numpy()
    
    @property
    def y(self) -> npt.NDArray:
        return self._gaze2d["y"].to_numpy()
    
    @property
    def n_values(self) -> int:
        return len(self._gaze2d)

    def __repr__(self):
        return self._gaze2d.__repr__()
    
    def __str__(self):
        return self._gaze2d.__str__()

    # ======================= LOADERS =======================

    @classmethod
    def from_tobii_glasses(cls, txt_file_path: str):
        df = pd.read_json(txt_file_path, lines=True)
        
        # cull moments where tracker was not worn
        idx = (list(df.data != {})).index(True)
        idx_last = len(df) - (list(df.data != {}))[::-1].index(True)
        df = df.iloc[idx:idx_last]
        
        # get rid of "type" column, and set index to "timestamp"
        # pop out all the fields
        df = df[["timestamp", "data"]].set_index("timestamp")["data"].apply(pd.Series)
        
        def process_2d_sample(sample: list[float]) -> pd.Series:
            # precision note: round to 4th decimal point
            return pd.Series({"x": round(sample[0], 4), "y": round(sample[1], 4)})
        
        gaze2d = df["gaze2d"].dropna().map(list).apply(process_2d_sample)
        
        return cls(gaze2d.index.to_numpy(), gaze2d["x"].to_numpy(), gaze2d["y"].to_numpy())
    
    @classmethod
    def from_pupil_capture(cls, csv_file_path: str):
        gaze = pd.read_csv(csv_file_path)
        return cls(gaze["gaze_timestamp"].to_numpy(), gaze["norm_pos_x"].to_numpy(), gaze["norm_pos_y"].to_numpy())

    # ======================= FEATURES ======================
 
    def smooth(self, method: Literal["nslr"] = "nslr"):
        ts = self.t
        signal = self._gaze2d[["x", "y"]].values
        
        if method == "nslr":
            reconstruction = nslr_fit_gaze(ts, signal)
            smooth_signal = reconstruction(ts)
            
            return Gaze2D(ts, smooth_signal[:, 0], smooth_signal[:, 1])

        raise NotImplementedError(f"Method provided {repr(method)} is not implemented")
    
    def ivt(self, screen_deg_x: float = 30, screen_deg_y: float = 17,
            velocity_threshold: float = 35, min_fixation_duration: float = 0.055):
        """
        Calculate the Inter-Visitation Time (IVT) of a gaze trajectory.
        """
        # Convert to degrees of visual angle
        deg_x = np.array(self.x) * screen_deg_x
        deg_y = np.array(self.y) * screen_deg_y
        
        # Calculate the time differences between consecutive samples
        dt = np.diff(self.t)
        
        # Calculate the distances between consecutive samples
        dx = np.diff(deg_x)[dt != 0]
        dy = np.diff(deg_y)[dt != 0]
        dt = dt[dt != 0]
        
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / dt

        # Median filter to smooth velocities
        velocity_filtered = medfilt(velocity, kernel_size=5)

        # I-VT classification
        fixation_mask = velocity_filtered < velocity_threshold
        
        # Feature lists
        # 0 = saccade, 1 = fixation
        labels = []
        durations = []
        velocities = []

        start_idx = 0
        
        for key, group in groupby(fixation_mask):
            group_len = len(list(group))
            end_idx = start_idx + group_len
            duration = np.sum(dt[start_idx:end_idx])

            if key:  # Fixation
                if duration >= min_fixation_duration:
                    labels.append(1)
                    durations.append(duration)
                    velocities.append(0)
            else:    # Saccade
                if duration > 0:
                    labels.append(0)
                    durations.append(duration)
                    velocities.append(np.mean(velocity[start_idx:end_idx]))
                
            start_idx = end_idx

        # Total duration
        total_time = self.t[-1] - self.t[0]
        
        return np.array(labels), np.array(durations), np.array(velocities), total_time
    
    def nslr_hmm(self, screen_deg_x: float = 30, screen_deg_y: float = 17):
        t = np.array(self.t)
        deg_x = np.array(self.x) * screen_deg_x
        deg_y = np.array(self.y) * screen_deg_y

        # Remove duplicate timestamps
        dt = np.diff(t)
        keep_mask = np.insert(dt != 0, 0, True)  # Always keep first sample

        t = t[keep_mask]
        deg_x = deg_x[keep_mask]
        deg_y = deg_y[keep_mask]

        gaze_data = np.column_stack((deg_x, deg_y))

        segmentation = nslr_fit_gaze(t, gaze_data)
        segment_classes = nslr_classify_segments(segmentation.segments)

        labels = []
        durations = []
        velocities = []

        for s, cls in zip(segmentation.segments, segment_classes):
            start, end = s.i[0], s.i[1]
            duration = t[end - 1] - t[start]
            if duration <= 0:
                continue
            dx = deg_x[end - 1] - deg_x[start]
            dy = deg_y[end - 1] - deg_y[start]
            velocity = np.sqrt(dx**2 + dy**2) / duration

            if cls == 1:   # FIXATION
                labels.append(1)
                velocities.append(0)
            elif cls == 2: # SACCADE
                labels.append(0)
                velocities.append(velocity)
            else:          # PSO, SMOOTH_PURSUIT, etc.
                labels.append(2)
                velocities.append(velocity)

            durations.append(duration)

        total_time = t[-1] - t[0]

        return np.array(labels), np.array(durations), np.array(velocities), total_time

    # ======================= PLOTTERS ======================

    def plot_gaze2d_time(self, ax: Axes | None = None, label: str = "signal", legend: bool = True):
        if ax is None:
            ax = plt.figure(figsize=(16, 8)).add_subplot(projection='3d')
        
        ax.plot(self._gaze2d["t"], self._gaze2d["x"], zs=self._gaze2d["y"], label=label)
        
        ax.set(
            xlabel="t", ylabel="x", zlabel="y",
            ylim = (0, 1), zlim = (0, 1)
        )
        
        if legend:
            ax.legend()
        
        return ax
