import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pywt
from scipy.signal import savgol_filter


class Pupil:
    def __init__(self, t: npt.ArrayLike, 
                 lpd: npt.ArrayLike, 
                 rpd: npt.ArrayLike):
        self._pupil = pd.DataFrame({
            "t": t,
            "lpd": lpd,
            "rpd": rpd
        })

        # null value handling
        # LPD[t] := LPD[t] ?? RPD[t]
        # RPD[t] := RPD[t] ?? LPD[t]
        mask_lpd_null = (self._pupil.lpd.isna()) & (~self._pupil.rpd.isna())
        mask_rpd_null = (~self._pupil.lpd.isna()) & (self._pupil.rpd.isna())
        self._pupil.loc[mask_lpd_null, "lpd"] = self._pupil.loc[mask_lpd_null, "rpd"]
        self._pupil.loc[mask_rpd_null, "rpd"] = self._pupil.loc[mask_rpd_null, "lpd"]

        # throw rows where both points are null
        self._pupil.dropna(inplace=True)
        
        # calculate average pupil diameter        
        self._pupil["pd"] = (self._pupil.lpd + self._pupil.rpd)/2
        
        # calculate Benign Anisocoria
        # https://ieeexplore.ieee.org/document/10629234
        self._pupil["ba"] = self._pupil.lpd - self._pupil.rpd        

    # ======================= GETTERS =======================

    @property
    def t(self) -> npt.NDArray:
        """Returns Timestamp Values in an np.ndarray"""
        return self._pupil["t"].to_numpy()
    
    @property
    def lpd(self) -> npt.NDArray:
        """Returns Left Pupil Diameter Values in an np.ndarray"""
        return self._pupil["lpd"].to_numpy()
    
    @property
    def rpd(self) -> npt.NDArray:
        """Returns Right Pupil Diameter Values in an np.ndarray"""
        return self._pupil["rpd"].to_numpy()
    
    @property
    def pd(self) -> npt.NDArray:
        """Returns Average Pupil Diameter Values in an np.ndarray"""
        return self._pupil["pd"].to_numpy()
    
    @property
    def ba(self) -> npt.NDArray:
        """Returns Benign Anisocoria Values in an np.ndarray"""
        return self._pupil["ba"].to_numpy()
    
    @property
    def n_values(self) -> int:
        """Returns Number of Samples in our dataset"""
        return len(self._pupil)
        
    def __repr__(self):
        return self._pupil.__repr__()
    
    def __str__(self):
        return self._pupil.__str__()

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
        
        eyeleft = df["eyeleft"].apply(lambda x: x if type(x) == dict else {}).apply(pd.Series)["pupildiameter"]
        eyeright = df["eyeright"].apply(lambda x: x if type(x) == dict else {}).apply(pd.Series)["pupildiameter"]
        
        return cls(df.index.to_numpy(), eyeleft.to_numpy(), eyeright.to_numpy())
    
    @classmethod
    def from_pupil_capture(cls, csv_file_path: str, use_3d: bool = True):
        pupil = pd.read_csv(csv_file_path)
        
        right_eye = pupil[pupil.eye_id == 0]
        right_eye = right_eye[right_eye["method"] == ("3d c++" if use_3d else "2d c++")]
        left_eye = pupil[pupil.eye_id == 1]
        left_eye = left_eye[left_eye["method"] == ("3d c++" if use_3d else "2d c++")]
        
        samples = min(len(right_eye), len(left_eye))

        right_eye = right_eye.iloc[:samples]
        left_eye = left_eye.iloc[:samples]
        
        timestamp = (right_eye["pupil_timestamp"].to_numpy()+left_eye["pupil_timestamp"].to_numpy())/2
        
        column = "diameter" + ("_3d" if use_3d else "")
        
        return cls(timestamp, left_eye[column].to_numpy(), right_eye[column].to_numpy())

    # ======================= FEATURES ======================

    def rolling_mean(self, window: int = 100):
        smoothed_pupil = self._pupil.set_index("t")[["lpd", "rpd"]].rolling(window, center=True).mean().dropna().reset_index()
        return Pupil(
            smoothed_pupil["t"],
            smoothed_pupil["lpd"],
            smoothed_pupil["rpd"]
        )
    
    def split_into_segments(self, window: int, step: int) -> list:
        segments = []

        start = 0
        length = self.n_values

        while start + window < length:
            segment_df = self._pupil.iloc[start: start+window]

            if not segment_df.empty:
                # Create a new Pupil object using sliced columns
                segment_pupil = Pupil(
                    t=segment_df["t"].to_numpy(),
                    lpd=segment_df["lpd"].to_numpy(),
                    rpd=segment_df["rpd"].to_numpy()
                )
                segments.append(segment_pupil)

            start += step


        # t_vals = self._pupil["t"].values
        # start_time = t_vals[0]
        # end_time = t_vals[-1]

        # current_start = start_time
        # current_end = current_start + segment_duration

        # while current_end <= end_time:
        #     # Slice the DataFrame for the current segment
        #     segment_df = self._pupil[(self._pupil["t"] >= current_start) & (self._pupil["t"] < current_end)]

        #     if not segment_df.empty:
        #         # Create a new Pupil object using sliced columns
        #         segment_pupil = Pupil(
        #             t=segment_df["t"].values,
        #             lpd=segment_df["lpd"].values,
        #             rpd=segment_df["rpd"].values
        #         )
        #         segments.append(segment_pupil)

        #     # Move to next window
        #     current_start += step
        #     current_end = current_start + segment_duration

        return segments

    def compute_modmax(self, d: np.ndarray) -> np.ndarray:
        # compute signal modulus
        m = np.fabs(d)
        # get previous values
        l = np.roll(m, 1)
        l[0] = m[0]
        # get next values
        r = np.roll(m, -1)
        r[-1] = m[-1]
        # if value is larger than both neighbours, and strictly
        # larger than either, then it is a local maximum
        is_peak = ((l <= m) & (m >= r)) & ((l < m) | (m > r))
        m[~is_peak] = 0
        return m
    
    def compute_threshold(self, x: np.ndarray,
                          thresh_mode: str) -> np.ndarray:
        # threshold using universal threshold λuniv = σ * sqrt(2logn)
        # where σ is the standard deviation of the noise
        thresh = np.std(x) * math.sqrt(2*np.log2(len(x)))
        if thresh_mode == "greater":
            x_t = np.where(np.abs(x) > thresh, x, 0.0)
        else:
            x_t = pywt.threshold(x, thresh, mode=thresh_mode)
        return x_t

    def compute_index(self, x: np.ndarray, thresh_mode: str) -> float:
        # detect modulus maxima
        x_m = self.compute_modmax(x)
        # threshold using universal threshold
        x_t = self.compute_threshold(x_m, thresh_mode)
        # get signal duration (in seconds)
        T = self.t[-1] - self.t[0]
        # compute metric
        metric = (x_t != 0).sum() / T
        return metric

    def ipa(self, wavelet: str = "sym16", 
            periodization: str = "per", level: int = 2,
            thresh_mode: str = "hard") -> float:
        # obtain 2-level DWT of pupil diameter signal d
        cA2, cD2, cD1 = pywt.wavedec(self.pd, wavelet,
                                     mode=periodization, level=level)
        # normalize by 1/2^j, j=2 for 2-level DWT
        cD2 /= 2 ** (level/2)
        # compute IPA
        ipa = self.compute_index(cD2, thresh_mode)
        return ipa
    
    def lhipa(self, wavelet: str = "sym16", 
              periodization: str = "per",
              thresh_mode: str = "less") -> float:
        # find max decomposition level
        w = pywt.Wavelet(wavelet) # type: ignore
        maxlevel = pywt.dwt_max_level(self.n_values, filter_len=w.dec_len)
        # set high and low frequency band indices
        hif, lof = 1, maxlevel//2
        # get detail coefficients of pupil diameter signal
        cD_H = pywt.downcoef("d", self.pd, wavelet, periodization, level=hif)
        cD_L = pywt.downcoef("d", self.pd, wavelet, periodization, level=lof)
        # normalise by 1/2^j
        cD_H /= 2 ** (hif/2)
        cD_L /= 2 ** (lof/2)
        # obtain the LH:HF ratio
        indices = [(2**(lof-hif))*i for i in range(len(cD_L))]
        cD_LH = cD_L / cD_H[indices]
        # compute LHIPA
        lhipa = self.compute_index(cD_LH, thresh_mode)

        return lhipa
    
    def ripa(
        self,
        window_size: int | None = None, # use None to get single value for entire sequence
        step_size: int = 1,
        low_pass_window: int = 11, 
        low_pass_order: int = 2,
        high_pass_window: int = 11, 
        high_pass_order: int = 4,
        thresh_mode: str = "greater"
    ) -> np.ndarray:
        if window_size is None:
            window_size = self.n_values
        
        windows_pd = np.stack([self.pd[i:i+window_size]for i in range(0, len(self.pd) - window_size + 1, step_size)])
        windows_t = np.stack([self.t[i:i+window_size]for i in range(0, len(self.t) - window_size + 1, step_size)])
        
        low_freq = savgol_filter(windows_pd, low_pass_window, low_pass_order, axis=1)
        high_freq = savgol_filter(windows_pd, high_pass_window, high_pass_order, deriv=1, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(low_freq, high_freq, out=np.zeros_like(low_freq), where=high_freq !=0 )
        
        # Detect modulus maxima
        m = np.abs(ratio)
        # Get previous values (with padding)
        l = np.roll(m, 1, axis=1)
        l[:, 0] = m[:, 0]
        # Get next values (with padding)
        r = np.roll(m, -1, axis=1)
        r[:, -1] = m[:, -1]
        # If value is larger than both neighbors and strictly larger than either, it's a local maximum
        is_peak = ((l <= m) & (m >= r)) & ((l < m) | (m > r))
        modmax = np.zeros_like(m)
        modmax[is_peak] = m[is_peak]
        
        thresh = np.std(modmax, axis=1, keepdims=True) * np.sqrt(2 * np.log2(modmax.shape[1]))
        if thresh_mode == "greater":
            thresholded = np.where(np.abs(modmax) > thresh, modmax, 0.0)
        else:
            thresholded = pywt.threshold(modmax, thresh, mode=thresh_mode)
        
        # Get signal duration (in seconds)
        T = windows_t[:, -1] - windows_t[:, 0]
        
        # Compute RIPA metric
        ripa = (thresholded != 0).sum(axis=1) / T
        
        return ripa

    # ======================= PLOTTERS ======================

    def plot_pd(self, ax: Axes | None = None, label: str = "diameter", legend: bool = True, start: int = 0, end: int | None = None, plot_kwargs: dict = {}):
        if ax is None:
            ax = plt.figure(figsize=(16, 8)).add_subplot()
        
        min_pt = (start+len(self._pupil))%len(self._pupil)
        max_pt = (end+len(self._pupil)+1)%len(self._pupil) if end is not None else len(self._pupil)
        
        plot_kwargs = {
            "legend": legend,
            "label": label,
            **plot_kwargs
        }
        
        if "x" in plot_kwargs: del plot_kwargs["x"]
        if "y" in plot_kwargs: del plot_kwargs["y"]
        if "ax" in plot_kwargs: del plot_kwargs["ax"]
        
        ax = self._pupil.iloc[min_pt:max_pt].plot(x="t", y="pd", ax=ax, **plot_kwargs)
        return ax
    
    def plot_lpd(self, ax: Axes | None = None, label: str = "diameter", legend: bool = True, start: int = 0, end: int | None = None, plot_kwargs: dict = {}):
        if ax is None:
            ax = plt.figure(figsize=(16, 8)).add_subplot()
        
        min_pt = (start+len(self._pupil))%len(self._pupil)
        max_pt = (end+len(self._pupil)+1)%len(self._pupil) if end is not None else len(self._pupil)
        
        plot_kwargs = {
            "legend": legend,
            "label": label,
            **plot_kwargs
        }
        
        if "x" in plot_kwargs: del plot_kwargs["x"]
        if "y" in plot_kwargs: del plot_kwargs["y"]
        if "ax" in plot_kwargs: del plot_kwargs["ax"]
        
        ax = self._pupil.iloc[min_pt:max_pt].plot(x="t", y="lpd", ax=ax, **plot_kwargs)
        return ax
    
    def plot_rpd(self, ax: Axes | None = None, label: str = "diameter", legend: bool = True, start: int = 0, end: int | None = None, plot_kwargs: dict = {}):
        if ax is None:
            ax = plt.figure(figsize=(16, 8)).add_subplot()
        
        min_pt = (start+len(self._pupil))%len(self._pupil)
        max_pt = (end+len(self._pupil)+1)%len(self._pupil) if end is not None else len(self._pupil)
        
        plot_kwargs = {
            "legend": legend,
            "label": label,
            **plot_kwargs
        }
        
        if "x" in plot_kwargs: del plot_kwargs["x"]
        if "y" in plot_kwargs: del plot_kwargs["y"]
        if "ax" in plot_kwargs: del plot_kwargs["ax"]
        
        ax = self._pupil.iloc[min_pt:max_pt].plot(x="t", y="rpd", ax=ax, **plot_kwargs)
        return ax
    
    def plot_ba(self, ax: Axes | None = None, start: int = 0, end: int | None = None, plot_kwargs: dict = {}):
        if ax is None:
            ax = plt.figure(figsize=(16, 8)).add_subplot()
        
        min_pt = (start+len(self._pupil))%len(self._pupil)
        max_pt = (end+len(self._pupil)+1)%len(self._pupil) if end is not None else len(self._pupil)
        
        plot_kwargs = {
            "color": "black",
            **plot_kwargs,
            "legend": False
        }
        
        if "x" in plot_kwargs: del plot_kwargs["x"]
        if "y" in plot_kwargs: del plot_kwargs["y"]
        if "ax" in plot_kwargs: del plot_kwargs["ax"]
        
        self._pupil.iloc[min_pt:max_pt].plot(x="t", y="ba", ax=ax, **plot_kwargs)
        ax.axhline(y=0, color="red", linewidth=1, linestyle="--")
        ax.set(
                title = "$BA[t] = LPD[t] - RPD[t]$",
                xlabel = "timestamp, $t$",
                ylabel = r"$\Delta$(pupil diameter)"
        )
        return ax
