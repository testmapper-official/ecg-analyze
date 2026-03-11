import numpy as np
from scipy import signal
import neurokit2 as nk

class ECGProcessor:
    def __init__(self, fs=360):
        self.fs = fs

    def filter_signal(self, raw_signal, fs=None, lowcut=0.5, highcut=45.0):
        """
        Полосовая фильтрация.
        Если fs передан, используется он, иначе значение из self.fs.
        """
        # Используем переданную fs или дефолтную из экземпляра
        current_fs = fs if fs is not None else self.fs
        
        nyquist = 0.5 * current_fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0: high = 0.99
        if low <= 0: low = 0.01
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, raw_signal)
        return filtered_signal

    def detect_r_peaks(self, ecg_signal, sampling_rate=None):
        """
        Детекция R-пиков.
        """
        # Используем переданную частоту или дефолтную
        current_fs = sampling_rate if sampling_rate is not None else self.fs

        cleaned = nk.ecg_clean(ecg_signal, sampling_rate=current_fs, method="pantompkins1985")
        signals, info = nk.ecg_peaks(cleaned, sampling_rate=current_fs, method="pantompkins1985")
        r_peaks = info['ECG_R_Peaks']
        return r_peaks

    def get_segments(self, ecg_signal, r_peaks, window_size=288):
        segments = []
        valid_indices = []
        half_win = window_size // 2
        
        for r in r_peaks:
            start = r - half_win
            end = r + half_win
            
            if start >= 0 and end < len(ecg_signal):
                segment = ecg_signal[start:end]
                if np.std(segment) > 0:
                    segment = (segment - np.mean(segment)) / np.std(segment)
                    segments.append(segment)
                    valid_indices.append(r)
                    
        return np.array(segments), valid_indices