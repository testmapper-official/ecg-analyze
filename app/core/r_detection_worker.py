from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import neurokit2 as nk
from app.core.signal import Signal

class RDetectionWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, signal_data, orig_fs):
        super().__init__()
        self.signal_data = signal_data
        self.orig_fs = orig_fs

    def run(self):
        try:
            self.progress.emit(20)
            sig_obj = Signal(data=self.signal_data, fs=self.orig_fs)
            sig_obj.standardize()
            data_360 = sig_obj.resampled_data
            
            self.progress.emit(50)
            signals, info = nk.ecg_peaks(data_360, sampling_rate=360, method="pantompkins1985")
            r_peaks = info['ECG_R_Peaks']
            
            self.progress.emit(90)
            results = [{'sample': int(r), 'auto_symbol': 'N', 'manual_symbol': ''} for r in r_peaks]
            
            self.progress.emit(100)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))