import os
import copy
import numpy as np
import wfdb
import pywt # Добавлено для вейвлет-фильтрации
from scipy.signal import resample

class Signal:
    TARGET_FS = 360

    def __init__(self, record_path=None, data=None, fs=360, annotations=None):
        self.raw_data = None
        self.resampled_data = None
        self.fs = fs
        self.annotations = [] 
        
        if record_path:
            self._load_from_file(record_path)
        elif data is not None:
            self.raw_data = data
            self.fs = fs
            self.resampled_data = data if fs == self.TARGET_FS else self._resample_data(data, fs)
            self.annotations = annotations if annotations else []
        else:
            raise ValueError("Необходимо указать record_path или data")

    def _load_from_file(self, record_path):
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            idx = 0
            if hasattr(record, 'sig_name'):
                for i, n in enumerate(record.sig_name):
                    if n.lower() in ['ii', 'mlii']: 
                        idx = i
                        break
            
            self.raw_data = record.p_signal[:, idx]
            self.fs = record.fs
            
            self.resampled_data = self.raw_data if self.fs == self.TARGET_FS else self._resample_data(self.raw_data, self.fs)
            
            ratio = self.TARGET_FS / self.fs
            for i, sym in enumerate(annotation.symbol):
                self.annotations.append({
                    'sample': int(annotation.sample[i] * ratio),
                    'symbol': sym.upper()
                })
        except Exception as e:
            print(f"Ошибка загрузки {record_path}: {e}")

    def _resample_data(self, data, orig_fs):
        num_samples = int(len(data) * self.TARGET_FS / orig_fs)
        return resample(data, num_samples)

    def standardize(self):
        mean = np.mean(self.resampled_data)
        std = np.std(self.resampled_data)
        if std > 0:
            self.resampled_data = (self.resampled_data - mean) / std
        return self

    def get_segment(self, peak_idx, window_samples=288):
        start = peak_idx - window_samples // 2
        end = peak_idx + window_samples // 2
        # ИСПРАВЛЕНО: <= вместо <, чтобы не получать None для центрального окна
        if start >= 0 and end <= len(self.resampled_data):
            return self.resampled_data[start:end]
        return None

    def time_shift(self, max_shift_ms=50):
        shift_samples = int(np.random.uniform(-max_shift_ms, max_shift_ms) * self.TARGET_FS / 1000)
        new_data = np.roll(self.resampled_data, shift_samples)
        
        new_sig = Signal(data=new_data, fs=self.TARGET_FS, annotations=copy.deepcopy(self.annotations))
        for ann in new_sig.annotations:
            ann['sample'] += shift_samples
        return new_sig

    def add_noise(self, noise_data, snr_db_range=(-12, 6)):
        clean_signal = self.resampled_data
        sig_len = len(clean_signal)
        
        # Если передан массив большей длины (весь файл шума), берем случайный кусок
        if len(noise_data) > sig_len:
            start_idx = np.random.randint(0, len(noise_data) - sig_len)
            noise_frag = noise_data[start_idx : start_idx + sig_len]
        # Если передан массив ровно под размер окна (уже нарезанный кусок), используем его
        elif len(noise_data) == sig_len:
            noise_frag = noise_data
        # Если шум короче сигнала (маловероятно, но зациклим)
        else:
            repeat_factor = int(np.ceil(sig_len / len(noise_data)))
            noise_frag = np.tile(noise_data, repeat_factor)[:sig_len]
        
        signal_rms = np.sqrt(np.mean(clean_signal ** 2))
        noise_rms = np.sqrt(np.mean(noise_frag ** 2))
        
        if noise_rms == 0: return copy.deepcopy(self)
        
        snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
        target_noise_rms = signal_rms / (10 ** (snr_db / 20))
        noisy_data = clean_signal + noise_frag * (target_noise_rms / noise_rms)
        
        return Signal(data=noisy_data, fs=self.TARGET_FS, annotations=copy.deepcopy(self.annotations))

    def wavelet_denoise(self, wavelet='db6', level=5):
        """Вейвлет-фильтрация. Возвращает НОВЫЙ инстанс"""
        sig = self.resampled_data
        
        # АВТОМАТИЧЕСКИЙ РАСЧЕТ УРОВНЯ ДЛЯ КОРОТКИХ ОКОН
        # Вычисляем максимально допустимый уровень разложения для текущей длины сигнала
        max_possible_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
        # Выбираем минимальный из запрошенного и возможного
        actual_level = min(level, max_possible_level)
        
        coeffs = pywt.wavedec(sig, wavelet, level=actual_level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
        coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
        denoised_data = pywt.waverec(coeffs, wavelet)[:len(sig)]
        
        return Signal(data=denoised_data, fs=self.TARGET_FS, annotations=copy.deepcopy(self.annotations))

    def respiratory_modulation(self, max_depth=0.3):
        """Аугментация: модуляция амплитуды из-за дыхания. Возвращает НОВЫЙ инстанс"""
        # Случайная частота дыхания 0.15 - 0.4 Гц
        resp_freq = np.random.uniform(0.15, 0.4)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.arange(len(self.resampled_data)) / self.TARGET_FS
        
        # Коэффициент модуляции (1.0 +/- max_depth)
        mod = 1.0 + max_depth * np.sin(2 * np.pi * resp_freq * t + phase)
        mod_data = self.resampled_data * mod
        
        return Signal(data=mod_data, fs=self.TARGET_FS, annotations=copy.deepcopy(self.annotations))

    def adc_clipping(self, threshold_percentile=95):
        """Аугментация: клиппинг АЦП. Возвращает НОВЫЙ инстанс"""
        # Находим порог клиппинга как процентиль от максимальной амплитуды сигнала
        max_abs = np.max(np.abs(self.resampled_data))
        if max_abs == 0: return copy.deepcopy(self)
        
        clip_level = max_abs * np.random.uniform(0.6, 0.9) # Срезаем от 10% до 40% верхушки
        clipped_data = np.clip(self.resampled_data, -clip_level, clip_level)
        
        return Signal(data=clipped_data, fs=self.TARGET_FS, annotations=copy.deepcopy(self.annotations))
    
    def get_qrs_duration_norm(self, segment, target_samples=288):
        """Оценка нормализованной длительности QRS в сегменте"""
        center = target_samples // 2
        abs_seg = np.abs(segment)
        
        # Сглаживаем для устойчивости к мелкому шуму
        w = 5
        abs_smooth = np.convolve(abs_seg, np.ones(w)/w, mode='same')
        
        peak_val = abs_smooth[center]
        if peak_val == 0: return 1.0 # Защита
        
        threshold = peak_val * 0.15 # 15% от высоты пика
        
        # Ищем начало (идем влево от пика)
        onset = center
        while onset > 0 and abs_smooth[onset] > threshold:
            onset -= 1
            
        # Ищем конец (идем вправо от пика)
        offset = center
        while offset < target_samples - 1 and abs_smooth[offset] > threshold:
            offset += 1
            
        duration_samples = offset - onset
        
        # Нормализуем относительно 80 мс (нормальная ширина QRS).
        # При 360 Гц: 80 мс = 28.8 сэмплов. 
        # Норма выдаст ~1.0, Блокада выдаст ~1.5 - 2.0
        norm_duration = duration_samples / 28.8 
        return norm_duration