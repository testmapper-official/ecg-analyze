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
    
    # -----------------------------------------------------------------
    # 1. Амплитуда R (п.1.1)
    def get_R_amplitude(self, segment, target_samples=288):
        center = target_samples // 2
        return float(np.max(segment[center - 5 : center + 6]))

    # 2. Амплитуда Q (п.1.2)
    def get_Q_amplitude(self, segment, target_samples=288):
        center = target_samples // 2
        return float(np.min(segment[center - 15 : center - 2]))

    # 3. Амплитуда S (п.1.2)
    def get_S_amplitude(self, segment, target_samples=288):
        center = target_samples // 2
        return float(np.min(segment[center + 3 : center + 16]))

    # 4. Отношение R/S
    def get_R_over_S_ratio(self, segment, epsilon=1e-8):
        return abs(self.get_R_amplitude(segment)) / (abs(self.get_S_amplitude(segment)) + epsilon)

    # 5. Суммарный размах QRS
    def get_total_swing(self, segment):
        return self.get_R_amplitude(segment) - min(self.get_Q_amplitude(segment), self.get_S_amplitude(segment))

    # 6. Отношение Q/R
    def get_Q_R_ratio(self, segment, epsilon=1e-8):
        return abs(self.get_Q_amplitude(segment)) / (abs(self.get_R_amplitude(segment)) + epsilon)

    # 7. QRS длительность по 50%
    def get_QRS_duration_50(self, segment, target_samples=288):
        center = target_samples // 2
        R_amp = self.get_R_amplitude(segment, target_samples)
        if R_amp == 0:
            return 0.0
        thr = 0.5 * R_amp
        onset = center
        while onset > 0 and segment[onset] > thr:
            onset -= 1
        offset = center
        while offset < target_samples - 1 and segment[offset] > thr:
            offset += 1
        return float(offset - onset)

    # 8. Асимметрия QRS
    def get_asymmetry_ratio(self, segment, target_samples=288, epsilon=1e-8):
        center = target_samples // 2
        R_amp = self.get_R_amplitude(segment, target_samples)
        if R_amp == 0:
            return 1.0
        thr = 0.5 * R_amp
        onset = center
        while onset > 0 and segment[onset] > thr:
            onset -= 1
        offset = center
        while offset < target_samples - 1 and segment[offset] > thr:
            offset += 1
        rise = center - onset
        fall = offset - center
        return rise / (fall + epsilon)

    # 9. Макс. подъём
    def get_max_upstroke(self, segment, target_samples=288):
        center = target_samples // 2
        diffs = np.diff(segment[center - 20 : center + 1])
        return float(np.max(diffs))

    # 10. Макс. спад
    def get_max_downstroke(self, segment, target_samples=288):
        center = target_samples // 2
        diffs = np.diff(segment[center : center + 21])
        return float(np.min(diffs))

    # 11. Отношение крутизны
    def get_mean_slope_ratio(self, segment, epsilon=1e-8):
        up = self.get_max_upstroke(segment)
        down = self.get_max_downstroke(segment)
        return abs(up) / (abs(down) + epsilon)

    # 12. Zero-crossings в ядре QRS
    def get_zero_crossings_qrs(self, segment, target_samples=288):
        center = target_samples // 2
        zcr = sum(
            1 for i in range(center - 15, center + 15)
            if i + 1 < target_samples and segment[i] * segment[i + 1] < 0
        )
        return float(zcr)

    # 13. Энергия QRS / общая энергия
    def get_energy_ratio(self, segment, target_samples=288):
        center = target_samples // 2
        qrs_energy = np.sum(segment[center - 20 : center + 21] ** 2)
        total_energy = np.sum(segment ** 2)
        return float(qrs_energy / total_energy) if total_energy != 0 else 0.0

    # 14. Эксцесс QRS
    def get_kurtosis_qrs(self, segment, target_samples=288):
        center = target_samples // 2
        region = segment[center - 20 : center + 21]
        return float(np.mean(region ** 4))

    # 15. Энтропия QRS (с защитой от вырожденного диапазона)
    def get_entropy_qrs(self, segment, target_samples=288, bins=12):
        center = target_samples // 2
        region = segment[center - 20 : center + 21]
        # Если размах менее 1e-8, гистограмма не имеет смысла → энтропия 0
        if np.max(region) - np.min(region) < 1e-8:
            return 0.0
        hist, _ = np.histogram(region, bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    # --- Дополнительные для мультикласса (A, E, B) ---
    # D2. Энергия P-волны
    def get_P_energy_ratio(self, segment, target_samples=288, epsilon=1e-8):
        center = target_samples // 2
        p_zone = segment[center - 72 : center - 43]
        noise_zone = segment[-20:]
        return float(np.mean(np.abs(p_zone)) / (np.mean(np.abs(noise_zone)) + epsilon))

    # D3. Полярность P-волны
    def get_P_polarity(self, segment, target_samples=288):
        center = target_samples // 2
        p_zone = segment[center - 72 : center - 43]
        max_p = np.max(p_zone)
        min_p = np.min(p_zone)
        return float((max_p - min_p) / (abs(max_p) + abs(min_p) + 1e-8))

    # D4. QRSd_ratio (15%/50%)
    def get_QRSd_ratio_15_50(self, segment, target_samples=288):
        qrsd_15 = self.get_qrs_duration_norm(segment, target_samples) * 28.8  # отсчёты
        qrsd_50 = self.get_QRS_duration_50(segment, target_samples)
        return qrsd_15 / (qrsd_50 + 1e-8)

    # D5. R' (вторичный R)
    def get_R2_presence(self, segment, target_samples=288):
        center = target_samples // 2
        R_amp = self.get_R_amplitude(segment, target_samples)
        s_idx = center + np.argmin(segment[center : center + 30])
        if s_idx + 5 < target_samples:
            r2 = np.max(segment[s_idx + 5 : min(s_idx + 40, target_samples)])
            return 1.0 if r2 > 0.15 * R_amp else 0.0
        return 0.0

    # D6. Патологический Q
    def get_pathologic_Q(self, segment, target_samples=288):
        Q_amp = self.get_Q_amplitude(segment, target_samples)
        R_amp = self.get_R_amplitude(segment, target_samples)
        return 1.0 if abs(Q_amp) > 0.25 * abs(R_amp) else 0.0

    # D7. Смещение ST
    def get_ST_dev(self, segment, target_samples=288):
        center = target_samples // 2
        return float(np.mean(segment[center + 25 : center + 51]))

    # D10. Время активации
    def get_activation_time(self, segment, target_samples=288):
        center = target_samples // 2
        R_amp = self.get_R_amplitude(segment, target_samples)
        if R_amp == 0:
            return 0.0
        thr = 0.15 * R_amp
        onset = center
        while onset > 0 and abs(segment[onset]) > thr:
            onset -= 1
        return float(center - onset)

    # D11. P/R
    def get_P_over_R(self, segment, target_samples=288):
        center = target_samples // 2
        p_zone = segment[center - 72 : center - 43]
        P_amp = np.max(np.abs(p_zone))
        R_amp = abs(self.get_R_amplitude(segment, target_samples))
        return float(P_amp / (R_amp + 1e-8))
    
    # D12. Индикатор расщепления R-пика (Количество смен знака производной)
    def get_qrs_velocity_changes(self, segment, target_samples=288):
        """Считает смену знака производной в зоне QRS. 
        Для нормы: 1-2 смены, для блокады (M-образный): 3+ смены."""
        center = target_samples // 2
        start_idx = max(0, center - 30)
        end_idx = min(target_samples, center + 30)
        qrs_region = segment[start_idx:end_idx]
        
        d1 = np.gradient(qrs_region)
        sign_changes = np.sum(np.diff(np.sign(d1)) != 0)
        
        return float(sign_changes)