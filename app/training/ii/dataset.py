import os
import numpy as np
import torch
import wfdb
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from app.core.signal import Signal

TARGET_FS = 360
SEGMENT_SAMPLES = 288 

TARGET_TRAIN = 7000  # 3500 Normal + 3500 PSS
TARGET_VAL = 1500    # 750 Normal + 750 PSS 
TARGET_TEST = 1500   # 750 Normal + 750 PSS

AUGMENTATION_PERCENT = 0.75 # 75% от суммарного норматива (добавит 5250 аугментаций к 7000 базе)

# СТРОГАЯ ФИЛЬТРАЦИЯ
NORMAL_SYMBOLS = ['N', 'E', 'A']
PSS_SYMBOLS = ['V', 'F', 'r']

class DatasetBuilder:
    def __init__(self, db_root='DB'):
        self.db_root = db_root
        self.mydb_path = os.path.join(db_root, 'mydb')
        self.nstdb_path = os.path.join(db_root, 'nstdb')
        self.noise_data = self._load_nstdb()

    def _load_nstdb(self):
        noises = {}
        for n_type in ['em', 'ma']:
            try:
                rec = wfdb.rdrecord(os.path.join(self.nstdb_path, n_type))
                if rec.fs != TARGET_FS:
                     noises[n_type] = Signal(data=rec.p_signal[:, 0], fs=rec.fs).resampled_data
                else:
                     noises[n_type] = rec.p_signal[:, 0]
                print(f"Успешно загружен шум NSTDB: {n_type} ({len(noises[n_type])} отсчетов)")
            except Exception as e:
                print(f"ОШИБКА ЗАГРУЗКИ ШУМА {n_type}: {e}")
        return noises

    def build(self):
        patients_data = self._collect_data()
        if not patients_data:
            raise ValueError("Данные не собраны!")
        return self._split_and_balance(patients_data)

    def _collect_data(self):
        records_file = os.path.join(self.mydb_path, 'RECORDS')
        if not os.path.exists(records_file): return None
        
        patients_data = defaultdict(list)
        
        with open(records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
        
        for rec_path in records:
            if not rec_path.startswith('II/'): continue
            parts = rec_path.split('/')
            patient_id = parts[1]
            full_path = os.path.join(self.mydb_path, rec_path)
            
            sig = Signal(record_path=full_path)
            sig.standardize()
            
            valid_peaks = [ann for ann in sig.annotations if ann['symbol'] in (NORMAL_SYMBOLS + PSS_SYMBOLS)]
            
            for i, ann in enumerate(valid_peaks):
                sym = ann['symbol']
                peak = ann['sample']
                
                if sym in NORMAL_SYMBOLS: binary_label = 0
                elif sym in PSS_SYMBOLS: binary_label = 1
                else: continue
                
                rr_prev = (valid_peaks[i]['sample'] - valid_peaks[i-1]['sample']) if i > 0 else int(0.8 * TARGET_FS)
                rr_norm = rr_prev / 288.0 
                
                segment = sig.get_segment(peak, SEGMENT_SAMPLES)
                
                if segment is not None:
                    patients_data[patient_id].append({
                        'segment': segment,
                        'binary_label': binary_label,
                        'original_symbol': sym,
                        'rr_norm': rr_norm
                    })
            del sig 
            
        return patients_data

    def _split_and_balance(self, patients_data):
        patient_ids = list(patients_data.keys())
        has_pss = [pid for pid in patient_ids if any(d['binary_label'] == 1 for d in patients_data[pid])]
        only_normal = [pid for pid in patient_ids if pid not in has_pss]
        
        pss_train, pss_temp = train_test_split(has_pss, test_size=0.3, random_state=42)
        pss_val, pss_test = train_test_split(pss_temp, test_size=0.5, random_state=42)
        
        norm_train, norm_temp = train_test_split(only_normal, test_size=0.3, random_state=42)
        norm_val, norm_test = train_test_split(norm_temp, test_size=0.5, random_state=42)
        
        train_patients = pss_train + norm_train
        val_patients = pss_val + norm_val
        test_patients = pss_test + norm_test

        X_train, y_train, sym_train = self._balance_split(patients_data, train_patients, TARGET_TRAIN, is_train=True)
        X_val, y_val, sym_val = self._balance_split(patients_data, val_patients, TARGET_VAL, is_train=False)
        X_test, y_test, sym_test = self._balance_split(patients_data, test_patients, TARGET_TEST, is_train=False)

        return X_train, y_train, sym_train, X_val, y_val, sym_val, X_test, y_test, sym_test

    def _balance_split(self, patients_data, patient_list, target_size, is_train):
        pool = []
        for pid in patient_list:
            pool.extend(patients_data[pid])
            
        normals = [d for d in pool if d['binary_label'] == 0]
        pss = [d for d in pool if d['binary_label'] == 1]
        
        num_per_class = min(len(normals), len(pss), target_size // 2)
        if num_per_class == 0:
            raise ValueError(f"Недостаточно данных ПСС или Нормы для формирования выборки!")
            
        sampled_pss_idx = np.random.choice(len(pss), num_per_class, replace=False)
        sampled_norm_idx = np.random.choice(len(normals), num_per_class, replace=True) # replace=True чтобы не падать
        
        final_data = [pss[i] for i in sampled_pss_idx] + [normals[i] for i in sampled_norm_idx]
        np.random.shuffle(final_data)
        
        X = np.zeros((len(final_data), 2, SEGMENT_SAMPLES))
        y = np.array([d['binary_label'] for d in final_data])
        sym = np.array([d['original_symbol'] for d in final_data])
        
        for i, d in enumerate(final_data):
            X[i, 0, :] = d['segment'] 
            X[i, 1, :] = d['rr_norm'] 
        
        if is_train:
            aug_count = int(target_size * AUGMENTATION_PERCENT)
            X_aug, y_aug, sym_aug = self._augment(final_data, aug_count)
            if len(X_aug) > 0:
                X = np.concatenate([X, X_aug], axis=0)
                y = np.concatenate([y, y_aug], axis=0)
                sym = np.concatenate([sym, sym_aug], axis=0)
            
        return torch.FloatTensor(X), y, sym

    # НОВЫЙ МЕТОД: Генерация диверсифицированного шума
    def _get_random_noise_segment(self, length):
        noise_types = list(self.noise_data.keys())
        if not noise_types:
            return np.zeros(length)
            
        # Расширенные стратегии:
        # 20% - Чистый физиологический (мышцы/электрод)
        # 10% - Смесь физиологий
        # 10% - Чистый приборный БГШ
        # 10% - Смесь Физиология + БГШ
        # 15% - Сетевая наводка 50 Гц
        # 15% - Прострел электрода (Pop)
        # 10% - Модуляция дыханием (возвращаем мультипликативный шум, единицы)
        # 10% - Клиппинг (возвращаем специальный флаг, обработаем вне)
        
        # Для простоты вернем аддитивные шумы, а мультипликативные/нелинейные сделаем отдельным методом
        strategy = np.random.choice([
            'pure', 'mixed_physio', 'awgn', 'mixed_total', 
            'powerline', 'pop'
        ], p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3]) # Повышаем шанс на самые опасные: 50Hz и Pop
            
        if strategy == 'pure' or len(noise_types) == 1:
            n_type = np.random.choice(noise_types)
            noise_base = self.noise_data[n_type]
            start_idx = np.random.randint(0, len(noise_base) - length)
            return noise_base[start_idx : start_idx + length]
            
        elif strategy == 'mixed_physio':
            base1 = self.noise_data[noise_types[0]]
            base2 = self.noise_data[noise_types[1]]
            start1 = np.random.randint(0, len(base1) - length)
            start2 = np.random.randint(0, len(base2) - length)
            frag1 = base1[start1 : start1 + length]
            frag2 = base2[start2 : start2 + length]
            w1 = np.random.uniform(0.3, 0.7)
            return w1 * frag1 + (1.0 - w1) * frag2
            
        elif strategy == 'awgn':
            return np.random.normal(0, 1.0, length)
            
        elif strategy == 'mixed_total':
            n_type = np.random.choice(noise_types)
            noise_base = self.noise_data[n_type]
            start_idx = np.random.randint(0, len(noise_base) - length)
            physio_frag = noise_base[start_idx : start_idx + length]
            awgn_frag = np.random.normal(0, 1.0, length)
            w1 = np.random.uniform(0.6, 0.8)
            return w1 * physio_frag + (1.0 - w1) * awgn_frag
            
        elif strategy == 'powerline':
            # 50 Гц (или 60 Гц) сетевая наводка
            freq = np.random.choice([50.0, 60.0])
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.arange(length) / TARGET_FS
            # Генерируем чистую синусоиду (амплитуду подгонит add_noise)
            return np.sin(2 * np.pi * freq * t + phase)
            
        elif strategy == 'pop':
            # "Прострел" электрода: резкий spike + экспоненциальный спад
            noise = np.zeros(length)
            pop_idx = np.random.randint(length // 4, 3 * length // 4) # Где-то в середине окна
            amplitude = np.random.choice([-1, 1]) * np.random.uniform(3.0, 10.0)
            decay_rate = np.random.uniform(0.01, 0.05)
            
            noise[pop_idx] = amplitude
            if pop_idx + 1 < length:
                # Экспоненциальный спад после скачка
                tail_len = length - pop_idx - 1
                noise[pop_idx+1:] = amplitude * np.exp(-decay_rate * np.arange(tail_len))
            return noise

    def _augment(self, base_data, aug_count):
        indices = np.random.choice(len(base_data), aug_count, replace=True)
        
        X_aug, y_aug, sym_aug = [], [], []
        for idx in indices:
            item = base_data[idx]
            temp_sig = Signal(data=item['segment'], fs=TARGET_FS)
            
            # 1. Искажения морфологии (ДО добавления шума)
            if np.random.rand() < 0.3: # 30% шанс модуляции дыханием
                temp_sig = temp_sig.respiratory_modulation()
            if np.random.rand() < 0.1: # 10% шанс клиппинга АЦП
                temp_sig = temp_sig.adc_clipping()
                
            # 2. Временной сдвиг
            shifted_sig = temp_sig.time_shift(max_shift_ms=30)
            
            # 3. Наложение шума
            if self.noise_data:
                noise_frag = self._get_random_noise_segment(SEGMENT_SAMPLES)
                augmented_sig = shifted_sig.add_noise(noise_frag, snr_db_range=(-12, 6))
            else:
                augmented_sig = shifted_sig
            
            # 4. Вейвлет-фильтрация и нормализация
            denoised_sig = augmented_sig.wavelet_denoise()
            denoised_sig.standardize()
            
            center = len(denoised_sig.resampled_data) // 2
            aug_seg = denoised_sig.get_segment(center, SEGMENT_SAMPLES)
            
            if aug_seg is not None:
                x_item = np.zeros((2, SEGMENT_SAMPLES))
                x_item[0, :] = aug_seg       
                x_item[1, :] = item['rr_norm']
                X_aug.append(x_item)
                y_aug.append(item['binary_label'])
                sym_aug.append(item['original_symbol'])
                
            del temp_sig, shifted_sig, augmented_sig, denoised_sig
            
        if len(X_aug) == 0: return np.array([]), np.array([]), np.array([])
        return np.array(X_aug), np.array(y_aug), np.array(sym_aug)
    
    def generate_noisy_test_set(self, X_test, y_test, sym_test):
        """Создает копию тестовой выборки, но сильно зашумленную"""
        X_noisy = torch.zeros_like(X_test)
        
        for i in range(X_test.shape[0]):
            segment = X_test[i, 0, :].numpy()
            rr_norm = X_test[i, 1, 0].item()
            
            # Создаем сигнал из окна
            temp_sig = Signal(data=segment, fs=TARGET_FS)
            shifted_sig = temp_sig.time_shift(max_shift_ms=20) # Слегка сдвигаем
            
            # В методе generate_noisy_test_set замените блок if self.noise_data:
            if self.noise_data:
                # Генерируем композитный/случайный шум для теста
                noise_frag = self._get_random_noise_segment(SEGMENT_SAMPLES)
                augmented_sig = shifted_sig.add_noise(noise_frag, snr_db_range=(-12, 6))
            else:
                augmented_sig = shifted_sig
                
            # Обязательно применяем вейвлет, как при обучении
            denoised_sig = augmented_sig.wavelet_denoise()
            denoised_sig.standardize()
            
            center = len(denoised_sig.resampled_data) // 2
            aug_seg = denoised_sig.get_segment(center, SEGMENT_SAMPLES)
            
            if aug_seg is not None:
                X_noisy[i, 0, :] = torch.FloatTensor(aug_seg)
                X_noisy[i, 1, :] = rr_norm
            else:
                X_noisy[i] = X_test[i] # Fallback
                
            del temp_sig, shifted_sig, augmented_sig, denoised_sig
            
        return X_noisy, y_test, sym_test