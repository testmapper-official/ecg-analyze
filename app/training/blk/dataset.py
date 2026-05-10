import os
import numpy as np
import torch
import wfdb
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from app.core.signal import Signal

TARGET_FS = 360
SEGMENT_SAMPLES = 288 

TARGET_TRAIN = 7000  # 3500 Normal + 3500 Blockade
TARGET_VAL = 1500    # 750 + 750
TARGET_TEST = 1500   # 750 + 750
AUGMENTATION_PERCENT = 0.75

# СТРОГАЯ ФИЛЬТРАЦИЯ: Только Норма и Блокады
NORMAL_SYMBOLS = ['N', 'V', 'A', 'F', 'E', 'r']
BLOCKADE_SYMBOLS = ['L', 'R'] # Объединяем в один класс

class DatasetBuilder:
    def __init__(self, db_root='DB'):
        self.db_root = db_root
        self.mydb_path = os.path.join(db_root, 'mydb')
        self.nstdb_path = os.path.join(db_root, 'nstdb')
        self.noise_data = self._load_nstdb()
        # Создаем папку для сохранения разделений
        self.splits_dir = os.path.join(db_root, 'splits')
        os.makedirs(self.splits_dir, exist_ok=True)

    def _load_nstdb(self):
        noises = {}
        for n_type in ['em', 'ma']:
            try:
                rec = wfdb.rdrecord(os.path.join(self.nstdb_path, n_type))
                if rec.fs != TARGET_FS:
                     noises[n_type] = Signal(data=rec.p_signal[:, 0], fs=rec.fs).resampled_data
                else:
                     noises[n_type] = rec.p_signal[:, 0]
            except: pass
        return noises

    def _save_split(self, patient_list, filename):
        filepath = os.path.join(self.splits_dir, filename)
        with open(filepath, 'w') as f:
            for pid in patient_list:
                f.write(f"{pid}\n")

    def build(self):
        patients_data = self._collect_data()
        if not patients_data: raise ValueError("Данные не собраны!")
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
            
            valid_peaks = [ann for ann in sig.annotations if ann['symbol'] in (NORMAL_SYMBOLS + BLOCKADE_SYMBOLS)]
            
            for i, ann in enumerate(valid_peaks):
                sym = ann['symbol']
                peak = ann['sample']
                
                if sym in NORMAL_SYMBOLS: label = 0
                elif sym in BLOCKADE_SYMBOLS: label = 1
                else: continue
                
                rr_prev = (valid_peaks[i]['sample'] - valid_peaks[i-1]['sample']) if i > 0 else int(0.8 * TARGET_FS)
                rr_norm = rr_prev / 288.0 
                segment = sig.get_segment(peak, SEGMENT_SAMPLES)
                
                if segment is not None:
                    qrs_dur = sig.get_qrs_duration_norm(segment)
                    
                    patients_data[patient_id].append({
                        'segment': segment,
                        'label': label,
                        'original_symbol': sym,
                        'rr_norm': rr_norm,
                        'qrs_dur': qrs_dur
                    })
            del sig 
        return patients_data

    def _split_and_balance(self, patients_data):
        patient_ids = list(patients_data.keys())
        has_blk = [pid for pid in patient_ids if any(d['label'] == 1 for d in patients_data[pid])]
        only_normal = [pid for pid in patient_ids if pid not in has_blk]
        
        if len(has_blk) >= 2:
            blk_train, blk_temp = train_test_split(has_blk, test_size=0.3, random_state=42)
            blk_val, blk_test = (train_test_split(blk_temp, test_size=0.5, random_state=42) if len(blk_temp) >= 2 else (blk_temp, []))
        else: blk_train, blk_val, blk_test = has_blk, [], []

        if len(only_normal) >= 2:
            n_train, n_temp = train_test_split(only_normal, test_size=0.3, random_state=42)
            n_val, n_test = (train_test_split(n_temp, test_size=0.5, random_state=42) if len(n_temp) >= 2 else (n_temp, []))
        else: n_train, n_val, n_test = only_normal, [], []

        train_patients = blk_train + n_train
        val_patients = blk_val + n_val
        test_patients = blk_test + n_test

        # Сохраняем разделения для каскада
        self._save_split(train_patients, 'blk_train.txt')
        self._save_split(val_patients, 'blk_val.txt')
        self._save_split(test_patients, 'blk_test.txt')

        X_train, y_train, sym_train = self._balance_split(patients_data, train_patients, TARGET_TRAIN, is_train=True)
        X_val, y_val, sym_val = self._balance_split(patients_data, val_patients, TARGET_VAL, is_train=False)
        X_test, y_test, sym_test = self._balance_split(patients_data, test_patients, TARGET_TEST, is_train=False)

        return X_train, y_train, sym_train, X_val, y_val, sym_val, X_test, y_test, sym_test

    def _balance_split(self, patients_data, patient_list, target_size, is_train):
        pool = []
        for pid in patient_list: pool.extend(patients_data[pid])
        if not pool:
            return torch.FloatTensor(np.zeros((0,2,SEGMENT_SAMPLES))), np.array([]), np.array([])
            
        normals = [d for d in pool if d['label'] == 0]
        blockades = [d for d in pool if d['label'] == 1]
        num_per_class = min(len(normals), len(blockades), target_size // 2)
        if num_per_class == 0: return torch.FloatTensor(np.zeros((0,2,SEGMENT_SAMPLES))), np.array([]), np.array([])
            
        sampled_blk_idx = np.random.choice(len(blockades), num_per_class, replace=False)
        sampled_norm_idx = np.random.choice(len(normals), num_per_class, replace=(len(normals) < num_per_class))
        final_data = [blockades[i] for i in sampled_blk_idx] + [normals[i] for i in sampled_norm_idx]
        np.random.shuffle(final_data)
        
        X = np.zeros((len(final_data), 2, SEGMENT_SAMPLES))
        y = np.array([d['label'] for d in final_data])
        sym = np.array([d['original_symbol'] for d in final_data])
        
        for i, d in enumerate(final_data): 
            X[i, 0, :] = d['segment'] 
            X[i, 1, :] = Signal(data=d['segment'], fs=360).get_qrs_duration_norm(d['segment']) * 2.0
        
        if is_train:
            aug_count = int(target_size * AUGMENTATION_PERCENT)
            X_aug, y_aug, sym_aug = self._augment(final_data, aug_count)
            if len(X_aug) > 0: X = np.concatenate([X, X_aug]); y = np.concatenate([y, y_aug]); sym = np.concatenate([sym, sym_aug])
        return torch.FloatTensor(X), y, sym

    def _get_random_noise_segment(self, length):
        noise_types = list(self.noise_data.keys())
        if not noise_types:
            return np.zeros(length)
            
        strategy = np.random.choice([
            'pure', 'mixed_physio', 'awgn', 'mixed_total', 
            'powerline', 'pop'
        ], p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3])
            
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
            freq = np.random.choice([50.0, 60.0])
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.arange(length) / TARGET_FS
            return np.sin(2 * np.pi * freq * t + phase)
            
        elif strategy == 'pop':
            noise = np.zeros(length)
            pop_idx = np.random.randint(length // 4, 3 * length // 4)
            amplitude = np.random.choice([-1, 1]) * np.random.uniform(3.0, 10.0)
            decay_rate = np.random.uniform(0.01, 0.05)
            
            noise[pop_idx] = amplitude
            if pop_idx + 1 < length:
                tail_len = length - pop_idx - 1
                noise[pop_idx+1:] = amplitude * np.exp(-decay_rate * np.arange(tail_len))
            return noise

    def _augment(self, base_data, aug_count):
        indices = np.random.choice(len(base_data), aug_count, replace=True)
        
        X_aug, y_aug, sym_aug = [], [], []
        for idx in indices:
            item = base_data[idx]
            temp_sig = Signal(data=item['segment'], fs=TARGET_FS)
            
            if np.random.rand() < 0.3:
                temp_sig = temp_sig.respiratory_modulation()
            if np.random.rand() < 0.1:
                temp_sig = temp_sig.adc_clipping()
                
            shifted_sig = temp_sig.time_shift(max_shift_ms=30)
            
            if self.noise_data:
                noise_frag = self._get_random_noise_segment(SEGMENT_SAMPLES)
                augmented_sig = shifted_sig.add_noise(noise_frag, snr_db_range=(-12, 6))
            else:
                augmented_sig = shifted_sig
            
            denoised_sig = augmented_sig.wavelet_denoise()
            denoised_sig.standardize()
            
            center = len(denoised_sig.resampled_data) // 2
            aug_seg = denoised_sig.get_segment(center, 288)
            
            if aug_seg is not None:
                noisy_qrs_dur = denoised_sig.get_qrs_duration_norm(aug_seg)
                
                x_item = np.zeros((2, 288))
                x_item[0, :] = aug_seg       
                x_item[1, :] = noisy_qrs_dur * 2.0
                X_aug.append(x_item); y_aug.append(item['label']); sym_aug.append(item['original_symbol'])
                
            del temp_sig, shifted_sig, augmented_sig, denoised_sig
            
        if len(X_aug) == 0: return np.array([]), np.array([]), np.array([])
        return np.array(X_aug), np.array(y_aug), np.array(sym_aug)
    
    def generate_noisy_test_set(self, X_test, y_test, sym_test):
        X_noisy = torch.zeros_like(X_test)
        for i in range(X_test.shape[0]):
            segment = X_test[i, 0, :].numpy()
            qrs_dur = X_test[i, 1, 0].item()
            
            temp_sig = Signal(data=segment, fs=360); shifted_sig = temp_sig.time_shift(max_shift_ms=20)
            if self.noise_data:
                noise_frag = self._get_random_noise_segment(288)
                augmented_sig = shifted_sig.add_noise(noise_frag, snr_db_range=(-3, 12))
            else: augmented_sig = shifted_sig
            denoised_sig = augmented_sig.wavelet_denoise(); denoised_sig.standardize()
            center = len(denoised_sig.resampled_data) // 2
            aug_seg = denoised_sig.get_segment(center, 288)
            if aug_seg is not None: 
                X_noisy[i, 0, :] = torch.FloatTensor(aug_seg)
                noisy_qrs_dur = Signal(data=aug_seg, fs=360).get_qrs_duration_norm(aug_seg)
                X_noisy[i, 1, :] = noisy_qrs_dur * 2.0
            else: X_noisy[i] = X_test[i]
            del temp_sig, shifted_sig, augmented_sig, denoised_sig
        return X_noisy, y_test, sym_test