import os, numpy as np, torch, wfdb
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.core.signal import Signal

TARGET_FS = 360
SEGMENT_SAMPLES = 288
NUM_FEATURES = 25 

# Строго как в TCN
NORMAL_SYMBOLS = ['N', 'E', 'A']
PSS_SYMBOLS = ['V', 'F', 'r'] # Переименуем в V для графиков
ALL_SYMBOLS = NORMAL_SYMBOLS + PSS_SYMBOLS

TARGET_TRAIN = 7000  # 3500 Normal + 3500 V
TARGET_VAL = 1500    # 750 Normal + 750 V
TARGET_TEST = 1500   # 750 Normal + 750 V
AUGMENTATION_PERCENT = 2.5  

class ParametricDatasetBuilder:
    def __init__(self, db_root='DB'):
        self.db_root = db_root
        self.mydb_path = os.path.join(db_root, 'mydb')
        self.nstdb_path = os.path.join(db_root, 'nstdb')
        self.noise_data = self._load_nstdb()
        self.splits_dir = os.path.join(db_root, 'splits')
        os.makedirs(self.splits_dir, exist_ok=True)
        self.scaler = StandardScaler()

    def _load_nstdb(self):
        noises = {}
        for n_type in ['em', 'ma']:
            try:
                rec = wfdb.rdrecord(os.path.join(self.nstdb_path, n_type))
                if rec.fs != TARGET_FS:
                    noises[n_type] = Signal(data=rec.p_signal[:, 0], fs=rec.fs).resampled_data
                else:
                    noises[n_type] = rec.p_signal[:, 0]
            except:
                pass
        return noises

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

    def _extract_features(self, segment, rr_norm):
        sig_obj = Signal(data=segment, fs=TARGET_FS)
        feats = []
        feats.append(sig_obj.get_R_amplitude(segment))
        feats.append(sig_obj.get_Q_amplitude(segment))
        feats.append(sig_obj.get_S_amplitude(segment))
        feats.append(sig_obj.get_R_over_S_ratio(segment))
        feats.append(sig_obj.get_total_swing(segment))
        feats.append(sig_obj.get_Q_R_ratio(segment))
        feats.append(sig_obj.get_QRS_duration_50(segment))
        feats.append(sig_obj.get_asymmetry_ratio(segment))
        feats.append(sig_obj.get_max_upstroke(segment))
        feats.append(sig_obj.get_max_downstroke(segment))
        feats.append(sig_obj.get_mean_slope_ratio(segment))
        feats.append(sig_obj.get_zero_crossings_qrs(segment))
        feats.append(sig_obj.get_energy_ratio(segment))
        feats.append(sig_obj.get_kurtosis_qrs(segment))
        feats.append(sig_obj.get_entropy_qrs(segment))
        feats.append(sig_obj.get_qrs_duration_norm(segment))
        feats.append(sig_obj.get_P_energy_ratio(segment))
        feats.append(sig_obj.get_P_polarity(segment))
        feats.append(sig_obj.get_QRSd_ratio_15_50(segment))
        feats.append(sig_obj.get_R2_presence(segment))
        feats.append(sig_obj.get_pathologic_Q(segment))
        feats.append(sig_obj.get_ST_dev(segment))
        feats.append(sig_obj.get_activation_time(segment))
        feats.append(sig_obj.get_P_over_R(segment))
        feats.append(rr_norm)
        
        feats = np.array(feats, dtype=np.float32)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def _collect_data(self):
        records_file = os.path.join(self.mydb_path, 'RECORDS')
        if not os.path.exists(records_file): 
            return None
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
            valid = [ann for ann in sig.annotations if ann['symbol'] in ALL_SYMBOLS]
            for i, ann in enumerate(valid):
                sym = ann['symbol']
                peak = ann['sample']
                rr_prev = (valid[i]['sample'] - valid[i-1]['sample']) if i > 0 else int(0.8 * TARGET_FS)
                rr_norm = rr_prev / 288.0
                segment = sig.get_segment(peak, SEGMENT_SAMPLES)
                if segment is not None:
                    feats = self._extract_features(segment, rr_norm)
                    binary_label = 0 if sym in NORMAL_SYMBOLS else 1
                    patients_data[patient_id].append({
                        'features': feats,
                        'label': binary_label,
                        'original_symbol': sym,
                        'segment': segment,
                        'rr_norm': rr_norm
                    })
            del sig
        return patients_data

    def _stratified_split_patients(self, patients_data):
        patient_ids = list(patients_data.keys())
        np.random.shuffle(patient_ids)
        
        patient_class_counts = {}
        global_counts = np.zeros(2) # 0: Normal, 1: V
        for pid in patient_ids:
            counts = np.zeros(2)
            for item in patients_data[pid]:
                counts[item['label']] += 1
            patient_class_counts[pid] = counts
            global_counts += counts
            
        target_val = np.maximum(2, global_counts * 0.15)
        target_test = np.maximum(2, global_counts * 0.15)
        
        current_val = np.zeros(2)
        current_test = np.zeros(2)
        
        val_ids, test_ids, train_ids = [], [], []
        assigned = set()
        
        def rarity(pid):
            c = patient_class_counts[pid]
            return np.sum(c / (global_counts + 1e-6))
            
        sorted_pids = sorted(patient_ids, key=rarity)
        
        for pid in sorted_pids:
            if pid in assigned: continue
            if np.all(current_val >= target_val): break
            c = patient_class_counts[pid]
            deficit = target_val - current_val
            if np.any((deficit > 0) & (c > 0)):
                val_ids.append(pid)
                current_val += c
                assigned.add(pid)
                
        for pid in sorted_pids:
            if pid in assigned: continue
            if np.all(current_test >= target_test): break
            c = patient_class_counts[pid]
            deficit = target_test - current_test
            if np.any((deficit > 0) & (c > 0)):
                test_ids.append(pid)
                current_test += c
                assigned.add(pid)
                
        for pid in sorted_pids:
            if pid in assigned: continue
            train_ids.append(pid)
            
        return train_ids, val_ids, test_ids

    def _oversample_light(self, base_pool, count):
        oversampled = []
        if not base_pool: return oversampled
        indices = np.random.choice(len(base_pool), count, replace=True)
        for idx in indices:
            item = base_pool[idx]
            segment = item['segment']
            temp_sig = Signal(data=segment, fs=TARGET_FS)
            shifted_sig = temp_sig.time_shift(max_shift_ms=8) 
            amp_factor = np.random.uniform(0.95, 1.05)
            aug_segment = shifted_sig.resampled_data * amp_factor
            feats = self._extract_features(aug_segment, item['rr_norm'])
            oversampled.append({
                'features': feats,
                'label': item['label'],
                'original_symbol': item['original_symbol'],
                'segment': aug_segment, 
                'rr_norm': item['rr_norm']
            })
            del temp_sig, shifted_sig
        return oversampled

    def _augment(self, base_pool, target_count):
        if not base_pool:
            return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int64), np.array([])
        indices = np.random.choice(len(base_pool), size=target_count, replace=True)
        X_aug, y_aug, sym_aug = [], [], []
        for idx in indices:
            item = base_pool[idx]
            segment = item['segment']
            temp_sig = Signal(data=segment, fs=TARGET_FS)
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
            aug_seg = denoised_sig.get_segment(center, SEGMENT_SAMPLES)
            if aug_seg is None:
                continue
            feats = self._extract_features(aug_seg, item['rr_norm'])
            X_aug.append(feats)
            y_aug.append(item['label'])
            sym_aug.append(item['original_symbol'])
            del temp_sig, shifted_sig, augmented_sig, denoised_sig
        if len(X_aug) == 0:
            return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int64), np.array([])
        return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int64), np.array(sym_aug)

    def _balance_split(self, patients_data, patient_list, target_per_class, is_train=False, return_items=False):
        pool = []
        for pid in patient_list:
            pool.extend(patients_data[pid])
        class_data = {0: [], 1: []}
        for item in pool:
            class_data[item['label']].append(item)
            
        final_data = []
        for c in range(2):
            available = len(class_data[c])
            if available == 0: continue
            
            taken = min(available, target_per_class)
            idxs = np.random.choice(available, size=taken, replace=False)
            final_data.extend([class_data[c][i] for i in idxs])
            
            needed = target_per_class - taken
            if needed > 0:
                oversampled = self._oversample_light(class_data[c], needed)
                final_data.extend(oversampled)
            
        np.random.shuffle(final_data)

        if not final_data:
            X = np.zeros((0, NUM_FEATURES), dtype=np.float32)
            y = np.zeros(0, dtype=np.int64)
            sym = np.array([])
        else:
            X = np.array([d['features'] for d in final_data], dtype=np.float32)
            y = np.array([d['label'] for d in final_data], dtype=np.int64)
            sym = np.array([d['original_symbol'] for d in final_data])

        if is_train:
            aug_count_per_class = int(target_per_class * AUGMENTATION_PERCENT)
            X_augs, y_augs, sym_augs = [], [], []
            for c in range(2):
                class_pool = [d for d in final_data if d['label'] == c]
                if not class_pool: continue
                X_aug, y_aug, sym_aug = self._augment(class_pool, aug_count_per_class)
                if len(X_aug) > 0:
                    X_augs.append(X_aug)
                    y_augs.append(y_aug)
                    sym_augs.append(sym_aug)
            if X_augs:
                X = np.concatenate([X] + X_augs, axis=0)
                y = np.concatenate([y] + y_augs, axis=0)
                sym = np.concatenate([sym] + sym_augs, axis=0)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if return_items:
            return X, y, sym, final_data
        else:
            return X, y, sym

    def build(self):
        patients_data = self._collect_data()
        if not patients_data:
            raise ValueError("Нет данных!")
        
        train_ids, val_ids, test_ids = self._stratified_split_patients(patients_data)
        
        print("\n--- Распределение экземпляров по выборкам (до оверсэмплинга) ---")
        for name, ids in [('Train', train_ids), ('Val', val_ids), ('Test', test_ids)]:
            counts = np.zeros(2, dtype=int)
            for pid in ids:
                for item in patients_data[pid]:
                    counts[item['label']] += 1
            print(f"{name:5} | Normal: {counts[0]}, V: {counts[1]}")
        print("-------------------------------------------------------")

        for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            with open(os.path.join(self.splits_dir, f'parametric_{name}.txt'), 'w') as f:
                for pid in ids:
                    f.write(f"{pid}\n")

        X_train, y_train, sym_train = self._balance_split(patients_data, train_ids, TARGET_TRAIN // 2, is_train=True)
        X_val, y_val, sym_val = self._balance_split(patients_data, val_ids, TARGET_VAL // 2, is_train=False)
        X_test, y_test, sym_test, test_items = self._balance_split(patients_data, test_ids, TARGET_TEST // 2, is_train=False, return_items=True)
        self.test_items = test_items

        if len(X_train) == 0:
            raise ValueError("Обучающая выборка оказалась пустой!")
            
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else X_val
        X_test_scaled = self.scaler.transform(X_test) if len(X_test) > 0 else X_test

        import joblib
        joblib.dump(self.scaler, os.path.join('models', 'parametric', 'scaler_pss.pkl'))

        return torch.FloatTensor(X_train_scaled), torch.from_numpy(y_train), \
               torch.FloatTensor(X_val_scaled), torch.from_numpy(y_val), \
               torch.FloatTensor(X_test_scaled), torch.from_numpy(y_test), sym_test

    def generate_noisy_test_set(self, X_test, y_test, sym_test):
        if not self.test_items:
            if len(X_test) == 0:
                return torch.zeros(0, NUM_FEATURES), y_test, sym_test
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 0.1, size=X_test.shape).astype(np.float32)
            return torch.from_numpy(X_test.numpy() + noise), y_test, sym_test

        X_noisy_list, y_noisy_list, sym_noisy_list = [], [], []
        for item in self.test_items:
            segment = item['segment'] 
            temp_sig = Signal(data=segment, fs=TARGET_FS)
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
            aug_seg = denoised_sig.get_segment(center, SEGMENT_SAMPLES)
            if aug_seg is None:
                aug_seg = segment
            feats = self._extract_features(aug_seg, item['rr_norm'])
            X_noisy_list.append(feats)
            y_noisy_list.append(item['label'])
            sym_noisy_list.append(item['original_symbol'])
            del temp_sig, shifted_sig, augmented_sig, denoised_sig
            
        if not X_noisy_list:
            return torch.zeros(0, NUM_FEATURES), torch.zeros(0, dtype=torch.long), np.array([])
            
        X_noisy = np.array(X_noisy_list, dtype=np.float32)
        X_noisy = np.nan_to_num(X_noisy, nan=0.0, posinf=0.0, neginf=0.0)
        X_noisy_scaled = self.scaler.transform(X_noisy)
        
        return torch.FloatTensor(X_noisy_scaled), torch.from_numpy(np.array(y_noisy_list, dtype=np.int64)), np.array(sym_noisy_list)