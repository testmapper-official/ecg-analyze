# -*- coding: utf-8 -*-
"""ЭТАП 1: Обучение TCN классификатора (Морфология: Норма, Блокада, ПЖС)"""

import os, sys, wfdb, logging, pywt, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import signal
from collections import defaultdict
import archive.config as config

logger = logging.getLogger(__name__)

class WaveletDenoiser:
    @staticmethod
    def denoise(sig, wavelet='db6', level=5):
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
        coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)[:len(sig)]

class NoiseAugmenter:
    def __init__(self):
        self.noise_cache = {}
        nstdb_path = os.path.join(config.DB_ROOT, 'nstdb')
        if os.path.exists(nstdb_path):
            for noise_type in ['ma', 'em']:
                try:
                    record = wfdb.rdrecord(os.path.join(nstdb_path, noise_type))
                    self.noise_cache[noise_type] = record.p_signal[:, 0]
                except: pass

    def add_noise(self, clean_signal, snr_db_range=(6, 10)):
        if not self.noise_cache: return clean_signal
        noise_type = np.random.choice(list(self.noise_cache.keys()))
        noise_base = self.noise_cache[noise_type]
        start_idx = np.random.randint(0, len(noise_base) - len(clean_signal))
        noise_frag = noise_base[start_idx : start_idx + len(clean_signal)]
        signal_rms = np.sqrt(np.mean(clean_signal ** 2))
        noise_rms = np.sqrt(np.mean(noise_frag ** 2))
        if noise_rms == 0: return clean_signal
        snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
        target_noise_rms = signal_rms / (10 ** (snr_db / 20))
        return clean_signal + noise_frag * (target_noise_rms / noise_rms)

# --- АРХИТЕКТУРА TCN (Оригинальная) ---
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x): return self.conv(nn.functional.pad(x, (self.padding, 0)))

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_out, n_out, kernel_size, dilation)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.relu = nn.ReLU(); self.drop = nn.Dropout(0.3)
    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x)))
        return self.relu(out + self.downsample(x))

class TCNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [64] * 6; dilations = [1, 2, 4, 8, 16, 32]
        layers = []
        for i in range(len(dilations)):
            layers.append(ResidualBlock(1 if i==0 else channels[i-1], channels[i], 5, dilations[i]))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], len(config.TCN_CLASSES))
    def forward(self, x): return self.fc(self.pool(self.network(x)).squeeze(-1))

# --- ЗАГРУЗЧИК ДАННЫХ ---
class DataProcessor:
    def __init__(self): self.noise_aug = NoiseAugmenter()

    def process(self):
        logger.info("Сбор данных (Этап 1)...")
        patients_data = self._collect_all_data()
        if not patients_data: sys.exit(1)
        return self._split_and_balance(patients_data)

    def _collect_all_data(self):
        patients_data = defaultdict(list)
        full_path = os.path.join(config.DB_ROOT, 'mitdb')
        if not os.path.exists(full_path): return patients_data
        files = [f for f in os.listdir(full_path) if f.endswith('.dat')]
        records = sorted(list(set([f.split('.')[0] for f in files])))
        
        for rec_name in records:
            try:
                record = wfdb.rdrecord(os.path.join(full_path, rec_name))
                annotation = wfdb.rdann(os.path.join(full_path, rec_name), 'atr')
                idx = 0
                if hasattr(record, 'sig_name'):
                    for i, n in enumerate(record.sig_name):
                        if n.lower() in ['ii', 'mlii']: idx = i; break
                
                ecg = signal.resample(record.p_signal[:, idx], int(len(record.p_signal[:, idx]) * config.TARGET_FS / record.fs))
                ratio = config.TARGET_FS / record.fs
                r_peaks = (annotation.sample * ratio).astype(int)
                symbols = [s.upper() for s in annotation.symbol]
                rr_ms = np.diff(r_peaks) / config.TARGET_FS * 1000

                for i, peak in enumerate(r_peaks):
                    start = peak - config.SEGMENT_SAMPLES // 2
                    end = peak + config.SEGMENT_SAMPLES // 2
                    if start >= 0 and end < len(ecg):
                        seg = ecg[start:end]
                        if np.std(seg) > 0:
                            seg_norm = (seg - np.mean(seg)) / np.std(seg)
                            window_rr = rr_ms[max(0, i-5):i]
                            local_rr_mean = np.mean(window_rr) if len(window_rr) > 0 else 800
                            rr_prev = rr_ms[i-1] if i > 0 else local_rr_mean
                            rr_ratio = rr_prev / local_rr_mean if local_rr_mean > 0 else 1.0
                            
                            label = config.TCN_TO_IDX['Normal']
                            if symbols[i] in ['L', 'R']: label = config.TCN_TO_IDX['Blockade']
                            elif symbols[i] in ['V', 'E']: label = config.TCN_TO_IDX['PVC']

                            patients_data[f"mit_{rec_name}"].append({
                                'segment': seg_norm, 'label': label,
                                'rr_prev': rr_prev, 'rr_mean': local_rr_mean, 'rr_ratio': rr_ratio
                            })
            except Exception as e: pass
        return patients_data

    def _split_and_balance(self, patients_data):
        # Разделение пациентов
        has_block, has_pvc, only_n = [], [], []
        for pid, segs in patients_data.items():
            cls = set(s['label'] for s in segs)
            if config.TCN_TO_IDX['Blockade'] in cls: has_block.append(pid)
            elif config.TCN_TO_IDX['PVC'] in cls: has_pvc.append(pid)
            else: only_n.append(pid)

        b_tr, b_tmp = train_test_split(has_block, test_size=0.35, random_state=42)
        b_v, b_te = train_test_split(b_tmp, test_size=0.5, random_state=42)
        p_tr, p_tmp = train_test_split(has_pvc, test_size=0.3, random_state=42)
        p_v, p_te = train_test_split(p_tmp, test_size=0.5, random_state=42)
        n_tr, n_tmp = train_test_split(only_n, test_size=0.3, random_state=42)
        n_v, n_te = train_test_split(n_tmp, test_size=0.5, random_state=42)

        def extract(pids, size, name):
            pool = [s for p in pids for s in patients_data[p]]
            df = pd.DataFrame(pool)
            dfs = [df[df['label']==c].sample(size//3, replace=True, random_state=42) for c in config.TCN_TO_IDX.values()]
            df_bal = pd.concat(dfs).sample(frac=1, random_state=42).head(size)
            
            X = np.zeros((len(df_bal), 1, config.SEGMENT_SAMPLES))
            for i, row in enumerate(df_bal.itertuples()):
                clean = WaveletDenoiser.denoise(np.array(row.segment))
                if name == 'train': clean = self.noise_aug.add_noise(clean)
                clean = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
                X[i, 0, :] = clean
            
            meta = df_bal[['rr_prev', 'rr_mean', 'rr_ratio']].to_dict('records')
            logger.info(f"Выборка {name}: {len(df_bal)} (Цель: {size})")
            return torch.FloatTensor(X), np.array(df_bal['label'].tolist()), meta

        X_tr, y_tr, m_tr = extract(b_tr+p_tr+n_tr, config.TARGET_TRAIN, 'train')
        X_v, y_v, m_v = extract(b_v+p_v+n_v, config.TARGET_VAL, 'val')
        X_te, y_te, m_te = extract(b_te+p_te+n_te, config.TARGET_TEST, 'test')
        return X_tr, y_tr, m_tr, X_v, y_v, m_v, X_te, y_te, m_te

def train_tcn():
    proc = DataProcessor()
    X_tr, y_tr, _, X_v, y_v, _, X_te, y_te, meta_te = proc.process()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loader = DataLoader(TensorDataset(X_tr, torch.LongTensor(y_tr)), batch_size=64, shuffle=True)
    best_loss, patience = float('inf'), 15
    
    logger.info(f"Обучение TCN на {device}...")
    for epoch in range(100):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); loss = criterion(model(bx), by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_v.to(device)), torch.LongTensor(y_v).to(device)).item()
        
        if v_loss < best_loss:
            best_loss = v_loss; torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, 'stage1_tcn.pth'))
        else:
            patience -= 1
            if patience == 0: break
            
    logger.info("Обучение завершено.")
    return model, X_te, y_te, meta_te