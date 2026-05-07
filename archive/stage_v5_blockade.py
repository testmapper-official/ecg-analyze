# -*- coding: utf-8 -*-
"""
ПЛАГИН V5: Строго Норма vs Блокада. 
ПЖС полностью исключены из логики плагина.
Train: MIT + PTB | Val/Test: INCART (разделение по пациентам)
"""

import os, sys, wfdb, logging, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import signal
from collections import defaultdict # ДОБАВЛЕН ПРОПУЩЕННЫЙ ИМПОРТ
import neurokit2 as nk
import archive.config as config

logger = logging.getLogger(__name__)

class V5Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        # Выход равен 2 (Normal, Blockade)
        self.fc = nn.Linear(64 * (config.SEGMENT_SAMPLES // 4), len(config.V5_CLASSES))
        
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x)
        return self.fc(x.view(x.size(0), -1))

class V5DataProcessor:
    def __init__(self):
        self.target_fs = config.TARGET_FS
        self.seg_len = config.SEGMENT_SAMPLES

    def process(self):
        logger.info("Сбор данных для модуля V5 (Строго Норма/Блокада)...")
        
        train_data = []
        val_data = []
        test_data = []

        # --- 1. MIT-BIH (Только Норма и Блокады, без ПЖС!) ---
        mit_path = os.path.join(config.DB_ROOT, 'mitdb')
        if os.path.exists(mit_path):
            logger.info("Парсинг MIT-BIH (V5)...")
            for rec_name in self._get_records(mit_path):
                segs = self._parse_mit_ptb(os.path.join(mit_path, rec_name), db_type='mit')
                if segs: train_data.extend([(s, l) for s, l in segs])

        # --- 2. PTB (Блокады) ---
        ptb_path = os.path.join(config.DB_ROOT, 'ptbdb')
        if os.path.exists(ptb_path):
            logger.info("Парсинг PTB (V5)...")
            for p_folder in sorted([f for f in os.listdir(ptb_path) if os.path.isdir(os.path.join(ptb_path, f))]):
                for rec_name in self._get_records(os.path.join(ptb_path, p_folder)):
                    segs = self._parse_mit_ptb(os.path.join(ptb_path, p_folder, rec_name), db_type='ptb')
                    if segs: train_data.extend([(s, l) for s, l in segs])

        # --- 3. INCART (Валидация и Тест - разбиваем по пациентам) ---
        incart_path = os.path.join(config.DB_ROOT, 'incartdb/files')
        if os.path.exists(incart_path):
            logger.info("Парсинг INCART (V5)...")
            incart_patients_data = defaultdict(list)
            for rec_name in self._get_records(incart_path):
                segs = self._parse_mit_ptb(os.path.join(incart_path, rec_name), db_type='incart')
                if segs:
                    patient_id = rec_name.split('_')[0] 
                    incart_patients_data[patient_id].extend(segs)
            
            patients = list(incart_patients_data.keys())
            val_p, test_p = train_test_split(patients, test_size=0.5, random_state=42)
            
            for p in val_p: val_data.extend([(s, l) for s, l in incart_patients_data[p]])
            for p in test_p: test_data.extend([(s, l) for s, l in incart_patients_data[p]])

        if not train_data or not val_data or not test_data:
            logger.error("Недостаточно данных для обучения V5.")
            return None

        return self._balance_and_tensor(train_data, val_data, test_data)

    def _get_records(self, path):
        files = [f for f in os.listdir(path) if f.endswith('.dat')]
        return sorted(list(set([f.split('.')[0] for f in files])))

    def _find_v5_idx(self, record):
        if hasattr(record, 'sig_name'):
            for i, n in enumerate(record.sig_name):
                if n.lower() == 'v5': return i
            for i, n in enumerate(record.sig_name):
                if n.lower().startswith('v') and len(n) <= 3: return i
        return 0

    def _parse_mit_ptb(self, rec_path, db_type, ann_ext='atr'):
        try:
            record = wfdb.rdrecord(rec_path)
            idx = self._find_v5_idx(record)
            ecg = signal.resample(record.p_signal[:, idx], int(len(record.p_signal[:, idx]) * self.target_fs / record.fs))
            r_peaks = []
            
            if db_type == 'ptb':
                _, info = nk.ecg_peaks(ecg, sampling_rate=self.target_fs)
                r_peaks = info['ECG_R_Peaks']
                header = wfdb.rdheader(rec_path)
                text = " ".join(header.comments).lower()
                base_label = config.V5_TO_IDX['Normal']
                if 'lbbb' in text or 'rbbb' in text: base_label = config.V5_TO_IDX['Blockade']
                labels = [base_label] * len(r_peaks)
            else:
                if not os.path.exists(f"{rec_path}.{ann_ext}"): return []
                annotation = wfdb.rdann(rec_path, ann_ext)
                ratio = self.target_fs / record.fs
                r_peaks = (annotation.sample * ratio).astype(int)
                symbols = [s.upper() for s in annotation.symbol]
                
                labels = []
                for sym in symbols:
                    if sym in ['V', 'E', '!']: 
                        continue 
                    elif sym in ['L', 'R']: 
                        labels.append(config.V5_TO_IDX['Blockade'])
                    else: 
                        labels.append(config.V5_TO_IDX['Normal'])

            segments = []
            for i, peak in enumerate(r_peaks):
                if i >= len(labels): break 
                
                start = peak - self.seg_len // 2
                end = peak + self.seg_len // 2
                if start >= 0 and end < len(ecg):
                    seg = ecg[start:end]
                    if np.std(seg) > 0:
                        seg_norm = (seg - np.mean(seg)) / np.std(seg)
                        segments.append((seg_norm, labels[i]))
            return segments
        except Exception as e:
            return []

    def _balance_and_tensor(self, train_raw, val_raw, test_raw):
        def transform(raw_list, size, name):
            df = pd.DataFrame(raw_list, columns=['segment', 'label'])
            balanced = []
            samples_per_cls = size // len(config.V5_CLASSES)
            
            for cls_idx in config.V5_CLASSES:
                subset = df[df['label'] == config.V5_TO_IDX[cls_idx]]
                if len(subset) == 0:
                    subset = df.sample(samples_per_cls, replace=True, random_state=42)
                balanced.append(subset.sample(samples_per_cls, replace=(len(subset) < samples_per_cls), random_state=42))
            
            df_bal = pd.concat(balanced).sample(frac=1, random_state=42).head(size)
            X = np.zeros((len(df_bal), 1, self.seg_len))
            for i, row in enumerate(df_bal.itertuples()):
                X[i, 0, :] = row.segment
            y = np.array(df_bal['label'].tolist())
            logger.info(f"V5 Выборка {name}: {len(df_bal)} сегментов (Классы: {dict(df_bal['label'].value_counts())})")
            return torch.FloatTensor(X), torch.LongTensor(y)

        X_tr, y_tr = transform(train_raw, 2500, 'Train (MIT+PTB)')
        X_v, y_v = transform(val_raw, 600, 'Val (INCART)')
        X_te, y_te = transform(test_raw, 600, 'Test (INCART)')
        return X_tr, y_tr, X_v, y_v, X_te, y_te


def train_v5_plugin():
    proc = V5DataProcessor()
    data = proc.process()
    if data is None: return None, None, None
        
    X_tr, y_tr, X_v, y_v, X_te, y_te = data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = V5Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    best_loss, patience = float('inf'), 10
    
    logger.info(f"Обучение плагина V5 на {device}...")
    for epoch in range(50):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_v.to(device)), y_v.to(device)).item()
            
        if v_loss < best_loss:
            best_loss = v_loss; torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, 'v5_plugin.pth'))
        else:
            patience -= 1
            if patience == 0: break
            
    logger.info("Обучение модуля V5 завершено.")
    return model, X_te, y_te