# -*- coding: utf-8 -*-
"""
Версия 8.0: Фокус исключительно на Экстрасистолиях (Блокады исключены).
Добавлен график распределения подвидов ПЖС (Rule-based).
"""

import os
import sys
import wfdb
import numpy as np
import pandas as pd
import logging
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score,
                             roc_curve, auc)
from scipy import signal
from scipy.spatial.distance import cosine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# --- КОНФИГУРАЦИЯ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("pipeline.log", mode='w', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

DB_ROOT = 'DB'
TARGET_FS = 360
SEGMENT_SAMPLES = 288 
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Блокады полностью исключены из системы!
# Финальные классы (1 базовый + 4 подвида от Rule-based)
FINAL_CLASSES = ['Normal', 'R_on_T', 'PVC_Interpolated', 'PVC_Monomorphic', 'PVC_Polymorphic']

# Базовые классы для TCN (Строго бинарная классификация)
TCN_BASE_CLASSES = ['Normal', 'PVC']
TCN_BASE_TO_IDX = {'Normal': 0, 'PVC': 1}

TARGET_TRAIN = 3500
TARGET_VAL = 750
TARGET_TEST = 750

DB_CONFIG = {'mitdb': ('mitdb', 'atr', 'mit')}

# ==========================================
# БЛОК 1: ПРЕДОБРАБОТКА
# ==========================================
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
        nstdb_path = os.path.join(DB_ROOT, 'nstdb')
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

class DataProcessor:
    def __init__(self):
        self.noise_aug = NoiseAugmenter()
        self.metadata = {'train': [], 'val': [], 'test': []} 

    def process(self):
        logger.info("Сбор данных (Блокады исключены, только Норма и ПЖС)...")
        patients_data = self._collect_all_data()
        if not patients_data: logger.error("Данные не собраны."); sys.exit(1)
        return self._patient_aware_split_and_balance(patients_data)

    def _classify_mit_symbol(self, sym):
        # Блокады ('L', 'R') принудительно считаем нормой, чтобы не путали ПЖС
        if sym == 'N' or sym == '/' or sym == 'L' or sym == 'R': return 0 # Normal
        if sym in ['V', 'E']: return 1 # PVC
        return 0 # Остальное - норма

    def _collect_all_data(self):
        patients_data = defaultdict(list)
        for db_key, (path, ann_ext, db_type) in DB_CONFIG.items():
            full_path = os.path.join(DB_ROOT, path)
            if not os.path.exists(full_path): continue
            files = [f for f in os.listdir(full_path) if f.endswith('.dat')]
            records = sorted(list(set([f.split('.')[0] for f in files])))
            for rec_name in records:
                pid = f"{db_key}_{rec_name}"
                segments = self._process_mit_record(os.path.join(full_path, rec_name), ann_ext)
                if segments:
                    patients_data[pid].extend(segments)
        return patients_data

    def _process_mit_record(self, rec_path, ann_ext):
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, ann_ext) if ann_ext else None
            if not annotation: return []
            idx = 0
            if hasattr(record, 'sig_name'):
                for i, n in enumerate(record.sig_name):
                    if n.lower() in ['ii', 'mlii']: idx = i; break
            ecg = signal.resample(record.p_signal[:, idx], int(len(record.p_signal[:, idx]) * TARGET_FS / record.fs))
            ratio = TARGET_FS / record.fs
            r_peaks = (annotation.sample * ratio).astype(int)
            symbols = [s.upper() for s in annotation.symbol]
            rr_samples = np.diff(r_peaks)
            rr_ms = rr_samples / TARGET_FS * 1000
            rr_mean = np.mean(rr_ms) if len(rr_ms) > 0 else 800

            segments_data = []
            for i, peak in enumerate(r_peaks):
                start = peak - SEGMENT_SAMPLES // 2
                end = peak + SEGMENT_SAMPLES // 2
                if start >= 0 and end < len(ecg):
                    seg = ecg[start:end]
                    if np.std(seg) > 0:
                        seg_norm = (seg - np.mean(seg)) / np.std(seg)
                        segments_data.append({
                            'segment': seg_norm,
                            'base_class': self._classify_mit_symbol(symbols[i]),
                            'rr_prev': rr_ms[i-1] if i > 0 else rr_mean,
                            'rr_next': rr_ms[i] if i < len(rr_ms) else rr_mean,
                            'rr_mean': rr_mean,
                            'raw_morph': seg_norm.copy()
                        })
            return segments_data
        except: return []

    def _patient_aware_split_and_balance(self, patients_data):
        patient_ids = list(patients_data.keys())
        has_pvc, only_normal = [], []
        
        for pid in patient_ids:
            classes = set([s['base_class'] for s in patients_data[pid]])
            if 1 in classes: has_pvc.append(pid) # 1 = PVC
            else: only_normal.append(pid)

        pvc_train, pvc_temp = train_test_split(has_pvc, test_size=0.3, random_state=42)
        pvc_val, pvc_test = train_test_split(pvc_temp, test_size=0.5, random_state=42)
        n_train, n_temp = train_test_split(only_normal, test_size=0.3, random_state=42)
        n_val, n_test = train_test_split(n_temp, test_size=0.5, random_state=42)
        
        train_patients = pvc_train + n_train
        val_patients = pvc_val + n_val
        test_patients = pvc_test + n_test

        def extract_and_balance(patient_list, target_size, split_name):
            pool_data = []
            for pid in patient_list: pool_data.extend(patients_data[pid])
            df = pd.DataFrame(pool_data)
            
            # Разделяем 3500 ровно пополам: 1750 Норма, 1750 ПЖС
            samples_per_class = target_size // 2
            balanced_dfs = []
            
            for cls_idx in [0, 1]:
                subset = df[df['base_class'] == cls_idx]
                if len(subset) == 0:
                    logger.error(f"Класс {TCN_BASE_CLASSES[cls_idx]} отсутствует в {split_name}!")
                    sys.exit(1)
                balanced_dfs.append(subset.sample(samples_per_class, replace=True, random_state=42))
            
            df_bal = pd.concat(balanced_dfs).sample(frac=1, random_state=42).head(target_size)
            X = np.array(df_bal['segment'].tolist()).reshape(-1, 1, SEGMENT_SAMPLES)
            y_base = np.array(df_bal['base_class'].tolist())
            meta = df_bal[['rr_prev', 'rr_next', 'rr_mean', 'raw_morph']].to_dict('records')
            
            X_processed = np.zeros_like(X)
            for i in range(len(X)):
                clean = WaveletDenoiser.denoise(X[i].flatten())
                if split_name == 'train':
                    clean = self.noise_aug.add_noise(clean, snr_db_range=(6, 10))
                clean = (clean - np.mean(clean)) / (np.std(clean) + 1e-8) # Строгая нормализация!
                X_processed[i] = clean.reshape(1, -1)
                
            self.metadata[split_name] = meta
            logger.info(f"{split_name}: {len(patient_list)} пациентов -> {X_processed.shape[0]} сегментов")
            return torch.FloatTensor(X_processed), y_base, meta

        X_train, y_train, meta_train = extract_and_balance(train_patients, TARGET_TRAIN, 'train')
        X_val, y_val, meta_val = extract_and_balance(val_patients, TARGET_VAL, 'val')
        X_test, y_test, meta_test = extract_and_balance(test_patients, TARGET_TEST, 'test')

        return X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test

# ==========================================
# БЛОК 2: ARHITEKTURA TCN
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.padding, 0)))

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
        channels = [64] * 5; dilations = [1, 2, 4, 8, 16]
        layers = []
        for i in range(len(dilations)):
            layers.append(ResidualBlock(1 if i==0 else channels[i-1], channels[i], 5, dilations[i]))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 2) # 2 класса: Норма и ПЖС
    def forward(self, x):
        return self.fc(self.pool(self.network(x)).squeeze(-1))

# ==========================================
# БЛОК 3: РУЧНЫЕ МЕТОДЫ (ТОЛЬКО ДЛЯ ПЖС)
# ==========================================
class HybridRuleRefiner:
    def __init__(self): self.ref = None

    def apply(self, base_preds, meta_list):
        final_preds = []
        for pred_idx, meta in zip(base_preds, meta_list):
            if pred_idx == 0: # Normal
                final_preds.append(FINAL_CLASSES.index('Normal'))
            else: # Это ПЖС (PVC) - запускаем детализацию
                rr_prev, rr_next, rr_mean = meta['rr_prev'], meta['rr_next'], meta['rr_mean']
                qt_est = 0.4 * np.sqrt(rr_mean / 1000) * 1000
                
                if rr_prev < (0.8 * qt_est): final_preds.append(FINAL_CLASSES.index('R_on_T'))
                elif (rr_prev + rr_next) < (2.2 * rr_mean): final_preds.append(FINAL_CLASSES.index('PVC_Interpolated'))
                else:
                    if self.ref is None: self.ref = meta['raw_morph']
                    sim = 1 - cosine(self.ref, meta['raw_morph'])
                    if sim > 0.7: final_preds.append(FINAL_CLASSES.index('PVC_Monomorphic'))
                    else: final_preds.append(FINAL_CLASSES.index('PVC_Polymorphic'))
        return np.array(final_preds)

# ==========================================
# БЛОК 4: ОБУЧЕНИЕ
# ==========================================
def train_model(X_train, y_train, X_val, y_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    best_loss, patience, history = float('inf'), 15, {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
    loader = DataLoader(torch.utils.data.TensorDataset(X_train, torch.LongTensor(y_train)), batch_size=64, shuffle=True)

    for epoch in range(100):
        model.train(); t_loss, t_corr, t_tot = 0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); out = model(bx); loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item(); t_corr += (out.argmax(1)==by).sum().item(); t_tot += by.size(0)
            
        model.eval()
        with torch.no_grad():
            v_out = model(X_val.to(device))
            v_loss = criterion(v_out, torch.LongTensor(y_val).to(device)).item()
            v_acc = (v_out.argmax(1).cpu() == torch.LongTensor(y_val)).float().mean().item()

        history['loss'].append(t_loss/len(loader)); history['acc'].append(t_corr/t_tot)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        
        if v_loss < best_loss: best_loss = v_loss; torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_tcn.pth'))
        else: 
            patience -= 1
            if patience == 0: break
    return model, history

# ==========================================
# БЛОК 5: ВИЗУАЛИЗАЦИЯ
# ==========================================
def plot_dashboard(X_train_np, model, history, X_test, y_test_base, meta_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Генерация графиков...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Вейвлет
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    clean = X_train_np[0, 0, :]
    noisy = NoiseAugmenter().add_noise(clean, snr_db_range=(6, 10))
    denoised = WaveletDenoiser.denoise(noisy)
    axes[0, 0].plot(clean, 'g'); axes[0, 0].set_title("1. Чистый сигнал (Норма)")
    axes[0, 1].plot(noisy, 'r'); axes[0, 1].set_title("2. Зашумленный (NSTDB, спорт)")
    axes[1, 0].plot(denoised, 'b'); axes[1, 0].set_title("3. После ДВП фильтрации")
    axes[1, 1].plot(clean, 'g', alpha=0.5, label='Оригинал')
    axes[1, 1].plot(denoised, 'b', alpha=0.8, label='ДВП')
    axes[1, 1].set_title("4. Сравнение Оригинал vs Восстановленный")
    for ax in axes.flat: ax.set_xlabel("Отсчеты"); ax.set_ylabel("Амплитуда")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '1_wavelet.png'), dpi=150); plt.close()

    # Предсказания TCN
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_tcn.pth'))); model.eval()
    with torch.no_grad(): logits = model(X_test.to(device)); base_preds = np.argmax(logits.cpu().numpy(), axis=1)
    
    # Ручные алгоритмы
    refiner = HybridRuleRefiner(); refiner.ref = None
    y_final = refiner.apply(base_preds, meta_test)

    # 2. Примеры QRS
    found_classes_idx = np.unique(y_final)
    num_found = len(found_classes_idx)
    fig, axes = plt.subplots(num_found, 5, figsize=(20, 3.5 * num_found))
    if num_found == 1: axes = axes.reshape(1, -1)
    plot_idx = 0
    for cls_idx in found_classes_idx:
        cls_name = FINAL_CLASSES[cls_idx]
        indices = np.where(y_final == cls_idx)[0]
        samples = np.random.choice(indices, min(5, len(indices)), replace=False)
        for j, idx in enumerate(samples):
            ax = axes[plot_idx, j]
            ax.plot(X_test[idx, 0, :].cpu().numpy(), 'k', linewidth=1.2)
            ax.axis('off')
            ax.set_title(f"{cls_name}", fontsize=16, fontweight='bold', pad=10, color='darkred')
        plot_idx += 1
    plt.suptitle("Примеры финальной классификации", fontsize=18, y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '2_qrs_examples.png'), dpi=150, bbox_inches='tight'); plt.close()

    # 3. Обучение
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['loss'], label='Train'); ax1.plot(history['val_loss'], label='Val'); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history['acc'], label='Train'); ax2.plot(history['val_acc'], label='Val'); ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '3_training.png'), dpi=150); plt.close()

    # ОЦЕНКА БАЗОВОЙ TCN
    print("\n" + "="*60)
    print("ОТЧЕТ: Нейросеть (Норма vs ПЖС)")
    print("="*60)
    print(classification_report(y_test_base, base_preds, target_names=TCN_BASE_CLASSES, zero_division=0))

    # 4. Confusion Matrix (2x2)
    cm = confusion_matrix(y_test_base, base_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TCN_BASE_CLASSES, yticklabels=TCN_BASE_CLASSES)
    plt.title("Confusion Matrix: Норма vs ПЖС"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '4_cm_base.png'), dpi=150); plt.close()

    # 5. ROC-AUC (Для бинарной классификации label_binarize не нужен)
    plt.figure(figsize=(7, 7))
    # y_test_base содержит 0 и 1. logits[:, 1] содержит вероятность класса ПЖС (индекс 1)
    fpr, tpr, _ = roc_curve(y_test_base, logits.cpu().numpy()[:, 1])
    plt.plot(fpr, tpr, label=f'PVC vs Normal (AUC = {auc(fpr, tpr):.2f})')

    # 6. Столбцы метрик (2 класса)
    prec = precision_score(y_test_base, base_preds, average=None, zero_division=0)
    rec = recall_score(y_test_base, base_preds, average=None, zero_division=0)
    f1 = f1_score(y_test_base, base_preds, average=None, zero_division=0)
    spec = []
    for i in range(2):
        tn = np.sum((y_test_base != i) & (base_preds != i))
        fp = np.sum((y_test_base != i) & (base_preds == i))
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    x = np.arange(2); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - 1.5*w, prec, w, label='Precision'); ax.bar(x - 0.5*w, rec, w, label='Recall')
    ax.bar(x + 0.5*w, f1, w, label='F1-Score'); ax.bar(x + 1.5*w, spec, w, label='Specificity')
    ax.set_xticks(x); ax.set_xticklabels(TCN_BASE_CLASSES); ax.set_ylim(0, 1.1); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '6_metrics.png'), dpi=150); plt.close()

        # ==========================================
    # 7. СРАВНЕНИЕ: Истинные подвиды vs Найденные подвиды
    # ==========================================
    logger.info("  [7/7] Сравнение распределения подвидов ПЖС...")
    
    pvc_subtypes = ['R_on_T', 'PVC_Interpolated', 'PVC_Monomorphic', 'PVC_Polymorphic']
    subtype_indices = [FINAL_CLASSES.index(name) for name in pvc_subtypes]
    
    # 1. Считаем подвиды, которые НАШЛА нейросеть (и обработали ручные алгоритмы)
    pred_counts = [np.sum(y_final == idx) for idx in subtype_indices]
    
    # 2. Считаем подвиды внутри ИСТИННЫХ ПЖС (которые разметили врачи как 'V')
    # Мы берем только те сегменты, где врач поставил 1 (ПЖС), 
    # и смотрим, как бы наши RR-алгоритмы их классифицировали.
    true_refiner = HybridRuleRefiner()
    y_true_math_subtypes = true_refiner.apply(y_test_base, meta_test) 
    true_counts = [np.sum(y_true_math_subtypes == idx) for idx in subtype_indices]

    # Построение сдвоенной столбчатой диаграммы
    x = np.arange(len(pvc_subtypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    # Исправленные, честные подписи
    bars1 = ax.bar(x - width/2, true_counts, width, label='Математическая разбивка ИСТИННЫХ ПЖС (из базы)', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Математическая разбивка НАЙДЕННЫХ ПЖС (нейросетью)', color='#d62728', edgecolor='black')

    ax.set_title("Сравнение структуры подвидов экстрасистолий\n(Истинная выборка врачей vs Выборка нейросети)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Количество комплексов", fontsize=12)
    ax.set_xlabel("Подвид Экстрасистолии (Вычислено по RR-интервалам)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pvc_subtypes, rotation=15, ha='right')
    ax.legend(fontsize=11)

    # Добавляем цифры
    for bar in bars1:
        yval = bar.get_height()
        if yval > 0: plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold', color='#1f77b4')
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0: plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold', color='#d62728')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '7_pvc_subtypes_comparison.png'), dpi=150)
    plt.close()

    print("\n" + "="*60)
    print("ОТЧЕТ: Проверка структуры найденных ПЖС относительно истинных")
    print("="*60)
    print(f"{'Подвид ПЖС':<20} | {'Истинных (в базе)':<18} | {'Найдено (TCN+Rules)':<18}")
    print("-" * 60)
    for name, true_c, pred_c in zip(pvc_subtypes, true_counts, pred_counts):
        print(f"{name:<20} | {true_c:<18} | {pred_c:<18}")
        
    logger.info("Все 7 графиков сохранены.")

def main():
    import warnings; warnings.filterwarnings("ignore")
    processor = DataProcessor()
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = processor.process()
    model, history = train_model(X_train, y_train, X_val, y_val)
    plot_dashboard(X_train.numpy(), model, history, X_test, y_test, meta_test)

if __name__ == "__main__":
    main()