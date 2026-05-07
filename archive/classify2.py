# -*- coding: utf-8 -*-
"""
Версия 7.0: Строгое разбиение по пациентам, исправленная логика Rule-Based, корректная визуализация.
"""

import os
import sys
import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
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
from sklearn.preprocessing import label_binarize
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

# Финальные классы (6 штук)
FINAL_CLASSES = ['Normal', 'Blockade', 'R_on_T', 'PVC_Interpolated', 'PVC_Monomorphic', 'PVC_Polymorphic']

# Базовые классы для TCN (3 штуки)
TCN_BASE_CLASSES = ['Normal', 'Blockade', 'PVC']
TCN_BASE_TO_IDX = {name: i for i, name in enumerate(TCN_BASE_CLASSES)}

TARGET_TRAIN = 3500
TARGET_VAL = 750
TARGET_TEST = 750

DB_CONFIG = {'mitdb': ('mitdb', 'atr', 'mit')}

# ==========================================
# БЛОК 1: ПРЕДОБРАБОТКА И СБОР ПО ПАЦИЕНТАМ
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
        self._load_nstdb()

    def _load_nstdb(self):
        nstdb_path = os.path.join(DB_ROOT, 'nstdb')
        if not os.path.exists(nstdb_path): return
        for noise_type in ['ma', 'em']:
            try:
                record = wfdb.rdrecord(os.path.join(nstdb_path, noise_type))
                self.noise_cache[noise_type] = record.p_signal[:, 0]
            except: pass

    def add_noise(self, clean_signal, snr_db_range=(3, 7)):
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
        logger.info("Сбор данных с привязкой к пациентам...")
        # Теперь собираем данные в формате {patient_id: [сегменты]}
        patients_data = self._collect_all_data()
        if not patients_data: logger.error("Данные не собраны."); sys.exit(1)
        return self._patient_aware_split_and_balance(patients_data)

    def _classify_mit_symbol(self, sym):
        # L и R объединены в общую Блокаду
        if sym == 'N' or sym == '/': return TCN_BASE_TO_IDX['Normal']
        if sym in ['L', 'R']: return TCN_BASE_TO_IDX['Blockade']
        if sym in ['V', 'E']: return TCN_BASE_TO_IDX['PVC']
        return TCN_BASE_TO_IDX['Normal']

    def _collect_all_data(self):
        patients_data = defaultdict(list)
        for db_key, (path, ann_ext, db_type) in DB_CONFIG.items():
            full_path = os.path.join(DB_ROOT, path)
            if not os.path.exists(full_path): continue
            
            files = [f for f in os.listdir(full_path) if f.endswith('.dat')]
            records = sorted(list(set([f.split('.')[0] for f in files])))
            
            for rec_name in records:
                # ID пациента = имя записи (например, '100', '101')
                patient_id = f"{db_key}_{rec_name}"
                segments = self._process_mit_record(os.path.join(full_path, rec_name), ann_ext)
                if segments:
                    patients_data[patient_id].extend(segments)
                    counts = Counter([s['base_class'] for s in segments])
                    logger.info(f"Пациент {patient_id}: {len(segments)} сегментов. Классы: {dict(counts)}")
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
            # --- ЗАЩИТА ОТ ЗАГРЯЗНЕНИЯ КЛАССОВ ---
            # Если у пациента есть Блокада, мы отбрасываем его ПЖС, 
            # так как форма ПЖС на фоне блокады не типична и собьет нейросеть с толку.
            classes_in_rec = set([s['base_class'] for s in segments_data])
            if TCN_BASE_TO_IDX['Blockade'] in classes_in_rec:
                segments_data = [s for s in segments_data if s['base_class'] != TCN_BASE_TO_IDX['PVC']]
                
            return segments_data
        except: return []

    def _patient_aware_split_and_balance(self, patients_data):
        patient_ids = list(patients_data.keys())
        
        # Разделяем пациентов на "корзины" в зависимости от того, какие редкие классы у них есть
        has_blockade = []
        has_pvc = []
        only_normal = []
        
        for pid in patient_ids:
            classes = set([s['base_class'] for s in patients_data[pid]])
            if TCN_BASE_TO_IDX['Blockade'] in classes:
                has_blockade.append(pid)
            elif TCN_BASE_TO_IDX['PVC'] in classes:
                has_pvc.append(pid)
            else:
                only_normal.append(pid)
                
        logger.info(f"Найдено пациентов: с Блокадами - {len(has_blockade)}, с ПЖС - {len(has_pvc)}, только Норма - {len(only_normal)}")

        # Разбиваем каждую "корзину" пациентов независимо (70/15/15)
        # Это 100% гарантирует наличие редких классов во всех трех выборках!
        b_train, b_temp = train_test_split(has_blockade, test_size=0.35, random_state=42)
        b_val, b_test = train_test_split(b_temp, test_size=0.5, random_state=42)
        
        pvc_train, pvc_temp = train_test_split(has_pvc, test_size=0.3, random_state=42)
        pvc_val, pvc_test = train_test_split(pvc_temp, test_size=0.5, random_state=42)
        
        n_train, n_temp = train_test_split(only_normal, test_size=0.3, random_state=42)
        n_val, n_test = train_test_split(n_temp, test_size=0.5, random_state=42)
        
        # Объединяем пациентов в финальные списки для выборок
        train_patients = b_train + pvc_train + n_train
        val_patients = b_val + pvc_val + n_val
        test_patients = b_test + pvc_test + n_test

        def extract_and_balance(patient_list, target_size, split_name):
            # Собираем все сегменты выбранных пациентов
            pool_data = []
            for pid in patient_list:
                pool_data.extend(patients_data[pid])
            
            df = pd.DataFrame(pool_data)
            balanced_dfs = []
            samples_per_class = target_size // len(TCN_BASE_CLASSES)
            
            for cls_idx in TCN_BASE_TO_IDX.values():
                subset = df[df['base_class'] == cls_idx]
                if len(subset) == 0:
                    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Класс {TCN_BASE_CLASSES[cls_idx]} отсутствует в {split_name}!")
                    sys.exit(1)
                # Дублируем (replace=True), так как сегментов одного пациента может не хватить для 3500
                balanced_dfs.append(subset.sample(samples_per_class, replace=True, random_state=42))
            
            df_bal = pd.concat(balanced_dfs).sample(frac=1, random_state=42).head(target_size)
            
            # Формат PyTorch (Batch, 1 Channel, 288 Length)
            X = np.array(df_bal['segment'].tolist()).reshape(-1, 1, SEGMENT_SAMPLES)
            y_base = np.array(df_bal['base_class'].tolist())
            meta = df_bal[['rr_prev', 'rr_next', 'rr_mean', 'raw_morph']].to_dict('records')
            
            # Аугментация (ДВП + NSTDB шум) применяется только к Train
            X_processed = np.zeros_like(X)
            for i in range(len(X)):
                clean = WaveletDenoiser.denoise(X[i].flatten())
                if split_name == 'train':
                    clean = self.noise_aug.add_noise(clean, snr_db_range=(6, 10))
                # КРИТИЧЕСКИ ВАЖНО: Ренормализация после шума! Иначе Loss взрывается, а Блокады путаются с ПЖС
                clean = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
                X_processed[i] = clean.reshape(1, -1)
                
            self.metadata[split_name] = meta
            logger.info(f"Выборка {split_name}: {len(patient_list)} пациентов -> {X_processed.shape[0]} сегментов (Цель: {target_size})")
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
        channels = [64] * 5  # Было 6 слоев, стало 5
        dilations = [1, 2, 4, 8, 16] # Убрали 32
        layers = []
        for i in range(len(dilations)):
            layers.append(ResidualBlock(1 if i==0 else channels[i-1], channels[i], 5, dilations[i]))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], len(TCN_BASE_CLASSES))
    def forward(self, x):
        return self.fc(self.pool(self.network(x)).squeeze(-1))

# ==========================================
# БЛОК 3: РУЧНЫЕ МЕТОДЫ (ТОЛЬКО ДЛЯ ЭКСТРАСИСТОЛ)
# ==========================================
class HybridRuleRefiner:
    def __init__(self):
        self.ref = None

    def apply(self, base_preds, meta_list):
        final_preds = []
        for pred_idx, meta in zip(base_preds, meta_list):
            cls_name = TCN_BASE_CLASSES[pred_idx]
            
            if cls_name == 'Normal': final_preds.append(FINAL_CLASSES.index('Normal'))
            elif cls_name == 'Blockade': final_preds.append(FINAL_CLASSES.index('Blockade')) # Общая блокада
            
            elif cls_name == 'PVC':
                rr_prev, rr_next, rr_mean = meta['rr_prev'], meta['rr_next'], meta['rr_mean']
                raw = meta['raw_morph']
                
                # R-on-T
                qt_est = 0.4 * np.sqrt(rr_mean / 1000) * 1000
                if rr_prev < (0.8 * qt_est):
                    final_preds.append(FINAL_CLASSES.index('R_on_T'))
                # Интерполированная
                elif (rr_prev + rr_next) < (2.2 * rr_mean):
                    final_preds.append(FINAL_CLASSES.index('PVC_Interpolated'))
                else:
                    # Моно/Полиморфная (порог 0.7 вместо 0.9, чтобы не было 0)
                    if self.ref is None: self.ref = raw
                    sim = 1 - cosine(self.ref, raw)
                    if sim > 0.7: final_preds.append(FINAL_CLASSES.index('PVC_Monomorphic'))
                    else: final_preds.append(FINAL_CLASSES.index('PVC_Polymorphic'))
        return np.array(final_preds)

# ==========================================
# БЛОК 4: ОБУЧЕНИЕ
# ==========================================
def train_model(X_train, y_train, X_val, y_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNClassifier().to(device)
    
    # Убираем веса классов! Данные уже сбалансированы 1/3, веса вызывали взрыв Loss
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Снижен LR с 0.001 до 0.0005
    
    best_loss, patience, history = float('inf'), 15, {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
    loader = DataLoader(torch.utils.data.TensorDataset(X_train, torch.LongTensor(y_train)), batch_size=64, shuffle=True)

    logger.info(f"Обучение TCN на {device} (до 100 эпох)...")
    for epoch in range(100):
        model.train(); t_loss, t_corr, t_tot = 0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); out = model(bx); loss = criterion(out, by)
            loss.backward()
            # Gradient Clipping - предотвращает резкие скачки Loss и помогает отличать Блокады от ПЖС
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
        
        if v_loss < best_loss: 
            best_loss = v_loss; torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_tcn.pth'))
        else: 
            patience -= 1
            if patience == 0: logger.info(f"Early stopping на эпохе {epoch+1}"); break
    return model, history

# ==========================================
# БЛОК 5: ВИЗУАЛИЗАЦИЯ
# ==========================================
def plot_dashboard(X_train_np, model, history, X_test, y_test_base, meta_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Генерация графиков...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Демонстрация Вейвлет-денойза и Шумов (Явное сравнение)
    logger.info("  [1/6] Демонстрация фильтрации...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    clean = X_train_np[0, 0, :]
    noisy = NoiseAugmenter().add_noise(clean, snr_db_range=(3, 5))
    denoised = WaveletDenoiser.denoise(noisy)
    
    axes[0, 0].plot(clean, 'g'); axes[0, 0].set_title("1. Чистый исходный сигнал (Норма)")
    axes[0, 1].plot(noisy, 'r'); axes[0, 1].set_title("2. Зашумленный (NSTDB, мышечная активность)")
    axes[1, 0].plot(denoised, 'b'); axes[1, 0].set_title("3. После ДВП фильтрации (Восстановленный)")
    # Показываем разницу
    axes[1, 1].plot(clean, 'g', alpha=0.5, label='Оригинал')
    axes[1, 1].plot(denoised, 'b', alpha=0.8, label='После ДВП')
    axes[1, 1].set_title("4. Наложение Оригинал vs Восстановленный")
    for ax in axes.flat: ax.set_xlabel("Отсчеты (288 = 800 мс)"); ax.set_ylabel("Амплитуда (мВ)")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '1_wavelet_comparison.png'), dpi=150); plt.close()

    # Предсказания
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_tcn.pth'))); model.eval()
    with torch.no_grad(): logits = model(X_test.to(device)); base_preds = np.argmax(logits.cpu().numpy(), axis=1)
    
    # Ручные алгоритмы (сбрасываем референс перед тестом)
    refiner = HybridRuleRefiner(); refiner.ref = None
    y_final = refiner.apply(base_preds, meta_test)

    # 2. Примеры QRS (ЯВНЫЕ НАЗВАНИЯ КЛАССОВ НАД КАЖДЫМ СИГНАЛОМ)
    logger.info("  [2/6] Примеры QRS...")
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
            # ПРИНУДИТЕЛЬНЫЙ ВЫВОД НАЗВАНИЯ КЛАССА НАД ГРАФИКОМ
            ax.set_title(f"{cls_name}", fontsize=16, fontweight='bold', pad=10, color='darkred')
        plot_idx += 1
        
    plt.suptitle("Примеры классификации сигналов по финальным классам", fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '2_qrs_examples_found.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Графики обучения
    logger.info("  [3/6] Графики обучения...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['loss'], label='Train'); ax1.plot(history['val_loss'], label='Val'); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history['acc'], label='Train'); ax2.plot(history['val_acc'], label='Val'); ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '3_training.png'), dpi=150); plt.close()

    # ОЦЕНКА БАЗОВОЙ TCN (3 КЛАССА)
    print("\n" + "="*60)
    print("ОТЧЕТ: Базовая нейросеть TCN (3 класса)")
    print("="*60)
    print(classification_report(y_test_base, base_preds, target_names=TCN_BASE_CLASSES, zero_division=0))

    # 4. Confusion Matrix TCN (3x3)
    logger.info("  [4/6] Матрица ошибок TCN...")
    cm = confusion_matrix(y_test_base, base_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TCN_BASE_CLASSES, yticklabels=TCN_BASE_CLASSES)
    plt.title("Confusion Matrix: Базовый TCN"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '4_cm_base_tcn.png'), dpi=150); plt.close()

    # 5. ROC-AUC (3 кривых)
    logger.info("  [5/6] ROC-AUC...")
    y_bin = label_binarize(y_test_base, classes=range(len(TCN_BASE_CLASSES)))
    plt.figure(figsize=(8, 8))
    for i in range(len(TCN_BASE_CLASSES)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], logits.cpu().numpy()[:, i])
        plt.plot(fpr, tpr, label=f'{TCN_BASE_CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.title("ROC-кривые TCN"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '5_roc_tcn.png'), dpi=150); plt.close()

    # 6. Метрики TCN (Столбцы)
    logger.info("  [6/6] Столбчатая диаграмма...")
    prec = precision_score(y_test_base, base_preds, average=None, zero_division=0)
    rec = recall_score(y_test_base, base_preds, average=None, zero_division=0)
    f1 = f1_score(y_test_base, base_preds, average=None, zero_division=0)
    spec = []
    for i in range(len(TCN_BASE_CLASSES)):
        tn = np.sum((y_test_base != i) & (base_preds != i))
        fp = np.sum((y_test_base != i) & (base_preds == i))
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    x = np.arange(len(TCN_BASE_CLASSES)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*w, prec, w, label='Precision'); ax.bar(x - 0.5*w, rec, w, label='Recall')
    ax.bar(x + 0.5*w, f1, w, label='F1-Score'); ax.bar(x + 1.5*w, spec, w, label='Specificity')
    ax.set_xticks(x); ax.set_xticklabels(TCN_BASE_CLASSES); ax.set_ylim(0, 1.1); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, '6_metrics_tcn.png'), dpi=150); plt.close()

    # ДОПОЛНИТЕЛЬНО: Распределение работы Rule-Based
    print("\n" + "="*60)
    print("ОТЧЕТ: Детализация Rule-Based (PVC -> подтипы)")
    print("="*60)
    counts = Counter(y_final)
    for idx, name in enumerate(FINAL_CLASSES): print(f" - {name}: {counts.get(idx, 0)}")

    logger.info("Все графики сохранены.")

# ==========================================
# ГЛАВНЫЙ ЦИКЛ
# ==========================================
def main():
    import warnings; warnings.filterwarnings("ignore")
    processor = DataProcessor()
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = processor.process()
    
    model, history = train_model(X_train, y_train, X_val, y_val)
    plot_dashboard(X_train.numpy(), model, history, X_test, y_test, meta_test)

if __name__ == "__main__":
    main()