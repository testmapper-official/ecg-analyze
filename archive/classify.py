# -*- coding: utf-8 -*-
"""
Комплексное решение для классификации QRS-комплексов.
Версия 4.1: Исправлена ошибка классификации, добавлено сохранение графиков.
"""

import os
import sys
import glob
import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, auc, accuracy_score, precision_score, 
                             recall_score, f1_score)
from sklearn.preprocessing import label_binarize
from scipy import signal
import matplotlib
matplotlib.use('Agg') # Использование бэкенда без GUI для корректного сохранения файлов
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- КОНФИГУРАЦИЯ И ЛОГИРОВАНИЕ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DB_ROOT = 'DB'
TARGET_SAMPLING_RATE = 360
SEGMENT_SAMPLES = 288  # 800 мс
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Создание директорий
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Унифицированные классы (16 классов)
CLASS_LABELS = [
    'Normal', 'Ventricular_Ectopic', 'Ventricular_Tachycardia', 'LBBB', 'RBBB',
    'Early_Repolarization', 'Brugada_Syndrome', 'LVH', 'Bigeminy', 'Trigeminy',
    'Bidirectional_VT', 'Posterior_MI', 'AIVR', 'Fragmented_QRS', 'Low_Voltage_QRS', 'R_on_T'
]

# Конфигурация баз данных
DB_CONFIG = {
    'mitdb': ('mitdb', 'atr', 'mit'),
    'sddb': ('sddb', 'ari', 'mit'),
    'cudb': ('cudb', 'atr', 'mit'),
    'incartdb': ('incartdb/files', 'atr', 'mit'),
    'edb': ('edb', 'atr', 'mit'),
    'ptbdb': ('ptbdb', None, 'ptb'),
    'brugadahuca': ('brugadahuca/files', None, 'brugada')
}

# ==========================================
# КЛАСС 1: ОБРАБОТКА ДАННЫХ И АУГМЕНТАЦИЯ
# ==========================================
class DataAugment:
    def __init__(self):
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.class_weights = None

    def process(self):
        """Основной метод запуска обработки."""
        logger.info("Запуск процесса сбора данных...")
        all_data = self._collect_all_data()
        
        if not all_data:
            logger.error("Данные не собраны. Проверьте пути к БД.")
            sys.exit(1)

        self._split_and_balance(all_data)
        self._plot_class_distribution()
        self._visualize_examples()

    def _collect_all_data(self):
        all_data = []
        for db_key, (path, ann_ext, db_type) in DB_CONFIG.items():
            full_path = os.path.join(DB_ROOT, path)
            if not os.path.exists(full_path):
                logger.warning(f"Путь не найден: {full_path}")
                continue
            
            logger.info(f"Обработка базы: {db_key}")
            
            if db_type == 'mit':
                files = [f for f in os.listdir(full_path) if f.endswith('.dat')]
                records = sorted(list(set([f.split('.')[0] for f in files])))
                for rec_name in records:
                    rec_path = os.path.join(full_path, rec_name)
                    segs, labs = self._process_mit_record(rec_path, ann_ext)
                    if segs:
                        stats = Counter(labs)
                        logger.info(f"{db_key}::{rec_name} - {len(segs)} примеров. Классы: {dict(stats)}")
                        all_data.extend([(f"{db_key}_{rec_name}", l, s) for s, l in zip(segs, labs)])

            elif db_type == 'ptb':
                patient_folders = [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
                for p_folder in sorted(patient_folders):
                    p_path = os.path.join(full_path, p_folder)
                    files = [f for f in os.listdir(p_path) if f.endswith('.dat')]
                    records = sorted(list(set([f.split('.')[0] for f in files])))
                    for rec_name in records:
                        rec_path = os.path.join(p_path, rec_name)
                        segs, labs = self._process_ptb_record(rec_path)
                        if segs:
                            stats = Counter(labs)
                            logger.info(f"PTB::{p_folder} - {len(segs)} примеров. Классы: {dict(stats)}")
                            all_data.extend([(f"{db_key}_{p_folder}_{rec_name}", l, s) for s, l in zip(segs, labs)])

            elif db_type == 'brugada':
                patient_folders = [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
                for p_folder in sorted(patient_folders):
                    p_path = os.path.join(full_path, p_folder)
                    files = [f for f in os.listdir(p_path) if f.endswith('.dat')]
                    records = sorted(list(set([f.split('.')[0] for f in files])))
                    for rec_name in records:
                        rec_path = os.path.join(p_path, rec_name)
                        segs = self._process_brugada_record(rec_path)
                        if segs:
                            label = CLASS_LABELS.index('Brugada_Syndrome')
                            logger.info(f"Brugada::{p_folder} - {len(segs)} примеров.")
                            all_data.extend([(f"{db_key}_{p_folder}_{rec_name}", label, s) for s in segs])
        return all_data

    # --- Вспомогательные методы обработки сигналов ---
    def _resample(self, sig, fs):
        if fs == TARGET_SAMPLING_RATE: return sig
        return signal.resample(sig, int(len(sig) * TARGET_SAMPLING_RATE / fs))

    def _get_lead_ii(self, record, rec_name):
        names = ['ii', 'mlii', 'v5']
        if hasattr(record, 'sig_name'):
            for i, n in enumerate(record.sig_name):
                if n.lower() in names: return i
        if record.p_signal.shape[1] >= 1:
            # logger.warning(f"Lead II не найден для {rec_name}, исп. канал 0") # Убрал лишний шум в логах
            return 0
        return None

    def _classify_sequence(self, symbols, rr_ms):
        n = len(symbols)
        classes = [CLASS_LABELS.index('Normal')] * n
        for i in range(n):
            sym = symbols[i]
            rr_curr = rr_ms[i] if i < len(rr_ms) else 1000
            
            # Морфология
            if sym == 'L': classes[i] = CLASS_LABELS.index('LBBB')
            elif sym == 'R': classes[i] = CLASS_LABELS.index('RBBB')
            elif sym in ['V', 'E', '!']: classes[i] = CLASS_LABELS.index('Ventricular_Ectopic')
            
            if sym in ['N', 'L', 'R', 'A', 'J', 'S', 'F']: continue

            # Паттерны VT/AIVR
            prev_v = (i > 0 and symbols[i-1] in ['V', 'E', '!'])
            next_v = (i < n-1 and symbols[i+1] in ['V', 'E', '!'])
            
            is_vt_aivr = False
            if prev_v or next_v:
                if prev_v and next_v:
                    classes[i] = CLASS_LABELS.index('Ventricular_Tachycardia') if rr_curr < 600 else CLASS_LABELS.index('AIVR')
                    is_vt_aivr = True

            # Бигеминия / Тригеминия
            if not is_vt_aivr:
                is_bigeminy = (i >= 2 and symbols[i-1] == 'N' and symbols[i-2] in ['V', 'E', '!'])
                is_trigeminy = (i >= 2 and symbols[i-1] == 'N' and symbols[i-2] == 'N')
                
                if is_bigeminy: classes[i] = CLASS_LABELS.index('Bigeminy')
                elif is_trigeminy: classes[i] = CLASS_LABELS.index('Trigeminy')

            # R-on-T
            if classes[i] == CLASS_LABELS.index('Ventricular_Ectopic') and rr_curr < 350:
                classes[i] = CLASS_LABELS.index('R_on_T')
                
        return classes

    def _process_mit_record(self, rec_path, ann_ext):
        try:
            if ann_ext and not os.path.exists(f"{rec_path}.{ann_ext}"):
                return [], []
            
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, ann_ext) if ann_ext else None
            
            idx = self._get_lead_ii(record, rec_path)
            if idx is None: return [], []
            
            ecg = record.p_signal[:, idx]
            ecg = self._resample(ecg, record.fs)
            
            if annotation:
                ratio = TARGET_SAMPLING_RATE / record.fs
                r_peaks = (annotation.sample * ratio).astype(int)
                symbols = [s.upper() for s in annotation.symbol]
                
                rr = np.diff(r_peaks) / TARGET_SAMPLING_RATE * 1000
                rr = np.concatenate([[1000], rr])
                
                classes = self._classify_sequence(symbols, rr)
                
                valid_segments = []
                valid_classes = []
                
                for i, peak in enumerate(r_peaks):
                    start = peak - SEGMENT_SAMPLES // 2
                    end = peak + SEGMENT_SAMPLES // 2
                    if start >= 0 and end < len(ecg):
                        seg = ecg[start:end]
                        if np.std(seg) > 0:
                            cls = classes[i]
                            # Low Voltage Check
                            amp = np.max(seg) - np.min(seg)
                            if amp < 0.5 and cls == CLASS_LABELS.index('Normal'):
                                cls = CLASS_LABELS.index('Low_Voltage_QRS')
                            
                            seg_norm = (seg - np.mean(seg)) / np.std(seg)
                            valid_segments.append(seg_norm)
                            valid_classes.append(cls)
                            
                return valid_segments, valid_classes
            return [], []
        except Exception as e:
            logger.error(f"Ошибка MIT {rec_path}: {e}")
            return [], []

    def _process_ptb_record(self, rec_path):
        try:
            record = wfdb.rdrecord(rec_path)
            header = wfdb.rdheader(rec_path)
            
            text = " ".join(header.comments).lower()
            cls = CLASS_LABELS.index('Normal')
            if 'posterior infarction' in text: cls = CLASS_LABELS.index('Posterior_MI')
            elif 'lbbb' in text: cls = CLASS_LABELS.index('LBBB')
            elif 'rbbb' in text: cls = CLASS_LABELS.index('RBBB')
            elif 'lvh' in text: cls = CLASS_LABELS.index('LVH')
            
            idx = self._get_lead_ii(record, rec_path)
            if idx is None: return [], []
            
            ecg = self._resample(record.p_signal[:, idx], record.fs)
            _, info = nk.ecg_peaks(ecg, sampling_rate=TARGET_SAMPLING_RATE)
            r_peaks = info['ECG_R_Peaks']
            
            segs, labs = [], []
            for peak in r_peaks:
                start = peak - SEGMENT_SAMPLES // 2
                end = peak + SEGMENT_SAMPLES // 2
                if start >= 0 and end < len(ecg):
                    seg = ecg[start:end]
                    if np.std(seg) > 0:
                        seg_norm = (seg - np.mean(seg)) / np.std(seg)
                        segs.append(seg_norm)
                        labs.append(cls)
            return segs, labs
        except Exception:
            return [], []

    def _process_brugada_record(self, rec_path):
        try:
            record = wfdb.rdrecord(rec_path)
            idx = self._get_lead_ii(record, rec_path)
            if idx is None: return []
            
            ecg = self._resample(record.p_signal[:, idx], record.fs)
            _, info = nk.ecg_peaks(ecg, sampling_rate=TARGET_SAMPLING_RATE)
            
            segments = []
            for peak in info['ECG_R_Peaks']:
                start = peak - SEGMENT_SAMPLES // 2
                end = peak + SEGMENT_SAMPLES // 2
                if start >= 0 and end < len(ecg):
                    seg = ecg[start:end]
                    if np.std(seg) > 0:
                        seg_norm = (seg - np.mean(seg)) / np.std(seg)
                        segments.append(seg_norm)
            return segments
        except Exception:
            return []

    # --- Разбиение и Балансировка ---
    def _split_and_balance(self, all_data):
        df = pd.DataFrame(all_data, columns=['pid', 'class_idx', 'segment'])
        
        patients = df['pid'].unique()
        train_p, temp_p = train_test_split(patients, test_size=0.3, random_state=42)
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
        
        train_df = df[df['pid'].isin(train_p)]
        val_df = df[df['pid'].isin(val_p)]
        test_df = df[df['pid'].isin(test_p)]
        
        self.X_train, self.y_train = self._balance_df(train_df)
        self.X_val, self.y_val = val_df['segment'].tolist(), val_df['class_idx'].tolist()
        self.X_test, self.y_test = test_df['segment'].tolist(), test_df['class_idx'].tolist()
        
        self.X_train = np.array(self.X_train).reshape(-1, SEGMENT_SAMPLES, 1)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val).reshape(-1, SEGMENT_SAMPLES, 1)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test).reshape(-1, SEGMENT_SAMPLES, 1)
        self.y_test = np.array(self.y_test)
        
        logger.info(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")

    def _balance_df(self, df, factor=1.5):
        counts = df['class_idx'].value_counts()
        # Если есть классы кроме Normal
        minority_counts = counts[counts.index != CLASS_LABELS.index('Normal')]
        max_minority = minority_counts.max() if not minority_counts.empty else counts.iloc[0]
        target_normal = int(factor * max_minority)
        
        balanced = []
        for cls in counts.index:
            subset = df[df['class_idx'] == cls]
            if cls == CLASS_LABELS.index('Normal') and len(subset) > target_normal:
                subset = subset.sample(target_normal, random_state=42)
            balanced.append(subset)
        
        final_df = pd.concat(balanced).sample(frac=1, random_state=42)
        return final_df['segment'].tolist(), final_df['class_idx'].tolist()

    def _plot_class_distribution(self):
        plt.figure(figsize=(12, 6))
        sns.countplot(x=self.y_train)
        plt.title('Распределение классов после балансировки (Train)')
        plt.xticks(ticks=range(len(CLASS_LABELS)), labels=CLASS_LABELS, rotation=90)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'class_distribution.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"График распределения сохранен: {path}")

    def _visualize_examples(self, num_examples=10):
        logger.info("Генерация визуализации примеров...")
        plt.figure(figsize=(20, 20))
        
        unique_classes = np.unique(self.y_train)
        
        for i, cls_idx in enumerate(unique_classes):
            indices = np.where(self.y_train == cls_idx)[0]
            if len(indices) == 0: continue
            
            selected = np.random.choice(indices, min(num_examples, len(indices)), replace=False)
            
            for j, idx in enumerate(selected):
                pos = i * num_examples + j + 1
                plt.subplot(len(unique_classes), num_examples, pos)
                plt.plot(self.X_train[idx].flatten(), 'k', linewidth=0.8)
                plt.axis('off')
                if j == 0:
                    plt.ylabel(CLASS_LABELS[cls_idx], fontsize=10)
                    
        plt.suptitle("Примеры QRS-комплексов по классам", fontsize=16)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'qrs_examples.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Примеры QRS комплексов сохранены: {path}")

# ==========================================
# КЛАСС 2: КОМПИЛЯЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
class ModelCompiler:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(128, activation='tanh', return_sequences=True),
            Dropout(0.4),
            LSTM(64, activation='tanh'),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs, stage_name="Training"):
        self.model = self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        ]
        
        train_gen = AugmentedGenerator(X_train, y_train, batch_size=64, noise_factor=0.15, num_classes=self.num_classes)
        
        logger.info(f"--- Начало этапа: {stage_name} ({epochs} эпох) ---")
        
        self.history = self.model.fit(
            train_gen,
            validation_data=(X_val, to_categorical(y_val, num_classes=self.num_classes)),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        ).history

    def save_model(self):
        existing = glob.glob(os.path.join(MODELS_DIR, 'model*'))
        next_idx = len(existing) + 1
        model_name = f"model{next_idx}"
        path = os.path.join(MODELS_DIR, model_name)
        
        self.model.save(path + '.h5')
        with open(path + '_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
        
        logger.info(f"Модель сохранена: {path}")
        return path

# Генератор для аугментации данных
class AugmentedGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, noise_factor, num_classes):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.noise_factor = noise_factor
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        augmented_x = np.zeros_like(batch_x)
        
        for i, seg in enumerate(batch_x):
            # Аугментация 10% данных
            if np.random.random() < 0.1:
                noise = np.random.normal(0, self.noise_factor * np.max(seg), seg.shape)
                augmented_x[i] = seg + noise
            else:
                augmented_x[i] = seg
                
        return augmented_x, to_categorical(batch_y, num_classes=self.num_classes)

# ==========================================
# КЛАСС 3: АНАЛИТИКА МОДЕЛИ
# ==========================================
class ModelAnalytic:
    def __init__(self, model, history, class_labels):
        self.model = model
        self.history = history
        self.class_labels = class_labels
        self.num_classes = len(class_labels)

    def analyze(self, X_test, y_test, prefix="model"):
        """Полный анализ модели."""
        print(f"\n--- Анализ модели ({prefix}) ---")
        self._plot_history(prefix)
        metrics = self._calculate_metrics(X_test, y_test)
        self._plot_confusion_matrix(X_test, y_test, prefix)
        self._plot_roc(X_test, y_test, prefix)
        return metrics

    def _plot_history(self, prefix):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Train Acc')
        plt.plot(self.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title('Accuracy')
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'{prefix}_history.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"График истории сохранен: {path}")

    def _calculate_metrics(self, X_test, y_test):
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Classification Report с явным указанием всех классов
        print(classification_report(
            y_test, 
            y_pred, 
            labels=range(len(self.class_labels)), # Фикс для ValueError
            target_names=self.class_labels, 
            zero_division=0
        ))
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Specificity calculation
        cm = confusion_matrix(y_test, y_pred, labels=range(len(self.class_labels)))
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        # Избегаем деления на ноль
        specificity = np.divide(TN, (TN + FP), out=np.zeros_like(TN, dtype=float), where=(TN+FP)!=0)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1:.4f}")
        print(f"Mean Specificity: {np.mean(specificity):.4f}")
        
        return {'accuracy': acc, 'f1': f1, 'specificity': np.mean(specificity)}

    def _plot_confusion_matrix(self, X_test, y_test, prefix):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(self.class_labels)))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Pred')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'{prefix}_confusion_matrix.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Матрица ошибок сохранена: {path}")

    def _plot_roc(self, X_test, y_test, prefix):
        y_pred_prob = self.model.predict(X_test)
        y_test_bin = label_binarize(y_test, classes=range(len(self.class_labels)))
        
        plt.figure(figsize=(12, 10))
        for i in range(self.num_classes):
            # Рисуем ROC только если класс присутствует в тесте
            if np.sum(y_test_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{self.class_labels[i]} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(loc='lower right')
        plt.title('ROC Curve')
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'{prefix}_roc_curve.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"ROC кривая сохранена: {path}")

# ==========================================
# ГЛАВНЫЙ ЦИКЛ
# ==========================================
def main():
    # 1. Подготовка данных
    data_handler = DataAugment()
    data_handler.process()
    
    # 2. Инициализация компилятора
    compiler = ModelCompiler(input_shape=(SEGMENT_SAMPLES, 1), num_classes=len(CLASS_LABELS))
    
    # 3. Тестовый прогон (5 эпох)
    compiler.train(data_handler.X_train, data_handler.y_train, 
                   data_handler.X_val, data_handler.y_val, 
                   epochs=5, stage_name="Test Run")
    
    # Анализ тестового прогона
    test_analytic = ModelAnalytic(compiler.model, compiler.history, CLASS_LABELS)
    test_metrics = test_analytic.analyze(data_handler.X_test, data_handler.y_test, prefix="test_run")
    
    logger.info(f"Test Run Metrics: Acc={test_metrics['accuracy']:.4f}")
    
    # 4. Полное обучение (до 100 эпох)
    # Внимание: так как мы используем EarlyStopping, реальное число эпох может быть меньше 100
    compiler.train(data_handler.X_train, data_handler.y_train, 
                   data_handler.X_val, data_handler.y_val, 
                   epochs=100, stage_name="Full Run")
    
    # Анализ полного прогона
    full_analytic = ModelAnalytic(compiler.model, compiler.history, CLASS_LABELS)
    full_metrics = full_analytic.analyze(data_handler.X_test, data_handler.y_test, prefix="full_run")
    
    # 5. Сохранение модели
    path = compiler.save_model()
    
    print(f"\nИтоговое обучение завершено. Лучшая модель: {path}")
    print(f"Финальные метрики: Accuracy={full_metrics['accuracy']:.4f}, F1={full_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()