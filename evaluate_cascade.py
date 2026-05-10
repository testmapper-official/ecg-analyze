import os
import numpy as np
import torch
import wfdb
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.signal import Signal
from app.core.holter_classifier import HolterClassifier

# Маппинг оригинальных символов БД в наши макро-классы
DB_SYMBOL_TO_CLASS = {
    'N': 'N', 'e': 'N', 'j': 'N',
    'A': 'A', 'a': 'A', 'J': 'A', 'S': 'A',
    'L': 'B', 'R': 'B',
    'E': 'E',
    'r': 'r', # R-on-T не входит в базовые 5 классов для баланса
    'V': 'V', 'F': 'V' 
}

# Маппинг выходных меток классификатора в макро-классы (схлопываем подтипы ПЖС в V)
PRED_TO_CLASS = {
    'N': 'N', 'A': 'A', 'B': 'B', 'E': 'E', 'r': 'V',
    'i': 'V', 'M': 'V', 'P': 'V'  
}

# 5 макро-классов для валидации
VAL_CLASSES = ['N', 'A', 'B', 'E', 'V']
TARGET_VAL_COUNT = 750

def load_test_patients(db_root):
    """Загружает ID пациентов из тестовых выборок BLK и PSS"""
    splits_dir = os.path.join(db_root, 'splits')
    blk_test_patients = set()
    pss_test_patients = set()
    
    blk_test_path = os.path.join(splits_dir, 'blk_test.txt')
    pss_test_path = os.path.join(splits_dir, 'pss_test.txt')
    
    if os.path.exists(blk_test_path):
        with open(blk_test_path, 'r') as f:
            for line in f:
                pid = line.strip()
                if pid: blk_test_patients.add(pid)
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Файл {blk_test_path} не найден. Сначала обучите BLK модель.")
        
    if os.path.exists(pss_test_path):
        with open(pss_test_path, 'r') as f:
            for line in f:
                pid = line.strip()
                if pid: pss_test_patients.add(pid)
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Файл {pss_test_path} не найден. Сначала обучите PSS модель.")
        
    return blk_test_patients, pss_test_patients

def load_noise_data(db_root):
    """Загружает физиологический шум из NSTDB для зашумления тестовой выборки"""
    nstdb_path = os.path.join(db_root, 'nstdb')
    noises = {}
    for n_type in ['em', 'ma']:
        try:
            rec = wfdb.rdrecord(os.path.join(nstdb_path, n_type))
            if rec.fs != 360:
                 noises[n_type] = Signal(data=rec.p_signal[:, 0], fs=rec.fs).resampled_data
            else:
                 noises[n_type] = rec.p_signal[:, 0]
        except Exception as e:
            print(f"ОШИБКА ЗАГРУЗКИ ШУМА {n_type}: {e}")
    return noises

def add_noise_to_signal(sig_obj, noise_data, snr_db_range=(-3, 12)):
    """Накладывает шум на весь сигнал целиком, затем фильтрует и нормализует"""
    if not noise_data or len(sig_obj.resampled_data) == 0:
        return sig_obj
        
    # Выбираем случайный тип шума (em или ma)
    n_type = np.random.choice(list(noise_data.keys()))
    noise_base = noise_data[n_type]
    
    # Метод add_noise внутри класса Signal сам заботится о нарезке/зацикливании шума
    noisy_sig = sig_obj.add_noise(noise_base, snr_db_range=snr_db_range)
    noisy_sig = noisy_sig.wavelet_denoise()
    noisy_sig.standardize()
    
    return noisy_sig

def calculate_rhythm_confidence(sig_obj, start_sample, end_sample, rhythm_type):
    """Считает % истинных меток в эпизоде, подтверждающих ритм"""
    true_in_span = [ann['symbol'] for ann in sig_obj.annotations 
                    if start_sample <= ann['sample'] <= end_sample]
    if not true_in_span: return 0.0
    
    if rhythm_type in ['(V', '(B', '(T']: 
        matches = sum(1 for s in true_in_span if s in ['V', 'F', 'r', 'E'])
    elif rhythm_type == '(BLK': 
        matches = sum(1 for s in true_in_span if s in ['L', 'R'])
    else: matches = 0
    return (matches / len(true_in_span)) * 100.0

def evaluate_cascade_validation(classifier, records, db_root, results_dir, blk_test_patients, pss_test_patients, noise_data):
    """Сбор 750 примеров на класс (чистые + зашумленные) и оценка каскада"""
    md_path = os.path.join(results_dir, "validation_and_rhythms.md")
    
    # Счетчики для валидации (одни на двоих, так как набор истинных классов один)
    val_counts = {cls: 0 for cls in VAL_CLASSES}
    
    # Раздельные хранилища для предсказаний
    y_true_clean, y_pred_clean = [], []
    y_pred_noisy = []
    
    # Хранилище для ритмов
    rhythms_data = defaultdict(list)
    
    print(f"\n=== ВАЛИДАЦИЯ КАСКАДА (Цель: {TARGET_VAL_COUNT} на класс) + ЗАШУМЛЕНИЕ ===")
    
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write("# Валидация каскадного классификатора и Отчет по ритмам\n\n")
        md_file.write("## 1. Сбор валидационной выборки\n\n")
        md_file.write(f"Цель: {TARGET_VAL_COUNT} экземпляров для классов {', '.join(VAL_CLASSES)}\n\n")
        
        for rec_path in records:
            full_path = os.path.join(db_root, rec_path)
            if not os.path.exists(full_path + '.atr'): continue
            
            # Проверяем, не набрали ли мы уже все 5 классов
            if all(count >= TARGET_VAL_COUNT for count in val_counts.values()):
                break
            
            # Определяем ID пациента для текущей записи
            parts = rec_path.split('/')
            patient_id = parts[1] if len(parts) > 1 else None
                
            print(f"  Обработка: {rec_path}")
            try:
                sig = Signal(record_path=full_path)
                sig.standardize()
                
                # 1. Получаем предсказания для чистого сигнала
                results_clean, rhythms = classifier.analyze_signal(sig)
                
                # 2. Создаем зашумленный сигнал и получаем предсказания для него
                if noise_data:
                    noisy_sig = add_noise_to_signal(sig, noise_data, snr_db_range=(-3, 12))
                    results_noisy, _ = classifier.analyze_signal(noisy_sig)
                else:
                    results_noisy = results_clean # Если шума нет, предсказания одинаковы
                
                # 3. Собираем ритмы в Markdown (только по чистому сигналу)
                for rhy in rhythms:
                    start_sec = rhy['start_sample'] / 360.0
                    end_sec = rhy['end_sample'] / 360.0
                    conf = calculate_rhythm_confidence(sig, rhy['start_sample'], rhy['end_sample'], rhy['rhythm'])
                    rhythms_data[rhy['rhythm']].append({
                        'record': rec_path, 'start': start_sec, 'end': end_sec, 'conf': conf
                    })
                
                # 4. Собираем валидационную выборку
                pred_map_clean = {r['sample']: r['label'] for r in results_clean}
                pred_map_noisy = {r['sample']: r['label'] for r in results_noisy}
                
                for ann in sig.annotations:
                    true_sym = ann['symbol'].upper()
                    true_cls = DB_SYMBOL_TO_CLASS.get(true_sym)
                    
                    # Если класс не входит в наши 5 валидационных, пропускаем
                    if true_cls not in VAL_CLASSES: continue
                    # Если уже набрали 750 для этого класса, пропускаем
                    if val_counts[true_cls] >= TARGET_VAL_COUNT: continue
                    
                    # ПРАВИЛА ФИЛЬТРАЦИИ ПОЛЬЗОВАТЕЛЯ:
                    if true_cls == 'B' and patient_id not in blk_test_patients:
                        continue
                    if true_cls == 'V' and patient_id not in pss_test_patients:
                        continue
                    
                    peak = ann['sample']
                    
                    # Чистые предсказания
                    pred_sym_clean = pred_map_clean.get(peak, 'N')
                    pred_cls_clean = PRED_TO_CLASS.get(pred_sym_clean, 'N')
                    
                    # Зашумленные предсказания
                    pred_sym_noisy = pred_map_noisy.get(peak, 'N')
                    pred_cls_noisy = PRED_TO_CLASS.get(pred_sym_noisy, 'N')
                    
                    # Сохраняем результаты (истинный класс один и тот же)
                    y_true_clean.append(true_cls)
                    y_pred_clean.append(pred_cls_clean)
                    y_pred_noisy.append(pred_cls_noisy)
                    
                    val_counts[true_cls] += 1
                    
            except Exception as e: 
                print(f"Ошибка при обработке {rec_path}: {e}")
            
        # Запись прогресса сбора в Markdown
        for cls in VAL_CLASSES:
            md_file.write(f"- **{cls}**: собрано {val_counts[cls]} / {TARGET_VAL_COUNT}\n")
        md_file.write("\n")

        # Запись ритмов в Markdown
        md_file.write("## 2. Детекция ритмов\n\n")
        for rhythm_type, episodes in rhythms_data.items():
            md_file.write(f"### Ритм: {rhythm_type}\n\n")
            md_file.write("| Запись | Начало (с) | Конец (с) | Достоверность (%) |\n")
            md_file.write("|---|---|---|---|\n")
            for ep in episodes:
                md_file.write(f"| {ep['record']} | {ep['start']:.2f} | {ep['end']:.2f} | {ep['conf']:.1f} |\n")
            md_file.write("\n")

    # 5. Вычисление метрик и построение матриц ошибок
    print("\nРасчет метрик для ЧИСТЫХ данных...")
    print(classification_report(y_true_clean, y_pred_clean, labels=VAL_CLASSES, zero_division=0))
    
    cm_clean = confusion_matrix(y_true_clean, y_pred_clean, labels=VAL_CLASSES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', xticklabels=VAL_CLASSES, yticklabels=VAL_CLASSES)
    plt.title(f"Clean Validation Cascade ({TARGET_VAL_COUNT}/cls)")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "validation_cascade_cm_clean.png"), dpi=150); plt.close()

    print("\nРасчет метрик для ЗАШУМЛЕННЫХ данных...")
    print(classification_report(y_true_clean, y_pred_noisy, labels=VAL_CLASSES, zero_division=0))
    
    cm_noisy = confusion_matrix(y_true_clean, y_pred_noisy, labels=VAL_CLASSES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_noisy, annot=True, fmt='d', cmap='Reds', xticklabels=VAL_CLASSES, yticklabels=VAL_CLASSES)
    plt.title(f"Noisy Validation Cascade ({TARGET_VAL_COUNT}/cls)")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "validation_cascade_cm_noisy.png"), dpi=150); plt.close()

def evaluate_pipeline(db_root='DB', results_dir='results/Cascade'):
    os.makedirs(results_dir, exist_ok=True)
    
    print("Загрузка каскадного классификатора...")
    classifier = HolterClassifier(models_dir='models')
    
    mydb_path = os.path.join(db_root, 'mydb')
    records_file = os.path.join(mydb_path, 'RECORDS')
    
    if os.path.exists(records_file):
        blk_test_patients, pss_test_patients = load_test_patients(db_root)
        print(f"Тестовых пациентов BLK: {len(blk_test_patients)} | PSS: {len(pss_test_patients)}")
        
        # Загружаем данные шума
        noise_data = load_noise_data(db_root)
        
        allowed_patients = blk_test_patients.union(pss_test_patients)
        
        with open(records_file, 'r') as f:
            all_records = [line.strip() for line in f if line.strip() and line.strip().startswith('II/')]
        
        independent_records = []
        for rec in all_records:
            parts = rec.split('/')
            if len(parts) > 1:
                patient_id = parts[1]
                if patient_id in allowed_patients:
                    independent_records.append(rec)
        
        print(f"Всего записей в БД: {len(all_records)}. Записей для независимого теста каскада: {len(independent_records)}")
        evaluate_cascade_validation(classifier, independent_records, mydb_path, results_dir, blk_test_patients, pss_test_patients, noise_data)
    
    print("\nВсе оценки завершены!")

if __name__ == "__main__":
    evaluate_pipeline()