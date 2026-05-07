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
from app.training.ii.dataset import DatasetBuilder

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

def evaluate_cascade_validation(classifier, records, db_root, results_dir):
    """Сбор 750 примеров на класс и оценка каскада + генерация Markdown по ритмам"""
    md_path = os.path.join(results_dir, "validation_and_rhythms.md")
    
    # Счетчики и хранилища для валидации
    val_counts = {cls: 0 for cls in VAL_CLASSES}
    y_true_val, y_pred_val = [], []
    
    # Хранилище для ритмов
    rhythms_data = defaultdict(list)
    
    print(f"\n=== ВАЛИДАЦИЯ КАСКАДА (Цель: {TARGET_VAL_COUNT} на класс) + РИТМЫ ===")
    
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
                
            print(f"  Обработка: {rec_path}")
            try:
                sig = Signal(record_path=full_path)
                results_clean, rhythms = classifier.analyze_signal(sig)
                
                # 1. Собираем ритмы в Markdown
                for rhy in rhythms:
                    start_sec = rhy['start_sample'] / 360.0
                    end_sec = rhy['end_sample'] / 360.0
                    conf = calculate_rhythm_confidence(sig, rhy['start_sample'], rhy['end_sample'], rhy['rhythm'])
                    rhythms_data[rhy['rhythm']].append({
                        'record': rec_path, 'start': start_sec, 'end': end_sec, 'conf': conf
                    })
                
                # 2. Собираем валидационную выборку
                pred_map = {r['sample']: r['label'] for r in results_clean}
                
                for ann in sig.annotations:
                    true_sym = ann['symbol'].upper()
                    true_cls = DB_SYMBOL_TO_CLASS.get(true_sym)
                    
                    # Если класс не входит в наши 5 валидационных, пропускаем
                    if true_cls not in VAL_CLASSES: continue
                    # Если уже набрали 750 для этого класса, пропускаем
                    if val_counts[true_cls] >= TARGET_VAL_COUNT: continue
                    
                    peak = ann['sample']
                    pred_sym = pred_map.get(peak, 'N')
                    pred_cls = PRED_TO_CLASS.get(pred_sym, 'N')
                    
                    y_true_val.append(true_cls)
                    y_pred_val.append(pred_cls)
                    val_counts[true_cls] += 1
                    
            except Exception as e: pass
            
        # Запись прогресса сбора в Markdown
        for cls in VAL_CLASSES:
            md_file.write(f"- **{cls}**: собрано {val_counts[cls]} / {TARGET_VAL_COUNT}\n")
        md_file.write("\n")

        # 3. Запись ритмов в Markdown
        md_file.write("## 2. Детекция ритмов\n\n")
        for rhythm_type, episodes in rhythms_data.items():
            md_file.write(f"### Ритм: {rhythm_type}\n\n")
            md_file.write("| Запись | Начало (с) | Конец (с) | Достоверность (%) |\n")
            md_file.write("|---|---|---|---|\n")
            for ep in episodes:
                md_file.write(f"| {ep['record']} | {ep['start']:.2f} | {ep['end']:.2f} | {ep['conf']:.1f} |\n")
            md_file.write("\n")

    # 4. Вычисление метрик и построение матрицы ошибок валидации
    print("\nРасчет метрик валидации...")
    print(classification_report(y_true_val, y_pred_val, labels=VAL_CLASSES, zero_division=0))
    
    cm = confusion_matrix(y_true_val, y_pred_val, labels=VAL_CLASSES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VAL_CLASSES, yticklabels=VAL_CLASSES)
    plt.title(f"Balanced Validation Cascade ({TARGET_VAL_COUNT}/cls)")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "validation_cascade_cm.png"), dpi=150); plt.close()

def evaluate_rount_mitdb(classifier, db_root, results_dir):
    """Оценка R-on-T на MIT-BIH Arrhythmia DB"""
    mitdb_path = os.path.join(db_root, 'mitdb')
    if not os.path.exists(mitdb_path):
        print("Папка mitdb не найдена, оценка R-on-T пропущена."); return
        
    print("\n=== ОЦЕНКА R-ON-T (MIT-BIH) ===")
    records = [f.split('.')[0] for f in os.listdir(mitdb_path) if f.endswith('.dat')]
    
    y_true, y_pred = [], []
    
    for rec in records:
        full_path = os.path.join(mitdb_path, rec)
        try:
            ann = wfdb.rdann(full_path, 'atr')
            if 'r' not in [s.upper() for s in ann.symbol]: continue
                
            print(f"  Анализ R-on-T: {rec}")
            sig = Signal(record_path=full_path)
            results, _ = classifier.analyze_signal(sig)
            
            pred_map = {r['sample']: r['label'] for r in results}
            
            for i, sym in enumerate(ann.symbol):
                sym = sym.upper()
                if sym == 'R':
                    peak = ann.sample[i]
                    pred_label = pred_map.get(peak, 'N')
                    y_true.append('r')
                    y_pred.append(PRED_TO_CLASS.get(pred_label, 'N'))
        except: pass

    if not y_true:
        print("Не найдено записей с меткой 'r' в mitdb!"); return

    print("\nОтчет по R-on-T (MIT-BIH):")
    print(classification_report(y_true, y_pred, labels=['r', 'V', 'N'], zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=['r', 'V', 'N'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['r', 'V', 'N'], yticklabels=['r', 'V', 'N'])
    plt.title("R-on-T Detection (MIT-BIH)"); plt.ylabel('True'); plt.xlabel('Pred')
    plt.savefig(os.path.join(results_dir, "rount_mitdb_cm.png"), dpi=150); plt.close()

def evaluate_rount_sddb(classifier, db_root, results_dir):
    """Оценка R-on-T и V на Sudden Cardiac Death DB"""
    sddb_path = os.path.join(db_root, 'sddb')
    if not os.path.exists(sddb_path):
        print("Папка sddb не найдена, оценка пропущена."); return
        
    print("\n=== ОЦЕНКА V и R-ON-T (SDDB) ===")
    records = [f.split('.')[0] for f in os.listdir(sddb_path) if f.endswith('.dat')]
    
    total_v_true = 0; total_v_pred = 0
    total_rount_true = 0; total_rount_pred = 0
    has_true_rount = False
    
    for rec in records:
        full_path = os.path.join(sddb_path, rec)
        try:
            ann = wfdb.rdann(full_path, 'atr')
            symbols = [s.upper() for s in ann.symbol]
            
            if 'R' in symbols: has_true_rount = True
                
            v_true_count = sum(1 for s in symbols if s == 'V')
            r_true_count = sum(1 for s in symbols if s == 'R')
            total_v_true += v_true_count
            total_rount_true += r_true_count
            
            print(f"  Анализ SDDB: {rec}")
            sig = Signal(record_path=full_path)
            results, _ = classifier.analyze_signal(sig)
            
            v_pred_count = sum(1 for r in results if r['label'] in ['V', 'r', 'i', 'M', 'P'])
            rount_pred_count = sum(1 for r in results if r['label'] == 'r')
            total_v_pred += v_pred_count
            total_rount_pred += rount_pred_count
            
        except: pass

    print("\n--- Статистика SDDB ---")
    print(f"Всего истинных ПЖС (V): {total_v_true}")
    print(f"Всего найдено ПЖС (V+подтипы): {total_v_pred}")
    print(f"Всего истинных R-on-T (r): {total_rount_true}")
    print(f"Всего найдено R-on-T (r): {total_rount_pred}")
    
    if total_v_pred > 0:
        rount_percent = (total_rount_pred / total_v_pred) * 100
        print(f"Процент R-on-T от найденных ПЖС: {rount_percent:.2f}%")
    else: print("ПЖС не найдено.")
        
    if has_true_rount:
        print("Внимание: В SDDB есть истинные метки 'r'. Строим матрицу ошибок.")
        y_true, y_pred = [], []
        for rec in records:
            full_path = os.path.join(sddb_path, rec)
            try:
                ann = wfdb.rdann(full_path, 'atr')
                sig = Signal(record_path=full_path)
                results, _ = classifier.analyze_signal(sig)
                pred_map = {r['sample']: r['label'] for r in results}
                for i, sym in enumerate(ann.symbol):
                    sym = sym.upper()
                    if sym in ['V', 'R']:
                        peak = ann.sample[i]
                        pred_label = pred_map.get(peak, 'N')
                        y_true.append(sym)
                        y_pred.append(PRED_TO_CLASS.get(pred_label, 'N'))
            except: pass
            
        cm = confusion_matrix(y_true, y_pred, labels=['V', 'R', 'N'])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['V', 'r', 'N'], yticklabels=['V', 'r', 'N'])
        plt.title("V vs R-on-T (SDDB)"); plt.ylabel('True'); plt.xlabel('Pred')
        plt.savefig(os.path.join(results_dir, "rount_sddb_cm.png"), dpi=150); plt.close()

def evaluate_pipeline(db_root='DB', results_dir='results/Cascade'):
    os.makedirs(results_dir, exist_ok=True)
    
    print("Загрузка каскадного классификатора...")
    classifier = HolterClassifier(models_dir='models')
    
    mydb_path = os.path.join(db_root, 'mydb')
    records_file = os.path.join(mydb_path, 'RECORDS')
    
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            # Берем все записи, чтобы набрать 750 редких классов
            records = [line.strip() for line in f if line.strip() and line.strip().startswith('II/')]
            
        evaluate_cascade_validation(classifier, records, mydb_path, results_dir)
        
    # evaluate_rount_mitdb(classifier, db_root, results_dir)
    # evaluate_rount_sddb(classifier, db_root, results_dir)
    
    print("\nВсе оценки завершены!")

if __name__ == "__main__":
    evaluate_pipeline()