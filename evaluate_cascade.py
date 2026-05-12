import os
import numpy as np
import torch
import wfdb
from collections import defaultdict
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, 
                             precision_score, recall_score, f1_score, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.signal import Signal
from app.core.holter_classifier import HolterClassifier

# --- НАСТРОЙКИ И МАППИНГИ ---
DB_ROOT = 'DB'
MYDB_PATH = os.path.join(DB_ROOT, 'mydb')
RESULTS_ROOT = 'results/final'
CASCADE_DIR = os.path.join(RESULTS_ROOT, 'Cascade')

VAL_CLASSES = ['N', 'A', 'B', 'E', 'V']
TARGET_VAL_COUNT = 750

DB_SYMBOL_TO_CLASS = {
    'N': 'N', 'e': 'N', 'j': 'N',
    'A': 'A', 'a': 'A', 'J': 'A', 'S': 'A',
    'L': 'B', 'R': 'B',
    'E': 'E',
    'r': 'V', 'V': 'V', 'F': 'V' 
}

PRED_TO_CLASS = {
    'N': 'N', 'A': 'A', 'B': 'B', 'E': 'E', 'r': 'V',
    'i': 'V', 'M': 'V', 'P': 'V'  
}

VALID_QRS_SYMBOLS = ['V', 'F', 'r', 'i', 'M', 'P', 'B', 'L', 'R', 'A', 'a', 'N', 'E', 'e']

# --- ФУНКЦИИ ГЕНЕРАЦИИ ШУМА ---
def generate_worst_case_noise(length, fs=360):
    t = np.arange(length) / fs
    awgn = np.random.normal(0, 0.5, length)
    drift = 0.5 * np.sin(2 * np.pi * np.random.uniform(0.2, 0.5) * t + np.random.uniform(0, 2*np.pi))
    emg = 0.2 * np.random.normal(0, 1, length) * np.sin(2 * np.pi * np.random.uniform(20, 50) * t)
    return awgn + drift + emg

def add_noise_to_signal(sig_obj):
    noisy_data = sig_obj.resampled_data + generate_worst_case_noise(len(sig_obj.resampled_data), 360)
    noisy_sig = Signal(data=noisy_data, fs=360, annotations=sig_obj.annotations)
    return noisy_sig

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ МЕТРИК ---
def calculate_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    spec_per_class = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(spec_per_class)

def save_metrics_and_plots(y_true, y_pred, class_name, output_dir, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    
    if len(y_true) == 0:
        print(f"WARNING: No data for {class_name} ({prefix}). Skipping metrics.")
        return {'Precision': 0, 'Recall': 0, 'F1': 0, 'Accuracy': 0, 'Specificity': 0, 'AUC': 0}

    # 1. Метрики
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    spec = calculate_specificity(y_true, y_pred, labels=np.unique(y_true + y_pred).tolist())
    
    # AUC-ROC (Псевдо-вероятности для мультикласса, так как каскад выдает жесткие метки)
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    y_prob = np.where(y_pred_bin == 1, 0.9, 0.1)
    
    roc_auc = 0.0
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'AUC-ROC - {class_name} ({prefix})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'{prefix}_auc_roc.png'), dpi=150)
        plt.close()

    # Столбчатая диаграмма метрик
    metrics = {'Precision': prec, 'Recall': rec, 'F1': f1, 'Accuracy': acc, 'Specificity': spec, 'AUC': roc_auc}
    plt.figure()
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.ylim(0, 1.05)
    plt.title(f'Metrics - {class_name} ({prefix})')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, f'{prefix}_metrics_bar.png'), dpi=150)
    plt.close()

    # Матрица ошибок 5x5
    cm = confusion_matrix(y_true, y_pred, labels=VAL_CLASSES)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VAL_CLASSES, yticklabels=VAL_CLASSES)
    plt.title(f'Confusion Matrix - {class_name} ({prefix})')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, f'{prefix}_confusion_matrix.png'), dpi=150)
    plt.close()

    return metrics

# --- ОЦЕНКА КАСКАДА ---
def evaluate_cascade(classifier, records):
    print("\n=== Оценка каскадного классификатора ===")
    val_counts = {cls: 0 for cls in VAL_CLASSES}
    
    y_true_bal_c, y_pred_bal_c = [], []
    y_true_bal_n, y_pred_bal_n = [], []
    y_true_full_c, y_pred_full_c = [], []
    y_true_full_n, y_pred_full_n = [], []
    
    for rec_path in records:
        full_path = os.path.join(MYDB_PATH, rec_path)
        if not os.path.exists(full_path + '.atr'): continue
        try:
            sig_clean = Signal(record_path=full_path)
            sig_noisy = add_noise_to_signal(sig_clean)
            
            results_clean, _ = classifier.analyze_signal(sig_clean)
            results_noisy, _ = classifier.analyze_signal(sig_noisy)
            
            pred_map_clean = {r['sample']: r['label'] for r in results_clean}
            pred_map_noisy = {r['sample']: r['label'] for r in results_noisy}
            
            for ann in sig_clean.annotations:
                true_sym = ann['symbol'].upper()
                true_cls = DB_SYMBOL_TO_CLASS.get(true_sym)
                if true_cls not in VAL_CLASSES: continue
                peak = ann['sample']
                
                # FULL
                pred_c_full = PRED_TO_CLASS.get(pred_map_clean.get(peak, 'N'), 'N')
                pred_n_full = PRED_TO_CLASS.get(pred_map_noisy.get(peak, 'N'), 'N')
                y_true_full_c.append(true_cls); y_pred_full_c.append(pred_c_full)
                y_true_full_n.append(true_cls); y_pred_full_n.append(pred_n_full)
                
                # BALANCED (750 / класс)
                if val_counts[true_cls] < TARGET_VAL_COUNT:
                    val_counts[true_cls] += 1
                    y_true_bal_c.append(true_cls); y_pred_bal_c.append(pred_c_full)
                    y_true_bal_n.append(true_cls); y_pred_bal_n.append(pred_n_full)
                    
        except Exception as e:
            print(f"Ошибка {rec_path}: {e}")

    res_bal_clean = save_metrics_and_plots(y_true_bal_c, y_pred_bal_c, "Cascade_Balanced", CASCADE_DIR, prefix="Clean")
    res_bal_noisy = save_metrics_and_plots(y_true_bal_n, y_pred_bal_n, "Cascade_Balanced", CASCADE_DIR, prefix="Noisy")
    res_full_clean = save_metrics_and_plots(y_true_full_c, y_pred_full_c, "Cascade_Full", CASCADE_DIR, prefix="Clean")
    res_full_noisy = save_metrics_and_plots(y_true_full_n, y_pred_full_n, "Cascade_Full", CASCADE_DIR, prefix="Noisy")

    return res_bal_clean, res_bal_noisy, res_full_clean, res_full_noisy

# --- ОЦЕНКА РИТМОВ ---
def evaluate_rhythms(classifier, records):
    print("\n=== Оценка ритмов ===")
    rhythm_stats = {'Clean': defaultdict(list), 'Noisy': defaultdict(list)}
    
    for rec_path in records:
        full_path = os.path.join(MYDB_PATH, rec_path)
        if not os.path.exists(full_path + '.atr'): continue
        try:
            sig_clean = Signal(record_path=full_path)
            sig_noisy = add_noise_to_signal(sig_clean)
            
            true_seq = [{'sample': ann['sample'], 'group': HolterClassifier._map_group(ann['symbol'].upper())} 
                        for ann in sig_clean.annotations if ann['symbol'].upper() in VALID_QRS_SYMBOLS]
            true_rhythms = HolterClassifier.detect_rhythms(true_seq)
            
            _, pred_rhythms_clean = classifier.analyze_signal(sig_clean)
            _, pred_rhythms_noisy = classifier.analyze_signal(sig_noisy)
            
            pred_rhy_clean_fmt = [{'start_sample': r['start_sample'], 'end_sample': r['end_sample'], 'type': r['rhythm'].strip('(')} for r in pred_rhythms_clean]
            pred_rhy_noisy_fmt = [{'start_sample': r['start_sample'], 'end_sample': r['end_sample'], 'type': r['rhythm'].strip('(')} for r in pred_rhythms_noisy]
            
            def merge_rhythms(rhythms):
                if not rhythms: return []
                merged = [rhythms[0]]
                for r in rhythms[1:]:
                    if r['type'] == merged[-1]['type'] and (r['start_sample'] - merged[-1]['end_sample']) / 360.0 <= 2.0:
                        merged[-1]['end_sample'] = r['end_sample']
                    else:
                        merged.append(r)
                return merged

            true_rhythms = merge_rhythms(true_rhythms)
            pred_rhy_clean_fmt = merge_rhythms(pred_rhy_clean_fmt)
            pred_rhy_noisy_fmt = merge_rhythms(pred_rhy_noisy_fmt)
            
            def process_rhythms(pred_rhy, true_rhy, stats_dict):
                matched_true = set()
                for pred in pred_rhy:
                    best_match = None
                    best_iou = 0
                    for i, true in enumerate(true_rhy):
                        if true['type'] == pred['type'] and i not in matched_true:
                            inter_start = max(pred['start_sample'], true['start_sample'])
                            inter_end = min(pred['end_sample'], true['end_sample'])
                            if inter_start < inter_end:
                                union_start = min(pred['start_sample'], true['start_sample'])
                                union_end = max(pred['end_sample'], true['end_sample'])
                                iou = (inter_end - inter_start) / (union_end - union_start)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = (i, true)
                    
                    if best_match:
                        matched_true.add(best_match[0])
                        true = best_match[1]
                        beats_in_pred = [ann for ann in sig_clean.annotations 
                                        if pred['start_sample'] <= ann['sample'] <= pred['end_sample']]
                        correct_beats = sum(1 for b in beats_in_pred if HolterClassifier._map_group(b['symbol'].upper()) == HolterClassifier._map_group(pred['type'][0]))
                        total_beats = len(beats_in_pred)
                        overlap_pct = (correct_beats / total_beats * 100) if total_beats > 0 else 0
                        
                        start_diff = (pred['start_sample'] - true['start_sample']) / 360.0
                        end_diff = (pred['end_sample'] - true['end_sample']) / 360.0
                        
                        if overlap_pct >= 50:
                            stats_dict['Partial_Over50'].append(pred['type'])
                            stats_dict['Start_Diff'].append(start_diff)
                            stats_dict['End_Diff'].append(end_diff)
                        if overlap_pct == 100:
                            stats_dict['Perfect_100'].append(pred['type'])
                    else:
                        stats_dict['False_Under50'].append(pred['type'])
                        
            process_rhythms(pred_rhy_clean_fmt, true_rhythms, rhythm_stats['Clean'])
            process_rhythms(pred_rhy_noisy_fmt, true_rhythms, rhythm_stats['Noisy'])
            
        except Exception as e:
            print(f"Ошибка ритмов {rec_path}: {e}")

    os.makedirs(CASCADE_DIR, exist_ok=True)
    with open(os.path.join(CASCADE_DIR, 'rhythm_report.txt'), 'w', encoding='utf-8') as f:
        for noise_type in ['Clean', 'Noisy']:
            f.write(f"=== {noise_type} ===\n")
            stats = rhythm_stats[noise_type]
            f.write(f"Ложные (<50% совпадения): {len(stats['False_Under50'])}\n")
            f.write(f"Частично верные (>50% совпадения): {len(stats['Partial_Over50'])}\n")
            f.write(f"Идеально верные (100% совпадения): {len(stats['Perfect_100'])}\n")
            
            if stats['Start_Diff']:
                sd_start = np.std(stats['Start_Diff'])
                sd_end = np.std(stats['End_Diff'])
                f.write(f"Стандартное отклонение начала (сек): {sd_start:.4f}\n")
                f.write(f"Стандартное отклонение конца (сек): {sd_end:.4f}\n")
            else:
                f.write("Нет данных для расчета SD\n")
            f.write("\n")

    for noise_type in ['Clean', 'Noisy']:
        stats = rhythm_stats[noise_type]
        categories = ['False (<50%)', 'Partial (>50%)', 'Perfect (100%)']
        counts = [len(stats['False_Under50']), len(stats['Partial_Over50']), len(stats['Perfect_100'])]
        plt.figure()
        plt.bar(categories, counts, color=['#d62728', '#ff7f0e', '#2ca02c'])
        plt.title(f'Rhythm Detection Quality - {noise_type}')
        plt.ylabel('Count')
        for i, v in enumerate(counts): plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(CASCADE_DIR, f'rhythm_quality_{noise_type}.png'), dpi=150)
        plt.close()

# --- СРАВНИТЕЛЬНЫЙ ГРАФИК (Чистый vs Зашумленный) ---
def plot_comparison(res_clean, res_noisy, class_name, output_dir):
    print("\n=== Построение сравнительных графиков Clean vs Noisy ===")
    metrics_to_plot = ['F1', 'Precision', 'Recall', 'Accuracy', 'Specificity', 'AUC']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    clean_vals = [res_clean[m] for m in metrics_to_plot]
    noisy_vals = [res_noisy[m] for m in metrics_to_plot]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, clean_vals, width, label='Clean', color='#1f77b4')
    rects2 = ax.bar(x + width/2, noisy_vals, width, label='Noisy', color='#ff7f0e')
    
    ax.set_ylabel('Score')
    ax.set_title(f'{class_name} Performance: Clean vs Noisy')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.savefig(os.path.join(output_dir, f'{class_name}_clean_vs_noisy.png'), dpi=150)
    plt.close()

# --- ГЛАВНЫЙ ЗАПУСК ---
def main():
    records_file = os.path.join(MYDB_PATH, 'RECORDS')
    if not os.path.exists(records_file):
        print(f"Файл RECORDS не найден в {MYDB_PATH}"); return
        
    with open(records_file, 'r') as f:
        records = [line.strip() for line in f if line.strip().startswith('II/')]
        
    print(f"Загружено записей из mydb: {len(records)}")
    
    classifier = HolterClassifier(models_dir='models')
    
    # 1. Оценка каскада
    res_bal_c, res_bal_n, res_full_c, res_full_n = evaluate_cascade(classifier, records)
    
    # 2. Сравнение Clean vs Noisy
    plot_comparison(res_bal_c, res_bal_n, "Cascade_Balanced", CASCADE_DIR)
    plot_comparison(res_full_c, res_full_n, "Cascade_Full", CASCADE_DIR)
    
    # 3. Оценка ритмов
    evaluate_rhythms(classifier, records)
    
    print("\n=== ВСЕ ОЦЕНКИ ЗАВЕРШЕНЫ ===")

if __name__ == "__main__":
    main()