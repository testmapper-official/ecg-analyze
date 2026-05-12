import os
import numpy as np
import torch
import wfdb
from collections import defaultdict
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.signal import Signal
from app.core.holter_classifier import HolterClassifier

# --- НАСТРОЙКИ И МАППИНГИ ---
DB_ROOT = 'DB'
MYDB_PATH = os.path.join(DB_ROOT, 'mydb')
SPLITS_DIR = os.path.join(DB_ROOT, 'splits')
RESULTS_ROOT = 'results/final'
TARGET_PER_CLASS = 750
N_BOOTSTRAPS = 1000

# Строго как в dataset.py
PSS_SYMBOLS = ['V', 'F', 'r']
NORMAL_PSS_SYMBOLS = ['N', 'E', 'A']

BLK_SYMBOLS = ['L', 'R']
NORMAL_BLK_SYMBOLS = ['N', 'V', 'A', 'F', 'E', 'r']

# Оптимальные пороги из обучения
THRESHOLDS = {
    'TCN_PSS': 0.8236, 'MLP_PSS': 0.5422,
    'TCN_BLK': 0.0663, 'MLP_BLK': 0.8245
}

MODEL_CONFIGS = {
    'TCN_PSS': {'split_file': 'pss_test.txt', 'target': 'PSS'},
    'MLP_PSS': {'split_file': 'parametric_test.txt', 'target': 'PSS'},
    'TCN_BLK': {'split_file': 'blk_test.txt', 'target': 'BLK'},
    'MLP_BLK': {'split_file': 'parametric_blk_test.txt', 'target': 'BLK'}
}

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
def specificity_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel(); return tn / (tn + fp + 1e-8)
    return 0.0

def compute_metrics_bootstrap_strict(pool_probs_c_normal, pool_probs_c_target, 
                                     pool_probs_n_normal, pool_probs_n_target, 
                                     threshold, n_bootstraps=1000):
    """
    Честный бутстрап: на каждой итерации формирует НОВУЮ выборку 750/750 из пула,
    считает метрики и так 1000 раз.
    """
    n_norm = len(pool_probs_c_normal)
    n_targ = len(pool_probs_c_target)
    
    sample_size_norm = min(n_norm, TARGET_PER_CLASS)
    sample_size_targ = min(n_targ, TARGET_PER_CLASS)
    
    if sample_size_norm < TARGET_PER_CLASS or sample_size_targ < TARGET_PER_CLASS:
        print(f"  [WARNING] В пуле недостаточно данных для {TARGET_PER_CLASS}/{TARGET_PER_CLASS}! Используем {sample_size_norm}/{sample_size_targ}")

    rng = np.random.default_rng(42)
    results_clean = {'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1-Score': []}
    results_noisy = {'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1-Score': []}
    
    for _ in range(n_bootstraps):
        # 1. Вытягиваем случайные индексы (без возвращения внутри итерации)
        idx_n = rng.choice(n_norm, size=sample_size_norm, replace=False)
        idx_t = rng.choice(n_targ, size=sample_size_targ, replace=False)
        
        # 2. Формируем таргет (0 для Normal, 1 для Target)
        y_true_batch = np.concatenate([np.zeros(sample_size_norm), np.ones(sample_size_targ)])
        
        # --- Clean ---
        y_prob_c = np.concatenate([pool_probs_c_normal[idx_n], pool_probs_c_target[idx_t]])
        y_pred_c = (y_prob_c >= threshold).astype(int)
        
        results_clean['Accuracy'].append(accuracy_score(y_true_batch, y_pred_c))
        results_clean['Recall'].append(recall_score(y_true_batch, y_pred_c, zero_division=0))
        results_clean['Specificity'].append(specificity_binary(y_true_batch, y_pred_c))
        results_clean['F1-Score'].append(f1_score(y_true_batch, y_pred_c, zero_division=0))
        
        # --- Noisy (берем те же индексы для парного сравнения Clean vs Noisy) ---
        y_prob_n = np.concatenate([pool_probs_n_normal[idx_n], pool_probs_n_target[idx_t]])
        y_pred_n = (y_prob_n >= threshold).astype(int)
        
        results_noisy['Accuracy'].append(accuracy_score(y_true_batch, y_pred_n))
        results_noisy['Recall'].append(recall_score(y_true_batch, y_pred_n, zero_division=0))
        results_noisy['Specificity'].append(specificity_binary(y_true_batch, y_pred_n))
        results_noisy['F1-Score'].append(f1_score(y_true_batch, y_pred_n, zero_division=0))
        
    for k in results_clean: results_clean[k] = np.array(results_clean[k])
    for k in results_noisy: results_noisy[k] = np.array(results_noisy[k])
    
    # Для отрисовки матриц и ROC вернем один "типичный" набор 750/750
    idx_n_fixed = rng.choice(n_norm, size=sample_size_norm, replace=False)
    idx_t_fixed = rng.choice(n_targ, size=sample_size_targ, replace=False)
    
    y_true_fixed = np.concatenate([np.zeros(sample_size_norm), np.ones(sample_size_targ)])
    y_prob_c_fixed = np.concatenate([pool_probs_c_normal[idx_n_fixed], pool_probs_c_target[idx_t_fixed]])
    y_prob_n_fixed = np.concatenate([pool_probs_n_normal[idx_n_fixed], pool_probs_n_target[idx_t_fixed]])
    
    return results_clean, results_noisy, y_true_fixed, y_prob_c_fixed, y_prob_n_fixed

# --- ФУНКЦИИ ОТРИСОВКИ ---
def plot_cm(y_true, y_pred, title, filename, target_name='Target'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', target_name], yticklabels=['Normal', target_name])
    plt.title(title); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()

def plot_roc(y_true, y_prob, threshold, title, filename):
    if len(np.unique(y_true)) < 2: return
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.scatter(fpr[np.argmin(np.abs(thresholds - threshold))], tpr[np.argmin(np.abs(thresholds - threshold))], marker='o', color='red', s=100, label=f'Threshold = {threshold:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc="lower right"); plt.savefig(filename, dpi=150); plt.close()

def plot_boxplots(clean_groups, noisy_groups, title, filename):
    metrics_names = list(clean_groups.keys())
    fig, axes = plt.subplots(1, len(metrics_names), figsize=(20, 6))
    fig.suptitle(title, fontsize=16)
    all_vals = list(clean_groups.values()) + list(noisy_groups.values())
    global_min = min(np.min(arr) for arr in all_vals)
    global_max = max(np.max(arr) for arr in all_vals)
    y_min, y_max = max(0, global_min - 0.05), min(1, global_max + 0.05)
    for i, metric in enumerate(metrics_names):
        ax = axes[i]; data = [clean_groups[metric], noisy_groups[metric]]
        bp = ax.boxplot(data, tick_labels=['Clean', 'Noisy'], patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor('royalblue'); bp['boxes'][1].set_facecolor('salmon')
        ax.set_title(metric); ax.set_ylim(y_min, y_max); ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()

def plot_comparison_bar(clean_groups, noisy_groups, title, output_dir):
    labels = list(clean_groups.keys())
    clean_means = [np.mean(clean_groups[l]) for l in labels]
    noisy_means = [np.mean(noisy_groups[l]) for l in labels]
    x = np.arange(len(labels)); width = 0.35; fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, clean_means, width, label='Clean Data', color='royalblue')
    rects2 = ax.bar(x + width/2, noisy_means, width, label='Noisy Data', color='salmon')
    ax.set_ylabel('Score'); ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.1); ax.legend(loc='lower right')
    for rect in rects1 + rects2:
        height = rect.get_height(); ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'comparison_metrics.png'), dpi=150); plt.close()

def print_metrics(clean_groups, noisy_groups, model_name, target_name):
    print(f"\n{'='*70}\nФИНАЛЬНЫЕ МЕТРИКИ {model_name}: Normal vs {target_name} (Bootstrap {N_BOOTSTRAPS} CI)\n{'='*70}")
    print(f"{'Метрика':<15} | {'Чистые данные':<25} | {'Зашумленные данные':<25}\n{'-'*70}")
    for metric in ['Accuracy', 'Recall', 'Specificity', 'F1-Score']:
        c_mean = np.mean(clean_groups[metric]); c_ci = np.percentile(clean_groups[metric], [2.5, 97.5])
        n_mean = np.mean(noisy_groups[metric]); n_ci = np.percentile(noisy_groups[metric], [2.5, 97.5])
        print(f"{metric:<15} | {c_mean:.4f} [{c_ci[0]:.4f}, {c_ci[1]:.4f}] | {n_mean:.4f} [{n_ci[0]:.4f}, {n_ci[1]:.4f}]")
    print(f"{'='*70}")

def evaluate_model(pool_probs_c_normal, pool_probs_c_target, pool_probs_n_normal, pool_probs_n_target, threshold, target_name, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Главный вызов: формируем 1000 выборок 750/750, считаем метрики
    groups_clean, groups_noisy, y_true_fixed, y_prob_c_fixed, y_prob_n_fixed = \
        compute_metrics_bootstrap_strict(pool_probs_c_normal, pool_probs_c_target, 
                                         pool_probs_n_normal, pool_probs_n_target, 
                                         threshold, n_bootstraps=N_BOOTSTRAPS)
    
    # Для графиков CM и ROC используем одну репрезентативную выборку
    y_pred_c_fixed = (y_prob_c_fixed >= threshold).astype(int)
    y_pred_n_fixed = (y_prob_n_fixed >= threshold).astype(int)
    
    plot_cm(y_true_fixed, y_pred_c_fixed, title=f"{model_name} CM: Clean (Thr={threshold:.2f})", filename=os.path.join(output_dir, "binary_cm_clean.png"), target_name=target_name)
    plot_cm(y_true_fixed, y_pred_n_fixed, title=f"{model_name} CM: Noisy (Thr={threshold:.2f})", filename=os.path.join(output_dir, "binary_cm_noisy.png"), target_name=target_name)
    
    plot_roc(y_true_fixed, y_prob_c_fixed, threshold, title=f"{model_name} ROC: Clean", filename=os.path.join(output_dir, "roc_clean.png"))
    plot_roc(y_true_fixed, y_prob_n_fixed, threshold, title=f"{model_name} ROC: Noisy", filename=os.path.join(output_dir, "roc_noisy.png"))
    
    plot_boxplots(groups_clean, groups_noisy, title=f"{model_name} Metrics CI (Normal vs {target_name})", filename=os.path.join(output_dir, "boxplots_binary.png"))
    plot_comparison_bar(groups_clean, groups_noisy, title=f"{model_name} Robustness Comparison", output_dir=output_dir)
    print_metrics(groups_clean, groups_noisy, model_name, target_name)


# --- ГЛАВНЫЙ ЗАПУСК ---
def main():
    records_file = os.path.join(MYDB_PATH, 'RECORDS')
    if not os.path.exists(records_file):
        print(f"Файл RECORDS не найден в {MYDB_PATH}"); return
        
    with open(records_file, 'r') as f:
        all_records = [line.strip() for line in f if line.strip().startswith('II/')]
        
    print(f"Всего записей в mydb: {len(all_records)}")
    
    classifier = HolterClassifier(models_dir='models')
    
    print("\n=== Последовательная оценка моделей на их тестовых сплитах ===")
    
    for model_name, config in MODEL_CONFIGS.items():
        split_path = os.path.join(SPLITS_DIR, config['split_file'])
        if not os.path.exists(split_path):
            print(f"Файл сплита {split_path} не найден! Пропуск модели {model_name}")
            continue
            
        with open(split_path, 'r') as f:
            test_patients = set(line.strip() for line in f)
            
        test_records = [r for r in all_records if r.split('/')[1] in test_patients]
        print(f"\n[{model_name}] Тестовых пациентов: {len(test_patients)}, Записей: {len(test_records)}")
        
        # Раздельные пулы для Normal и Target
        pool_probs_c_normal = []
        pool_probs_c_target = []
        pool_probs_n_normal = []
        pool_probs_n_target = []
        
        for rec_path in test_records:
            full_path = os.path.join(MYDB_PATH, rec_path)
            if not os.path.exists(full_path + '.atr'): continue
            try:
                sig_clean = Signal(record_path=full_path)
                sig_noisy = add_noise_to_signal(sig_clean)
                
                raw_preds_c, _ = classifier._prepare_raw_predictions(sig_clean)
                raw_preds_n, _ = classifier._prepare_raw_predictions(sig_noisy)
                
                pred_map_c = {p['peak']: p for p in raw_preds_c}
                pred_map_n = {p['peak']: p for p in raw_preds_n}
                
                for ann in sig_clean.annotations:
                    sym = ann['symbol']
                    peak = ann['sample']
                    
                    pc = pred_map_c.get(peak)
                    pn = pred_map_n.get(peak)
                    if not pc or not pn: continue
                    
                    if config['target'] == 'PSS':
                        if sym in PSS_SYMBOLS: yt = 1
                        elif sym in NORMAL_PSS_SYMBOLS: yt = 0
                        else: continue
                    else:
                        if sym in BLK_SYMBOLS: yt = 1
                        elif sym in NORMAL_BLK_SYMBOLS: yt = 0
                        else: continue
                        
                    # Раскидываем вероятности по корзинам
                    if yt == 0:
                        pool_probs_c_normal.append(pc[f'prob_{model_name}'])
                        pool_probs_n_normal.append(pn[f'prob_{model_name}'])
                    else:
                        pool_probs_c_target.append(pc[f'prob_{model_name}'])
                        pool_probs_n_target.append(pn[f'prob_{model_name}'])
                    
            except Exception as e:
                print(f"Ошибка {rec_path}: {e}")
                
        # Переводим в numpy для скорости
        pool_probs_c_normal = np.array(pool_probs_c_normal)
        pool_probs_c_target = np.array(pool_probs_c_target)
        pool_probs_n_normal = np.array(pool_probs_n_normal)
        pool_probs_n_target = np.array(pool_probs_n_target)
        
        print(f"[{model_name}] Собрано комплексов: Normal={len(pool_probs_c_normal)}, Target={len(pool_probs_c_target)}")
        
        if len(pool_probs_c_normal) > 0 and len(pool_probs_c_target) > 0:
            thr = THRESHOLDS[model_name]
            target_name = 'Blockade' if config['target'] == 'BLK' else 'V'
            output_dir = os.path.join(RESULTS_ROOT, 'Cascade', model_name)
            
            evaluate_model(pool_probs_c_normal, pool_probs_c_target, pool_probs_n_normal, pool_probs_n_target, thr, target_name, model_name, output_dir)
        else:
            print(f"Нет данных для оценки {model_name}!")
    
    print("\n=== ВСЕ ОЦЕНКИ ЗАВЕРШЕНЫ ===")

if __name__ == "__main__":
    main()