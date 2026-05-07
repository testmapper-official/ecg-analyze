# -*- coding: utf-8 -*-
"""Главная точка входа: Сборка Pipeline с плагином V5"""

import os, sys, logging, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from collections import Counter

import archive.config as config
from archive.stage1_tcn import train_tcn, TCNClassifier
from archive.stage_v5_blockade import train_v5_plugin, V5Net # НОВЫЙ ИМПОРТ
from archive.stage2_morphology import MorphologyClassifier
from archive.stage3_rhythm import RhythmAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore")
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def run_pipeline():
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("ЗАПУСК КАСКАДА С МОДУЛЕМ ВЕРИФИКАЦИИ V5")
    logger.info("="*50)

    # ==========================================
    # ЭТАП 1: Обучение базового TCN (Lead II)
    # ==========================================
    model_ii, X_test_ii, y_test_ii, meta_test = train_tcn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ii.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'stage1_tcn.pth')))
    model_ii.eval()

    with torch.no_grad():
        logits_ii = model_ii(X_test_ii.to(device))
        tcn_ii_preds_idx = np.argmax(logits_ii.cpu().numpy(), axis=1)
    tcn_ii_preds_str = [config.TCN_CLASSES[i] for i in tcn_ii_preds_idx]

    logger.info("ОТЧЕТ ЭТАП 1: Lead II")
    print(classification_report(y_test_ii, tcn_ii_preds_idx, target_names=config.TCN_CLASSES, zero_division=0))

    # ==========================================
    # ПЛАГИН: Обучение модуля V5 (если есть данные)
    # ==========================================
    v5_preds_str = None
    if config.V5_MODULE_ENABLED:
        model_v5, X_test_v5, y_test_v5 = train_v5_plugin()
        
        if model_v5 is not None:
            model_v5.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'v5_plugin.pth')))
            model_v5.eval()
            
            with torch.no_grad():
                logits_v5 = model_v5(X_test_v5.to(device))
                v5_preds_idx = np.argmax(logits_v5.cpu().numpy(), axis=1)
            v5_preds_str = [config.V5_CLASSES[i] for i in v5_preds_idx]
            
            logger.info("ОТЧЕТ ПЛАГИНА V5")
            print(classification_report(y_test_v5, v5_preds_idx, target_names=config.V5_CLASSES, zero_division=0))

    # ==========================================
    # ЭТАП 2: Каскадная морфология (II + V5)
    # ==========================================
    morph_classifier = MorphologyClassifier()
    morph_preds_idx = []
    
    for i in range(len(tcn_ii_preds_str)):
        m = meta_test[i]
        raw_morph = X_test_ii[i, 0, :].cpu().numpy()
        
        # Берем предсказание V5, если оно есть, иначе None
        current_v5_pred = v5_preds_str[i] if v5_preds_str is not None and i < len(v5_preds_str) else None
        
        pred_idx = morph_classifier.process_beat(
            tcn_ii_class_str=tcn_ii_preds_str[i],
            v5_class_str=current_v5_pred, # ПЕРЕДАЕМ V5 В КАЧЕСТВЕ АРБИТРА
            rr_prev=m['rr_prev'],
            rr_mean=m['rr_mean'],
            raw_morph=raw_morph
        )
        morph_preds_idx.append(pred_idx)
        
    morph_preds_idx = np.array(morph_preds_idx)
    morph_preds_str = [config.MORPH_CLASSES[i] for i in morph_preds_idx]

    logger.info("ОТЧЕТ ЭТАП 2: Итоговая классификация (с учетом V5)")
    morph_counts = Counter(morph_preds_str)
    print("Распределение предсказанных классов:")
    for cls in config.MORPH_CLASSES:
        print(f" - {cls}: {morph_counts.get(cls, 0)} ударов")

    # ==========================================
    # ЭТАП 3: Анализ ритмов
    # ==========================================
    rhythm_classifier = RhythmAnalyzer()
    rhythm_preds = [rhythm_classifier.process_beat(morph_preds_str[i], meta_test[i]['rr_prev']) for i in range(len(morph_preds_str))]

    logger.info("ОТЧЕТ ЭТАП 3: Ритм")
    for cls, cnt in Counter(rhythm_preds).items():
        print(f" - {cls}: {cnt} ударов")

    # ==========================================
    # ГРАФИКИ
    # ==========================================
    logger.info("Генерация графиков...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Матрица II
    cm1 = confusion_matrix(y_test_ii, tcn_ii_preds_idx)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Greens', xticklabels=config.TCN_CLASSES, yticklabels=config.TCN_CLASSES)
    plt.title("Этап 1: Lead II"); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, '1_cm_tcn_ii.png'), dpi=150); plt.close()

    # 2. Матрица V5 (если есть)
    if v5_preds_str is not None:
        cm_v5 = confusion_matrix(y_test_v5, v5_preds_idx)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_v5, annot=True, fmt='d', cmap='Purples', xticklabels=config.V5_CLASSES, yticklabels=config.V5_CLASSES)
        plt.title("Плагин: Отведение V5"); plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, '2_cm_v5_plugin.png'), dpi=150); plt.close()

    # 3. Итоговые примеры морфологии
    found_idx = [i for i in range(len(config.MORPH_CLASSES)) if np.sum(morph_preds_idx == i) > 0]
    if len(found_idx) > 0:
        fig, axes = plt.subplots(len(found_idx), 5, figsize=(20, 3.5 * len(found_idx)))
        if len(found_idx) == 1: axes = axes.reshape(1, -1)
        for row, cls_idx in enumerate(found_idx):
            indices = np.where(morph_preds_idx == cls_idx)[0]
            samples = np.random.choice(indices, min(5, len(indices)), replace=False)
            for j, idx in enumerate(samples):
                axes[row, j].plot(X_test_ii[idx, 0, :].cpu().numpy(), 'k')
                axes[row, j].axis('off')
                axes[row, j].set_title(config.MORPH_CLASSES[cls_idx], fontsize=14, color='darkred', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, '3_morph_final_examples.png'), dpi=150); plt.close()

    logger.info("КАСКАД С V5 ЗАВЕРШЕН.")

if __name__ == "__main__":
    run_pipeline()