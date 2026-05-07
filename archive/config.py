# -*- coding: utf-8 -*-
"""Общие константы и параметры Технического Задания (ТЗ)"""

# --- ТЗ ПАРАМЕТРЫ ---
TARGET_FS = 360
SEGMENT_SAMPLES = 288
DB_ROOT = 'DB'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

TARGET_TRAIN = 3500
TARGET_VAL = 750
TARGET_TEST = 750

# --- КЛАССЫ ---
TCN_CLASSES = ['Normal', 'Blockade', 'PVC']
TCN_TO_IDX = {name: i for i, name in enumerate(TCN_CLASSES)}

MORPH_CLASSES = ['Normal', 'Blockade', 'R_on_T', 'PVC_Interpolated', 'PVC_Monomorphic', 'PVC_Polymorphic']
MORPH_TO_IDX = {name: i for i, name in enumerate(MORPH_CLASSES)}

RHYTHM_CLASSES = ['Normal_Sinus', 'Bigeminy', 'Trigeminy', 'Ventricular_Tachycardia']

# --- ПОРОГИ ---
PREMATURE_THRESHOLD = 0.85
VT_RATE_THRESHOLD = 600

# --- НАСТРОЙКИ ПЛАГИНА V5 (СТРОГО НОРМА VS БЛОКАДА) ---
V5_MODULE_ENABLED = True
V5_CLASSES = ['Normal', 'Blockade'] # УБРАНЫ ПЖС
V5_TO_IDX = {name: i for i, name in enumerate(V5_CLASSES)}