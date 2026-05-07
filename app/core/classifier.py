# app/core/classifier.py
import os
import numpy as np
from scipy.spatial.distance import cosine
# ВАЖНО: Не импортируем torch в начале файла!

from app import BASE_DIR

# Финальный список классов (исключены все, кроме нормальных ритмов и экстрасистолий)
CLASS_LABELS = [
    'Normal', 'R_on_T', 'PVC_Interpolated', 
    'PVC_Monomorphic', 'PVC_Polymorphic', 
    'Bigeminy', 'Trigeminy'
]

# ==========================================================
# АРХИТЕКТУРА TCN (Должна строго совпадать с classify2.py)
# ==========================================================
import torch
import torch.nn as nn

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
        self.fc = nn.Linear(channels[-1], 2) # 2 класса: 0-Normal, 1-PVC
    def forward(self, x):
        return self.fc(self.pool(self.network(x)).squeeze(-1))
# ==========================================================


class ECGClassifier:
    def __init__(self, model_name='best_tcn.pth'):
        self.model = None
        self.model_path = os.path.join(BASE_DIR, 'models', model_name)
        self.labels = CLASS_LABELS
        self.device = torch.device('cpu') # Для Холтера всегда CPU
        self.pvc_morph_ref = None # Эталон формы для проверки Мономорфности
        
    def load_model(self):
        """Ленивая загрузка PyTorch модели."""
        if self.model is not None:
            return True

        try:
            print("Инициализация PyTorch...")
            if os.path.exists(self.model_path):
                self.model = TCNClassifier().to(self.device)
                # map_location='cpu' обязателен для работы на машинах без видеокарты
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval() # Перевод в режим предсказания
                print(f"Модель успешно загружена: {self.model_path}")
                return True
            else:
                print(f"Файл модели не найден: {self.model_path}")
                return False
        except Exception as e:
            print(f"Ошибка при загрузке модели PyTorch: {e}")
            return False

    def _refine_pvc(self, meta):
        """Ручные методы (Rule-based) для детализации ПЖС"""
        rr_prev, rr_next, rr_mean = meta['rr_prev'], meta['rr_next'], meta['rr_mean']
        raw_morph = meta['raw_morph']
        
        # 1. R-on-T (Интервал RR' меньше 80% от расчетного QT)
        qt_est = 0.4 * np.sqrt(rr_mean / 1000) * 1000
        if rr_prev < (0.8 * qt_est):
            return self.labels.index('R_on_T')
            
        # 2. Интерполированная (Отсутствие компенсаторной паузы)
        if (rr_prev + rr_next) < (2.2 * rr_mean):
            return self.labels.index('PVC_Interpolated')
            
        # 3. Мономорфная / Полиморфная (Косинусное сходство)
        if self.pvc_morph_ref is None:
            self.pvc_morph_ref = raw_morph # Запоминаем первую ПЖС как эталон
            
        similarity = 1 - cosine(self.pvc_morph_ref, raw_morph)
        if similarity > 0.7:
            return self.labels.index('PVC_Monomorphic')
        else:
            return self.labels.index('PVC_Polymorphic')

    def _detect_sequences(self, base_preds):
        """
        Поиск Бигеминии (N V N V) и Тригеминии (N N V N N V) 
        в массиве базовых предсказаний нейросети (0=Normal, 1=PVC).
        """
        seq_labels = [None] * len(base_preds)
        i = 0
        n = len(base_preds)
        
        while i < n:
            # --- Поиск Тригеминии (N N V) ---
            if (i + 2 < n and 
                base_preds[i] == 0 and base_preds[i+1] == 0 and base_preds[i+2] == 1):
                
                # Проверяем, повторяется ли паттерн (N N V N N V)
                if (i + 5 < n and 
                    base_preds[i+3] == 0 and base_preds[i+4] == 0 and base_preds[i+5] == 1):
                    seq_labels[i] = self.labels.index('Normal')
                    seq_labels[i+1] = self.labels.index('Normal')
                    seq_labels[i+2] = self.labels.index('Trigeminy')
                    seq_labels[i+3] = self.labels.index('Normal')
                    seq_labels[i+4] = self.labels.index('Normal')
                    seq_labels[i+5] = self.labels.index('Trigeminy')
                    i += 6
                    continue
            
            # --- Поиск Бигеминии (N V) ---
            elif (i + 1 < n and 
                  base_preds[i] == 0 and base_preds[i+1] == 1):
                
                # Проверяем, повторяется ли паттерн (N V N V)
                if (i + 3 < n and 
                    base_preds[i+2] == 0 and base_preds[i+3] == 1):
                    seq_labels[i] = self.labels.index('Normal')
                    seq_labels[i+1] = self.labels.index('Bigeminy')
                    seq_labels[i+2] = self.labels.index('Normal')
                    seq_labels[i+3] = self.labels.index('Bigeminy')
                    i += 4
                    continue
            
            # Если паттернов нет, оставляем как есть (обработка ниже в predict)
            i += 1
            
        return seq_labels

    def predict(self, segments, rr_meta):
        """
        Предсказание классов.
        
        ПАРАМЕТРЫ:
        - segments: np.array формы (N, 1, 288) - нормализованные сегменты
        - rr_meta: list из словарей длиной N. 
          Каждый словарь ДОЛЖЕН содержать:
          {'rr_prev': float, 'rr_next': float, 'rr_mean': float, 'raw_morph': np.array(288)}
        """
        if self.model is None:
            if not self.load_model():
                return []

        if len(segments) != len(rr_meta):
            print("Ошибка: Количество сегментов не совпадает с количеством RR-метаданных.")
            return []

        try:
            # 1. Базовое предсказание TCN (0 - Normal, 1 - PVC)
            with torch.no_grad():
                inputs = torch.FloatTensor(segments).to(self.device)
                logits = self.model(inputs)
                base_preds = np.argmax(logits.cpu().numpy(), axis=1)

            # 2. Попытка найти последовательности (Бигеминия/Тригеминия)
            # Функция вернет None для тех индексов, где паттерн не найден
            seq_labels = self._detect_sequences(base_preds)

            results = []
            
            # 3. Финальная сборка результатов
            for i in range(len(base_preds)):
                # Если последовательность найдена, берем метку из нее
                if seq_labels[i] is not None:
                    final_label_idx = seq_labels[i]
                    confidence = 95.0 # Даем высокий процент доверия для четких паттернов
                else:
                    # Если это норма от нейросети
                    if base_preds[i] == 0:
                        final_label_idx = self.labels.index('Normal')
                        confidence = np.max(logits.cpu().numpy()[i]) * 100
                    # Если это ПЖС от нейросети -> запускаем ручные методы
                    else:
                        final_label_idx = self._refine_pvc(rr_meta[i])
                        confidence = 90.0 # Базовое доверие к rule-based алгоритмам

                results.append({
                    'label': self.labels[final_label_idx],
                    'confidence': round(confidence, 1),
                    'class_idx': final_label_idx
                })
                
            return results
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return []