# app/core/classifier.py
import os
import numpy as np
# ВАЖНО: Не импортируем tensorflow и load_model в начале файла!
# Это предотвращает конфликт DLL при инициализации приложения.

from app import BASE_DIR

CLASS_LABELS = [
    'Normal', 'Ventricular_Ectopic', 'Ventricular_Tachycardia', 'LBBB', 'RBBB',
    'Early_Repolarization', 'Brugada_Syndrome', 'LVH', 'Bigeminy', 'Trigeminy',
    'Bidirectional_VT', 'Posterior_MI', 'AIVR', 'Fragmented_QRS', 'Low_Voltage_QRS', 'R_on_T'
]

class ECGClassifier:
    def __init__(self, model_name='model1.h5'):
        self.model = None
        self.model_path = os.path.join(BASE_DIR, 'models', model_name)
        self.labels = CLASS_LABELS
        # Не загружаем модель в __init__, чтобы не тормозить запуск
        
    def load_model(self):
        """Загрузка модели. TensorFlow импортируется только здесь."""
        if self.model is not None:
            return True

        try:
            # Ленивый импорт TensorFlow
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            print("Загрузка TensorFlow...")
            
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"Модель успешно загружена: {self.model_path}")
                return True
            else:
                print(f"Файл модели не найден: {self.model_path}")
                return False
        except Exception as e:
            print(f"Ошибка при загрузке модели или TensorFlow: {e}")
            return False

    def predict(self, segments):
        """
        Предсказание классов.
        Сначала проверяет, загружена ли модель.
        """
        if self.model is None:
            # Пытаемся загрузить "на лету", если еще не загружено
            if not self.load_model():
                return []

        try:
            # Импорт для работы с массивами, если tf еще не импортирован глобально
            if len(segments.shape) == 2:
                segments = segments.reshape(-1, 288, 1)

            preds = self.model.predict(segments, verbose=0)
            results = []
            
            for i, pred in enumerate(preds):
                class_idx = np.argmax(pred)
                confidence = np.max(pred) * 100
                label = self.labels[class_idx]
                
                results.append({
                    'label': label,
                    'confidence': confidence,
                    'class_idx': class_idx
                })
            return results
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return []