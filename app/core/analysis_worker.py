from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class AnalysisWorker(QThread):
    """
    Воркер для выполнения анализа ECG в отдельном потоке.
    Интегрирован с обновленным ECGClassifier (PyTorch + Rule-based).
    """
    progress = pyqtSignal(int)  # Сигнал прогресса (0-100)
    finished = pyqtSignal(list) # Сигнал завершения с результатами
    error = pyqtSignal(str)     # Сигнал ошибки

    def __init__(self, signal, fs, processor, classifier):
        super().__init__()
        self.signal = signal
        self.fs = fs
        self.processor = processor
        self.classifier = classifier

    def run(self):
        try:
            # 1. Детекция R-пиков
            self.progress.emit(10)
            r_peaks = self.processor.detect_r_peaks(self.signal)
            self.progress.emit(30)
            
            if len(r_peaks) == 0:
                self.finished.emit([])
                return

            # 2. Сегментация
            segments, valid_indices = self.processor.get_segments(self.signal, r_peaks)
            self.progress.emit(50)
            
            if len(segments) == 0:
                self.finished.emit([])
                return

            # 3. ПОДГОТОВКА МЕТАДАННЫХ ДЛЯ RULE-BASED АЛГОРИТМОВ
            # Вычисление RR-интервалов ТОЛЬКО для валидных сегментов (по valid_indices)
            valid_rr_samples = np.diff(valid_indices)
            valid_rr_ms = (valid_rr_samples / self.fs) * 1000.0
            valid_rr_mean_ms = np.mean(valid_rr_ms) if len(valid_rr_ms) > 0 else 800.0

            rr_meta = []
            # Итерируемся строго по количеству готовых сегментов!
            for i in range(len(segments)):
                prev_rr = valid_rr_ms[i-1] if i > 0 else valid_rr_mean_ms
                next_rr = valid_rr_ms[i] if i < len(valid_rr_ms) else valid_rr_mean_ms
                
                # Извлекаем форму комплекса для сравнения мономорфности
                raw_morph = segments[i].flatten() 
                
                rr_meta.append({
                    'rr_prev': prev_rr,
                    'rr_next': next_rr,
                    'rr_mean': valid_rr_mean_ms,
                    'raw_morph': raw_morph
                })

            # 4. Проверка размерности для PyTorch (ожидают (N, 1, 288))
            if segments.ndim == 2:
                segments = segments.reshape(-1, 1, 288)

            # 5. Предсказание (Передаем сегменты и метаданные)
            results = self.classifier.predict(segments, rr_meta)
            self.progress.emit(90)
            
            # 6. Формирование результатов
            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append({
                    'sample': valid_indices[i],
                    'label': res['label'],
                    'probability': res['confidence'],
                    'fs': self.fs
                })
            
            # 7. Фильтрация "Нормы" (Оставляем только патологии для отрисовки маркеров)
            pathologies_only = [
                r for r in formatted_results 
                if r['label'] != 'Normal'
            ]
            
            self.progress.emit(100)
            self.finished.emit(pathologies_only)
            
        except Exception as e:
            # Выводим полную ошибку в консоль для удобства отладки
            import traceback
            print(f"Ошибка в AnalysisWorker: {e}")
            traceback.print_exc()
            self.error.emit(str(e))