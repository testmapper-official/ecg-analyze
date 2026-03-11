from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class AnalysisWorker(QThread):
    """
    Воркер для выполнения анализа ECG в отдельном потоке.
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

            # 3. Предсказание (пакетная обработка или цикл с прогрессом)
            # TensorFlow predict работает быстро, но если сегментов очень много, можно разбить
            results = self.classifier.predict(segments)
            self.progress.emit(90)
            
            # 4. Формирование результатов
            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append({
                    'sample': valid_indices[i],
                    'label': res['label'],
                    'probability': res['confidence'],
                    'fs': self.fs
                })
            
            # 5. Фильтрация "Нормы" (Исключаем Normal)
            # Предполагаем, что метка нормального ритма точно 'Normal'
            pathologies_only = [r for r in formatted_results if r['label'] != 'Normal']
            
            self.progress.emit(100)
            self.finished.emit(pathologies_only)
            
        except Exception as e:
            self.error.emit(str(e))