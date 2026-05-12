from PyQt5.QtCore import QThread, pyqtSignal
from app.core.signal import Signal
from app.core.holter_classifier import HolterClassifier

class AnalysisWorker(QThread):
    progress_step = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, signal_data, fs, models_dir='models', existing_r_peaks=None, pss_model_type='TCN', blk_model_type='TCN'):
        super().__init__()
        self.signal_data = signal_data
        self.fs = fs
        self.models_dir = models_dir
        self.existing_r_peaks = existing_r_peaks
        self.pss_model_type = pss_model_type
        self.blk_model_type = blk_model_type

    def run(self):
        try:
            self.progress_step.emit("Инициализация сигнала...")
            self.progress_percent.emit(20)
            
            sig = Signal(data=self.signal_data, fs=self.fs)
            
            if self.existing_r_peaks is not None and len(self.existing_r_peaks) > 0:
                self.progress_step.emit("Использование существующих R-пиков...")
                sig.annotations = [{'sample': int(p), 'symbol': 'N'} for p in self.existing_r_peaks]
            
            self.progress_step.emit(f"Запуск классификатора (ПЖС: {self.pss_model_type}, БЛК: {self.blk_model_type})...")
            self.progress_percent.emit(50)
            
            # Передаем выбранные модели в HolterClassifier
            classifier = HolterClassifier(
                models_dir=self.models_dir, 
                pss_model_type=self.pss_model_type, 
                blk_model_type=self.blk_model_type
            )
            
            # Анализ сигнала (классификатор внутри сам вызывает движок ритмов,
            # но из results_clean мы берем только одиночные комплексы, 
            # т.к. в UI ритмы пересчитываются динамически в _rebuild_merged_annotations)
            results_clean, _ = classifier.analyze_signal(sig)
            
            self.progress_step.emit("Формирование результатов...")
            self.progress_percent.emit(90)
            
            unified_output = []
            for p in results_clean:
                unified_output.append({
                    'sample': p['sample'], 
                    'symbol': p['label'], 
                    'auto_symbol': p['label'], 
                    'manual_symbol': '', 
                    'is_rhythm': False
                })
                
            unified_output.sort(key=lambda x: x.get('sample', 0))
            
            self.progress_percent.emit(100)
            self.finished.emit(unified_output)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
