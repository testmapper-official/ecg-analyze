# app/core/data_loader.py
import numpy as np
import os

class DataLoader:
    @staticmethod
    def load_file(file_path):
        """
        Универсальный метод загрузки.
        Возвращает словарь: {'signal': np.array, 'fs': int, 'leads': list, 'annotations': list}
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.dat', '.hea']: # WFDB format
            return DataLoader._load_wfdb(file_path)
        elif ext == '.edf': # EDF format
            return DataLoader._load_edf(file_path)
        elif ext == '.csv': # CSV format
            return DataLoader._load_csv(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    @staticmethod
    def _load_wfdb(file_path):
        import wfdb
        # Убираем расширение для wfdb
        record_path = file_path.replace('.dat', '').replace('.hea', '')
        try:
            record = wfdb.rdrecord(record_path)
            # Попытка загрузить аннотации
            try:
                ann = wfdb.rdann(record_path, 'atr')
                annotations = list(zip(ann.sample, ann.symbol))
            except:
                annotations = []

            return {
                'signal': record.p_signal,
                'fs': record.fs,
                'leads': record.sig_name,
                'annotations': annotations,
                'units': record.units
            }
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения WFDB: {e}")

    @staticmethod
    def _load_edf(file_path):
        try:
            import pyedflib
            f = pyedflib.EdfReader(file_path)
            n_channels = f.signals_in_file
            signal = np.zeros((f.getNSamples()[0], n_channels))
            for i in range(n_channels):
                signal[:, i] = f.readSignal(i)
            
            return {
                'signal': signal,
                'fs': f.getSampleFrequency(0), # Берем частоту первого канала
                'leads': f.getSignalLabels(),
                'annotations': [], # EDF annotations parsing is more complex
                'units': [f.getLabel(i) for i in range(n_channels)]
            }
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения EDF: {e}")

    @staticmethod
    def _load_csv(file_path):
        # Предполагаем, что первая строка - заголовок, первый столбец - время или просто сэмплы
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            # Если данные одномерные, превращаем в 2D
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            return {
                'signal': data,
                'fs': 250, # Дефолтное значение для CSV, нужно уточнять у пользователя
                'leads': [f'Lead {i+1}' for i in range(data.shape[1])],
                'annotations': [],
                'units': ['mV'] * data.shape[1]
            }
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения CSV: {e}")