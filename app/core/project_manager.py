import os
import json
import shutil
import time
import wfdb
import numpy as np

class ProjectManager:
    def __init__(self):
        self.projects_dir = "projects"
        self.history_file = os.path.join(self.projects_dir, "history.json")
        os.makedirs(self.projects_dir, exist_ok=True)
        self.current_project_dir = None
        self.record_name = ""

    def create_project(self, file_path):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.record_name = base_name
        self.current_project_dir = os.path.join(self.projects_dir, base_name)
        os.makedirs(self.current_project_dir, exist_ok=True)

        # ИСПРАВЛЕНО: Надежное формирование путей для копирования
        # Отсекаем расширение у исходного файла, чтобы подставить нужное
        base_src_path = os.path.splitext(file_path)[0]
        
        for ext in ['.dat', '.hea', '.atr']:
            src = base_src_path + ext
            if os.path.exists(src):
                dst = os.path.join(self.current_project_dir, base_name + ext)
                shutil.copy2(src, dst)

        self.update_history(base_name)
        return self.current_project_dir

    def update_history(self, name):
        history = self.get_history()
        # Удаляем старую запись, если она есть
        history = [h for h in history if (h['name'] if isinstance(h, dict) else h) != name]
        # Добавляем новую с текущим временем
        history.insert(0, {"name": name, "last_opened": time.time()})
        # Оставляем только 10 последних
        history = history[:10]
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)

    def get_history(self):
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def get_atr_path(self):
        return os.path.join(self.current_project_dir, self.record_name + '.atr')

    def get_auto_atr_path(self):
        return os.path.join(self.current_project_dir, self.record_name + '.autoatr')

    def save_auto_annotations(self, annotations_360hz, orig_fs):
        if not annotations_360hz: return
        ratio = orig_fs / 360.0
        samples = np.array([int(ann['sample'] * ratio) for ann in annotations_360hz], dtype=np.int64)
        symbols = [ann.get('auto_symbol', ann.get('symbol', 'N')) for ann in annotations_360hz]
        
        # wfdb.wrann корректно записывает только файл аннотаций (.autoatr), не трогая .hea
        wfdb.wrann(self.record_name, 'autoatr', sample=samples, symbol=symbols, write_dir=self.current_project_dir)

    def load_auto_annotations(self, orig_fs):
        auto_path = self.get_auto_atr_path().replace('.autoatr', '')
        if os.path.exists(auto_path + '.autoatr'):
            try:
                ann = wfdb.rdann(auto_path, 'autoatr')
                ratio = 360.0 / orig_fs
                return [{'sample': int(s * ratio), 'auto_symbol': sym, 'manual_symbol': ''} for s, sym in zip(ann.sample, ann.symbol)]
            except Exception as e:
                print(f"Ошибка чтения авто-аннотаций: {e}")
        return []

    def save_manual_annotation(self, annotations_360hz, orig_fs):
        if not annotations_360hz: return
        ratio = orig_fs / 360.0
        samples = np.array([int(ann['sample'] * ratio) for ann in annotations_360hz], dtype=np.int64)
        # Сохраняем именно то, что отметил врач
        symbols = [ann.get('manual_symbol', ann.get('symbol', 'N')) for ann in annotations_360hz]
        
        # wfdb.wrann корректно перезаписывает только .atr
        wfdb.wrann(self.record_name, 'atr', sample=samples, symbol=symbols, write_dir=self.current_project_dir)

    def export_merged_signal(self, signal_data, fs, merged_annotations_360hz, orig_fs, export_path):
        ratio = orig_fs / 360.0
        samples = np.array([int(ann['sample'] * ratio) for ann in merged_annotations_360hz], dtype=np.int64)
        symbols = [ann.get('manual_symbol') if ann.get('manual_symbol') else ann.get('auto_symbol') for ann in merged_annotations_360hz]
        
        wfdb.wrsamp(os.path.basename(export_path), fs=fs, units=['mV'], sig_name=['MLII'], p_signal=signal_data.reshape(-1, 1), write_dir=os.path.dirname(export_path))
        wfdb.wrann(os.path.basename(export_path), 'atr', sample=samples, symbol=symbols, write_dir=os.path.dirname(export_path))