import os
import numpy as np
import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QFileDialog, QMessageBox, QProgressBar, QLabel, QPushButton, QProgressDialog, QMenuBar, QAction, QSizeGrip)
from PyQt5.QtCore import Qt

from app.ui.toolbar import AppToolBar
from app.ui.custom_title_bar import CustomTitleBar
from app.ui.ecg_viewer import ECGViewer
from app.ui.navigation_plot import NavigationPlot
from app.ui.annotation_grid import AnnotationGrid
from app.ui.pathology_panel import PathologyPanel

from app.core.data_loader import DataLoader
from app.core.project_manager import ProjectManager
from app.core.analysis_worker import AnalysisWorker
from app.core.r_detection_worker import RDetectionWorker

DANGER_MAP = {
    'r': 3, 'VT': 3, 'VF': 3, 'Couplet': 3, 
    'B': 2, 'L': 2, 'R': 2, 'SVT': 2, 'SB': 2, 'Bigeminy': 2, 'Trigeminy': 2, 
    'V': 1, 'M': 1, 'P': 1, 'i': 1, 'A': 1, 'F': 1, 'E': 1, 'e': 1   
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holter Monitor")
        self.resize(1400, 800)

        # Включаем безрамочное окно для использования кастомного титула
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.project_mgr = ProjectManager()
        self.current_signal_data = None
        self.orig_fs = 360
        self.current_lead_idx = 0
        
        self.manual_annotations_360 = []
        self.auto_annotations_360 = []
        self.merged_annotations_360 = []
        self.avg_rr_orig = 360 
        self.view_auto_mode = False

        self.init_ui()
        self.connect_signals()
        self._update_recent_menus()

    def init_ui(self):
        # Центральный виджет и главный вертикальный лейаут
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_v_layout = QVBoxLayout(central_widget)
        main_v_layout.setContentsMargins(0, 0, 0, 0)
        main_v_layout.setSpacing(0) # Убираем зазоры между титулом, меню и тулбаром

        # 1. Кастомный титулбар (Самый верхний элемент)
        self.custom_title_bar = CustomTitleBar(self)
        main_v_layout.addWidget(self.custom_title_bar)

        # 2. Менюбар (Файл и т.д.)
        self.menubar = QMenuBar(self)
        self.menubar.setNativeMenuBar(False) # Принудительно отрисовываем внутри окна
        self.menubar.setMaximumHeight(25)    # Фиксируем высоту, чтобы не прыгал стиль
        file_menu = self.menubar.addMenu("Файл")
        self.act_open_file = QAction("Открыть файл сигнала", self)
        self.act_open_project = QAction("Открыть проект", self)
        file_menu.addAction(self.act_open_file)
        file_menu.addAction(self.act_open_project)
        self.recent_menu = file_menu.addMenu("Недавние проекты")
        file_menu.addSeparator()
        self.act_exit = QAction("Выход", self)
        self.act_exit.triggered.connect(self.close)
        file_menu.addAction(self.act_exit)
        main_v_layout.addWidget(self.menubar)

        # 3. Тулбар с кнопками
        self.toolbar = AppToolBar()
        self.toolbar.setMovable(False)
        # Убираем стандартную обводку тулбара, чтобы он выглядел как часть интерфейса, а не плавающее окно
        self.toolbar.setStyleSheet("QToolBar { border: none; spacing: 5px; padding: 2px; }")
        main_v_layout.addWidget(self.toolbar)

        # 4. Основной контент (ЭКГ, панели)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5) # Небольшие отступы для красоты

        right_layout = QVBoxLayout()
        self.ecg_viewer = ECGViewer()
        self.annotation_grid = AnnotationGrid()
        
        nav_layout = QHBoxLayout()
        self.btn_left = QPushButton("◄")
        self.btn_left.setFixedSize(30, 30)
        self.btn_left.setAutoRepeat(True); self.btn_left.setAutoRepeatDelay(500); self.btn_left.setAutoRepeatInterval(100)
        self.btn_right = QPushButton("►")
        self.btn_right.setFixedSize(30, 30)
        self.btn_right.setAutoRepeat(True); self.btn_right.setAutoRepeatDelay(500); self.btn_right.setAutoRepeatInterval(100)

        self.navigation_plot = NavigationPlot()

        nav_layout.addWidget(self.btn_left)
        nav_layout.addWidget(self.navigation_plot, stretch=1)
        nav_layout.addWidget(self.btn_right)

        right_layout.addWidget(self.ecg_viewer, stretch=5)
        right_layout.addWidget(self.annotation_grid, stretch=1)
        right_layout.addLayout(nav_layout, stretch=2)

        self.pathology_panel = PathologyPanel()
        content_layout.addLayout(right_layout, stretch=4)
        content_layout.addWidget(self.pathology_panel, stretch=1)

        main_v_layout.addWidget(content_widget, stretch=1)

        # 5. Статусбар (остается стандартным, внизу)
        self.status_label = QLabel("Готово")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar, 1)
        self.statusBar().addPermanentWidget(self.status_label, 1)
        
        # Добавляем Grip для изменения размера окна в правом нижнем углу, т.к. рамок нет
        self.size_grip = QSizeGrip(self)
        self.statusBar().addPermanentWidget(self.size_grip, 0)

    def connect_signals(self):
        self.act_open_file.triggered.connect(self.open_file)
        self.act_open_project.triggered.connect(self.open_project)
        self.toolbar.save_project_clicked.connect(self.save_project)
        self.toolbar.detect_r_clicked.connect(self.detect_r_peaks)
        self.toolbar.analyze_clicked.connect(self.start_analysis)
        self.toolbar.export_wfdb_clicked.connect(self.export_wfdb)
        self.toolbar.lead_changed.connect(self.change_lead)
        self.toolbar.view_auto_toggled.connect(self.set_view_auto_mode)
        self.toolbar.visual_std_toggled.connect(self.toggle_visual_standardize) 
        
        self.navigation_plot.range_changed.connect(self.update_viewport)
        self.pathology_panel.jump_to_position.connect(self.jump_to_sample)
        self.ecg_viewer.zoom_requested.connect(self.handle_zoom)
        self.btn_left.clicked.connect(lambda: self.handle_scroll(-self.avg_rr_orig))
        self.btn_right.clicked.connect(lambda: self.handle_scroll(self.avg_rr_orig))
        self.annotation_grid.manual_label_changed.connect(self.update_manual_label)
        self.ecg_viewer.annotation_clicked.connect(self.on_annotation_clicked)

    def set_view_auto_mode(self, checked):
        self.view_auto_mode = checked
        self._rebuild_merged_annotations()

    def toggle_visual_standardize(self, checked):
        self.ecg_viewer.set_visual_standardize(checked)
        self.navigation_plot.set_visual_standardize(checked)
        if self.current_signal_data is not None:
            self.navigation_plot.set_signal(self.current_signal_data, self.orig_fs, self.current_lead_idx)
        start, end = self.navigation_plot.roi.getRegion()
        self.update_viewport(int(start), int(end))

    def _update_recent_menus(self):
        self.recent_menu.clear()
        history = self.project_mgr.get_history()
        if not history:
            action = self.recent_menu.addAction("Пусто"); action.setEnabled(False); return
        for item in history:
            if isinstance(item, str): proj_name = item; date_str = "Ранее"
            else: proj_name = item.get("name", "Unknown"); date_str = time.strftime('%d.%m.%Y %H:%M', time.localtime(item.get("last_opened", 0)))
            proj_path = os.path.join(self.project_mgr.projects_dir, proj_name, proj_name + '.hea')
            action = self.recent_menu.addAction(f"{proj_name} ({date_str})")
            action.triggered.connect(lambda checked, p=proj_path: self.load_project(p, create_new=False))

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть сигнал ЭКГ", "", "WFDB Files (*.hea *.dat)")
        if not file_path: return
        self.load_project(file_path, create_new=True)

    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "projects", "WFDB Files (*.hea)")
        if not file_path: return
        self.load_project(file_path, create_new=False)

    def load_project(self, file_path, create_new=False):
        try:
            if create_new: self.project_mgr.create_project(file_path)
            else:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                self.project_mgr.record_name = base_name
                self.project_mgr.current_project_dir = os.path.dirname(file_path)
                self.project_mgr.update_history(base_name)
            
            data = DataLoader._load_wfdb(file_path)
            self.current_signal_data = data['signal']
            self.orig_fs = data['fs']
            self.toolbar.set_leads(data['leads'])
            
            # Обновляем текст в кастомном титулбаре
            title_text = f"Holter Monitor - Запись: {self.project_mgr.record_name} | Частота: {self.orig_fs} Гц"
            self.custom_title_bar.title.setText(title_text)

            ratio = 360.0 / self.orig_fs
            self.manual_annotations_360 = [{'sample': int(s * ratio), 'manual_symbol': sym, 'auto_symbol': '', 'is_manual': True} for s, sym in data['annotations']]
            self.auto_annotations_360 = self.project_mgr.load_auto_annotations(self.orig_fs)
            for ann in self.auto_annotations_360: 
                ann['is_manual'] = False
                if 'manual_symbol' not in ann: ann['manual_symbol'] = ''
                if 'auto_symbol' not in ann: ann['auto_symbol'] = ann.get('symbol', '')

            self._rebuild_merged_annotations()
            self.navigation_plot.set_signal(self.current_signal_data, self.orig_fs, self.current_lead_idx)
            self.status_label.setText(f"Загружено: {os.path.basename(file_path)}")
            self._update_recent_menus()

            if not self.merged_annotations_360:
                QMessageBox.information(self, "Разметка отсутствует", "Для навигации необходимы R-пики. Будет запущена автоматическая детекция.")
                self.detect_r_peaks()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def save_project(self):
        if not self.project_mgr.current_project_dir: return
        try:
            manual_to_save = []
            for ann in self.manual_annotations_360:
                sym = ann['manual_symbol']
                if sym in ['M', 'P', 'i', 'r']: sym = 'V'
                manual_to_save.append({**ann, 'manual_symbol': sym})
                
            self.project_mgr.save_manual_annotation(manual_to_save, self.orig_fs)
            self.project_mgr.save_auto_annotations(self.auto_annotations_360, self.orig_fs)
            self.status_label.setText("Проект успешно сохранен!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def update_manual_label(self, sample_360, new_label):
        found = False
        for ann in self.manual_annotations_360:
            if abs(ann['sample'] - sample_360) < 18: ann['manual_symbol'] = new_label; found = True; break
        if not found:
            self.manual_annotations_360.append({'sample': sample_360, 'manual_symbol': new_label, 'auto_symbol': '', 'is_manual': True})
            self.manual_annotations_360.sort(key=lambda x: x['sample'])
        self._rebuild_merged_annotations()

    def _calculate_avg_rr(self):
        singles = [ann for ann in self.merged_annotations_360 if not ann.get('is_rhythm') and ann.get('sample')]
        if len(singles) > 2:
            diffs = np.diff([ann['sample'] for ann in singles])
            self.avg_rr_orig = int(np.median(diffs) * (self.orig_fs / 360.0))
        else: self.avg_rr_orig = int(0.8 * self.orig_fs)

    def _rebuild_merged_annotations(self):
        merged_dict = {}
        
        auto_beats = [a for a in self.auto_annotations_360 if not a.get('is_rhythm')]
        manual_beats = [m for m in self.manual_annotations_360 if not m.get('is_rhythm')]

        for m in manual_beats:
            key = m['sample']
            merged_dict[key] = {'sample': key, 'manual_symbol': m.get('manual_symbol', ''), 'auto_symbol': '', 'is_rhythm': False}

        tol = 18
        for a in auto_beats:
            found_key = None
            for mk in merged_dict.keys():
                if abs(mk - a['sample']) < tol: found_key = mk; break
            if found_key is not None:
                merged_dict[found_key]['auto_symbol'] = a.get('auto_symbol', a.get('symbol', ''))
            else:
                key = a['sample']
                merged_dict[key] = {'sample': key, 'manual_symbol': '', 'auto_symbol': a.get('auto_symbol', a.get('symbol', '')), 'is_rhythm': False}

        base_singles = list(merged_dict.values())
        base_singles.sort(key=lambda x: x['sample'])
        
        seq_for_detector = []
        for ann in base_singles:
            if self.view_auto_mode:
                active_label = ann.get('auto_symbol') if ann.get('auto_symbol') else ann.get('manual_symbol', 'N')
            else:
                active_label = ann.get('manual_symbol') if ann.get('manual_symbol') else ann.get('auto_symbol', 'N')
            ann['symbol'] = active_label
            seq_for_detector.append({'sample': ann['sample'], 'label': active_label})

        from app.core.holter_classifier import HolterClassifier
        seq_grouped = [{'sample': p['sample'], 'group': HolterClassifier._map_group(p['label'])} for p in seq_for_detector]
        detected_rhythms = HolterClassifier.detect_rhythms(seq_grouped)
        
        unified_output = list(base_singles)
            
        for r in detected_rhythms:
            unified_output.append({
                'is_rhythm': True, 'symbol': r['type'], 
                'start_sample': r['start_sample'], 'end_sample': r['end_sample'], 
                'danger': DANGER_MAP.get(r['type'], 1)
            })
            
        self.merged_annotations_360 = unified_output
        self.merged_annotations_360.sort(key=lambda x: x.get('start_sample', x.get('sample', 0)))
        
        self._calculate_avg_rr()
        self.pathology_panel.set_analysis_results(self.merged_annotations_360, self.view_auto_mode)
        start, end = self.navigation_plot.roi.getRegion()
        self.update_viewport(int(start), int(end))

    def update_viewport(self, start_sample_orig, end_sample_orig):
        if self.current_signal_data is None: return
        self.ecg_viewer.update_view(start_sample_orig, end_sample_orig, self.current_signal_data, self.orig_fs, self.current_lead_idx, self.avg_rr_orig)
        self.ecg_viewer.set_annotations_and_highlights(self.merged_annotations_360, start_sample_orig, end_sample_orig, self.view_auto_mode)
        plot_width = self.ecg_viewer.plot_widget.width()
        self.annotation_grid.update_grid(start_sample_orig, end_sample_orig, self.merged_annotations_360, self.orig_fs, plot_width, self.view_auto_mode)

    def change_lead(self, idx):
        self.current_lead_idx = idx
        if self.current_signal_data is not None:
            self.navigation_plot.set_signal(self.current_signal_data, self.orig_fs, idx)
            start, end = self.navigation_plot.roi.getRegion()
            self.update_viewport(int(start), int(end))

    def jump_to_sample(self, sample_360):
        ratio = self.orig_fs / 360.0
        center_orig = int(sample_360 * ratio)
        start_curr, end_curr = self.navigation_plot.roi.getRegion()
        width = int(end_curr - start_curr)
        start = max(0, center_orig - width // 2)
        end = start + width
        if end > len(self.current_signal_data): end = len(self.current_signal_data); start = end - width
        self.navigation_plot.update_roi_from_external(start, end)

    def handle_scroll(self, delta_samples_orig):
        start, end = self.navigation_plot.roi.getRegion()
        start, end = int(start) + delta_samples_orig, int(end) + delta_samples_orig
        max_len = len(self.current_signal_data) if self.current_signal_data is not None else 0
        width = int(end - start)
        if start < 0: start = 0; end = width
        if end > max_len: end = max_len; start = max_len - width
        self.navigation_plot.update_roi_from_external(start, end)

    def handle_zoom(self, delta_qrs_orig):
        start, end = self.navigation_plot.roi.getRegion()
        start, end = int(start) - delta_qrs_orig, int(end) + delta_qrs_orig
        max_len = len(self.current_signal_data) if self.current_signal_data is not None else 0
        min_width = int(0.5 * self.orig_fs)
        if start < 0: start = 0
        if end > max_len: end = max_len
        if (end - start) < min_width: center = (start + end) // 2; start = center - min_width // 2; end = center + min_width // 2
        self.navigation_plot.update_roi_from_external(start, end)

    def detect_r_peaks(self):
        if self.current_signal_data is None: return
        self.progress_dialog = QProgressDialog("Детекция R-пиков...", None, 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal); self.progress_dialog.setMinimumDuration(0); self.progress_dialog.forceShow()
        self.r_worker = RDetectionWorker(self.current_signal_data[:, self.current_lead_idx], self.orig_fs)
        self.r_worker.progress.connect(self.progress_dialog.setValue)
        self.r_worker.finished.connect(self._on_r_detection_finished)
        self.r_worker.start()

    def _on_r_detection_finished(self, r_peaks_360):
        self.progress_dialog.close()
        self.auto_annotations_360 = r_peaks_360
        self._rebuild_merged_annotations()
        self.status_label.setText("Детекция R-пиков завершена")

    def start_analysis(self):
        if self.current_signal_data is None: return
        self.progress_bar.setVisible(True)
        models_dir = 'models' 
        if not os.path.exists(os.path.join(models_dir, 'TCN_BLK.pth')): 
            QMessageBox.warning(self, "Ошибка", f"Модели не найдены в папке: {models_dir}"); 
            return
            
        existing_r_peaks = [ann['sample'] for ann in self.merged_annotations_360 if not ann.get('is_rhythm')]
        self.ann_worker = AnalysisWorker(self.current_signal_data[:, self.current_lead_idx], self.orig_fs, models_dir, existing_r_peaks=existing_r_peaks)
        self.ann_worker.progress_step.connect(self.status_label.setText)
        self.ann_worker.progress_percent.connect(self.progress_bar.setValue)
        self.ann_worker.finished.connect(self.on_analysis_finished)
        self.ann_worker.start()

    def on_analysis_finished(self, results_360):
        self.progress_bar.setVisible(False)
        self.auto_annotations_360 = results_360
        self.project_mgr.save_auto_annotations(self.auto_annotations_360, self.orig_fs)
        self._rebuild_merged_annotations()
        self.status_label.setText("Каскадный анализ завершен")

    def on_annotation_clicked(self, ann_data):
        sample = ann_data.get('start_sample', ann_data.get('sample'))
        if sample is not None: self.pathology_panel.highlight_item(sample, is_rhythm=ann_data.get('is_rhythm', False))

    def export_wfdb(self):
        if self.current_signal_data is None: return
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для экспорта WFDB")
        if not folder: return
        
        singles_to_export = [ann for ann in self.merged_annotations_360 if not ann.get('is_rhythm')]
        
        self.project_mgr.export_merged_signal(
            self.current_signal_data[:, self.current_lead_idx], 
            self.orig_fs, 
            singles_to_export, 
            self.orig_fs, 
            os.path.join(folder, self.project_mgr.record_name + "_exported")
        )
        self.status_label.setText(f"Сигнал WFDB экспортирован в {folder}")
        QMessageBox.information(self, "Экспорт", "Сигнал успешно экспортирован!")