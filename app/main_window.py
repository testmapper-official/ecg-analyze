import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QSplitter, QMessageBox, QProgressBar, QLabel)
from PyQt5.QtCore import Qt, QThread
from app.widgets.custom_title_bar import CustomTitleBar
from app.widgets.ecg_viewer import ECGViewer
from app.widgets.navigation_plot import NavigationPlot
from app.widgets.pathology_panel import PathologyPanel
from app.widgets.toolbar import Toolbar
from app.core import DataLoader, ECGProcessor, ECGClassifier, AnalysisWorker
from app.utils.style_loader import StyleLoader
from app import DB_DIR

class MainWindow(QMainWindow):
    def __init__(self, style_loader):
        super().__init__()
        self.style_loader = style_loader
        self.setWindowTitle("Holter Monitor")
        self.resize(1200, 800)

        self.processor = ECGProcessor()
        self.classifier = ECGClassifier()
        self.worker = None # Для хранения потока анализа
        
        self.is_filter_active = False
        
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # --- Title Bar ---
        self.title_bar = CustomTitleBar(self)
        self.main_layout.addWidget(self.title_bar)
        
        # --- Central Content ---
        self.content_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel (Pathology)
        self.pathology_panel = PathologyPanel()
        self.pathology_panel.jump_to_position.connect(self.jump_to_signal_position)
        self.pathology_panel.analyze_requested.connect(self.start_analysis_async)
        self.content_splitter.addWidget(self.pathology_panel)
        
        # Right Panel (Plots)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        self.toolbar = Toolbar(self.style_loader)
        self.toolbar.open_file_clicked.connect(self.load_file_action)
        self.toolbar.lead_changed.connect(self.on_lead_changed)
        self.toolbar.style_changed.connect(self.on_style_changed)
        self.toolbar.filter_toggled.connect(self.apply_filter)
        self.plot_layout.addWidget(self.toolbar)
        
        # Plots
        self.ecg_viewer = ECGViewer()
        self.nav_plot = NavigationPlot()
        self.nav_plot.range_changed.connect(self.update_main_view)
        
        self.plot_layout.addWidget(self.ecg_viewer, stretch=2)
        self.plot_layout.addWidget(self.nav_plot, stretch=1)
        
        # Progress Bar (создаем, но скрываем изначально)
        self.progress_container = QWidget()
        self.progress_layout = QHBoxLayout(self.progress_container)
        self.progress_label = QLabel("Анализ сигнала:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_bar)
        self.plot_layout.addWidget(self.progress_container)
        
        self.content_splitter.addWidget(self.plot_container)
        
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 4)
        
        self.main_layout.addWidget(self.content_splitter)
        
        # Data
        self.raw_signal = None
        self.filtered_signal = None
        self.current_signal = None 
        self.ecg_data = None
        self.current_lead_idx = 0
        self.fs = 360

    def on_style_changed(self, style_name):
        self.style_loader.apply_style(self, style_name)

    def apply_filter(self, checked):
        self.is_filter_active = checked
        if self.raw_signal is not None:
            if checked:
                if self.filtered_signal is None:
                    self.filtered_signal = np.zeros_like(self.raw_signal)
                    for i in range(self.raw_signal.shape[1]):
                        self.filtered_signal[:, i] = self.processor.filter_signal(self.raw_signal[:, i], fs=self.fs)
                self.current_signal = self.filtered_signal
            else:
                self.current_signal = self.raw_signal
            self.refresh_plots()

    def refresh_plots(self):
        if self.current_signal is None: return
        self.nav_plot.set_signal(self.current_signal, self.fs, lead_idx=self.current_lead_idx)
        start, end = self.nav_plot.roi.getRegion()
        self.update_main_view(int(start), int(end))

    def load_file_action(self):
        start_dir = DB_DIR if DB_DIR and os.path.exists(DB_DIR) else ""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл ЭКГ", start_dir, 
                                                  "All Supported (*.dat *.hea *.edf *.csv);;MIT-BIH (*.dat);;EDF (*.edf);;CSV (*.csv)", options=options)
        if file_path:
            try:
                self.ecg_data = DataLoader.load_file(file_path)
                self.raw_signal = self.ecg_data['signal']
                self.current_signal = self.raw_signal
                self.filtered_signal = None 
                self.fs = self.ecg_data['fs']
                self.setup_ui_after_load()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл:\n{e}")

    def setup_ui_after_load(self):
        self.toolbar.set_leads(self.ecg_data['leads'])
        self.toolbar.btn_filter.setChecked(False)
        self.is_filter_active = False
        self.refresh_plots()
        
        if self.ecg_data['annotations']:
            # Фильтруем аннотации, если нужно, или показываем все
            self.pathology_panel.set_analysis_results([
                {'sample': s, 'label': sym, 'probability': 100.0, 'fs': self.ecg_data['fs']}
                for s, sym in self.ecg_data['annotations'][:20] 
            ])
        else:
            self.pathology_panel.list_widget.clear()

    def on_lead_changed(self, index):
        self.current_lead_idx = index
        self.refresh_plots()

    def update_main_view(self, start_sample, end_sample):
        if self.current_signal is None: return
        start_sample = max(0, start_sample)
        end_sample = min(len(self.current_signal), end_sample)
        self.ecg_viewer.update_view(start_sample, end_sample, self.current_signal, self.fs, self.current_lead_idx)
        if self.ecg_data and self.ecg_data['annotations']:
             self.ecg_viewer.set_annotations(self.ecg_data['annotations'], start_sample, end_sample)

    def jump_to_signal_position(self, sample_idx):
        if self.current_signal is None: return
        window_width = int(5 * self.fs)
        start = max(0, sample_idx - window_width // 2)
        end = start + window_width
        self.nav_plot.update_roi_from_external(start, end)
        self.update_main_view(start, end)

    def start_analysis_async(self):
        """Запуск анализа в отдельном потоке"""
        if self.current_signal is None:
            QMessageBox.warning(self, "Внимание", "Сначала откройте файл ЭКГ.")
            return

        # Блокируем UI
        self.toolbar.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)

        # Создаем и запускаем воркера
        signal_lead = self.current_signal[:, self.current_lead_idx]
        self.worker = AnalysisWorker(signal_lead, self.fs, self.processor, self.classifier)
        
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        
        self.worker.start()

    def on_analysis_finished(self, results):
        """Обработка завершения анализа"""
        self.toolbar.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        if not results:
            QMessageBox.information(self, "Результат", "Анализ завершен. Патологий не обнаружено.")
        else:
            QMessageBox.information(self, "Результат", f"Анализ завершен. Обнаружено патологий: {len(results)}")
            
        self.pathology_panel.set_analysis_results(results)
        self.worker = None

    def on_analysis_error(self, error_msg):
        """Обработка ошибок потока"""
        self.toolbar.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QMessageBox.critical(self, "Ошибка анализа", f"Произошла ошибка:\n{error_msg}")
        self.worker = None