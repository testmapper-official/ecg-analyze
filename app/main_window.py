from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QSplitter, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
from app.widgets.custom_title_bar import CustomTitleBar
from app.widgets.ecg_viewer import ECGViewer
from app.widgets.navigation_plot import NavigationPlot
from app.widgets.pathology_panel import PathologyPanel
from app.core.data_loader import DataLoader
from app.utils.style_loader import StyleLoader
from app.core import DataLoader, ECGProcessor, ECGClassifier
from app.widgets import Toolbar

class MainWindow(QMainWindow):
    def __init__(self, style_loader):
        super().__init__()
        self.style_loader = style_loader
        self.setWindowTitle("Holter Monitor")
        self.resize(1200, 800)

        self.processor = ECGProcessor()
        self.classifier = ECGClassifier() # По умолчанию загрузит model1.h5
        
        # Убираем стандартный заголовок
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Главный контейнер
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Кастомный Title Bar
        self.title_bar = CustomTitleBar(self)
        self.main_layout.addWidget(self.title_bar)
        
        # Центральная часть (Splitter)
        self.splitter = QSplitter(Qt.Horizontal)
        
        # --- Левая панель (Патологии) ---
        self.pathology_panel = PathologyPanel()
        self.pathology_panel.jump_to_position.connect(self.jump_to_signal_position)
        self.splitter.addWidget(self.pathology_panel)
        
        # --- Правая часть (Графики) ---
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        
        # Панель инструментов для выбора отведения
        self.toolbar = Toolbar(self.style_loader)
        self.toolbar.open_file_clicked.connect(self.load_file_action)
        self.toolbar.lead_changed.connect(self.on_lead_changed)
        self.toolbar.style_changed.connect(self.on_style_changed)
        self.toolbar.filter_toggled.connect(self.apply_filter)
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        self.toolbar_layout.addWidget(self.create_lead_selector())
        self.toolbar_layout.addStretch()
        
        # Графики
        self.ecg_viewer = ECGViewer()
        self.nav_plot = NavigationPlot()
        self.nav_plot.range_changed.connect(self.update_main_view)
        
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.ecg_viewer, stretch=2)
        self.plot_layout.addWidget(self.nav_plot, stretch=1)
        
        self.splitter.addWidget(self.plot_container)
        
        # Размеры панелей
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)
        
        self.main_layout.addWidget(self.splitter)
        
        # Данные
        self.ecg_data = None
        self.current_lead = 0
        
        # Подключение меню (через кнопки или меню, здесь упрощенно через title_bar или контекст)
        # Для примера добавим возможность открытия через D&D или сразу загрузим тест
        self.load_file_action() # Можно убрать для автозапуска диалога

    def create_lead_selector(self):
        self.lead_combo = QComboBox()
        self.lead_combo.currentIndexChanged.connect(self.on_lead_changed)
        return self.lead_combo

    def load_file_action(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл ЭКГ", "", 
                                                  "All Supported (*.dat *.hea *.edf *.csv);;MIT-BIH (*.dat);;EDF (*.edf);;CSV (*.csv)", options=options)
        if file_path:
            try:
                self.ecg_data = DataLoader.load_file(file_path)
                self.setup_ui_after_load()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл:\n{e}")

    def setup_ui_after_load(self):
        # Настройка списка отведений
        self.lead_combo.clear()
        self.lead_combo.addItems(self.ecg_data['leads'])
        
        # Передача данных в виджеты
        self.ecg_viewer.set_signal(self.ecg_data['signal'], self.ecg_data['fs'])
        self.nav_plot.set_signal(self.ecg_data['signal'], self.ecg_data['fs'])
        
        # Передача аннотаций, если есть
        if self.ecg_data['annotations']:
            self.pathology_panel.set_analysis_results([
                {'sample': s, 'label': sym, 'probability': 100.0, 'fs': self.ecg_data['fs']}
                for s, sym in self.ecg_data['annotations'][:20] # первые 20 для теста
            ])

    def on_lead_changed(self, index):
        self.current_lead = index
        self.ecg_viewer.current_lead_idx = index
        self.nav_plot.set_signal(self.ecg_data['signal'], self.ecg_data['fs'], lead_idx=index)
        
        # Принудительное обновление текущего вида
        start, end = self.nav_plot.roi.getRegion()
        self.update_main_view(int(start), int(end))

    def update_main_view(self, start_sample, end_sample):
        self.ecg_viewer.update_view(start_sample, end_sample)
        # Обновление аннотаций на текущем кадре
        if self.ecg_data and self.ecg_data['annotations']:
             self.ecg_viewer.set_annotations(self.ecg_data['annotations'], start_sample, end_sample)

    def jump_to_signal_position(self, sample_idx):
        # Центрирование вида на указанной позиции
        window_width = int(5 * self.ecg_data['fs']) # 5 секунд
        start = max(0, sample_idx - window_width // 2)
        end = start + window_width
        
        self.nav_plot.update_roi_from_external(start, end)
        self.update_main_view(start, end)