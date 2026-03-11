# app/widgets/toolbar.py
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QComboBox, 
                             QLabel, QSlider, QFileDialog, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt

class Toolbar(QWidget):
    # Сигналы для взаимодействия с главным окном
    open_file_clicked = pyqtSignal()
    lead_changed = pyqtSignal(int)
    style_changed = pyqtSignal(str)
    filter_toggled = pyqtSignal(bool)
    
    def __init__(self, style_loader):
        super().__init__()
        self.style_loader = style_loader
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Кнопка открытия файла
        self.btn_open = QPushButton("Открыть файл")
        self.btn_open.setFixedWidth(100)
        self.btn_open.clicked.connect(self.open_file_clicked)
        
        # Выбор отведения
        layout.addWidget(self.btn_open)
        layout.addWidget(QLabel("Отведение:"))
        self.combo_lead = QComboBox()
        self.combo_lead.setMinimumWidth(100)
        self.combo_lead.currentIndexChanged.connect(self.lead_changed)
        layout.addWidget(self.combo_lead)
        
        layout.addSpacing(20)
        
        # Переключатель фильтрации
        self.btn_filter = QPushButton("Фильтрация: ВЫКЛ")
        self.btn_filter.setCheckable(True)
        self.btn_filter.setFixedWidth(130)
        self.btn_filter.clicked.connect(self.toggle_filter)
        layout.addWidget(self.btn_filter)
        
        layout.addStretch()
        
        # Выбор стиля
        layout.addWidget(QLabel("Тема:"))
        self.combo_style = QComboBox()
        self.combo_style.addItems(self.style_loader.get_available_styles())
        self.combo_style.currentTextChanged.connect(self.style_changed)
        layout.addWidget(self.combo_style)

    def set_leads(self, leads):
        """Заполнение списка отведений"""
        self.combo_lead.clear()
        self.combo_lead.addItems(leads)

    def toggle_filter(self, checked):
        if checked:
            self.btn_filter.setText("Фильтрация: ВКЛ")
            self.btn_filter.setStyleSheet("background-color: #006666; color: white;")
        else:
            self.btn_filter.setText("Фильтрация: ВЫКЛ")
            self.btn_filter.setStyleSheet("") # Сброс стиля
        self.filter_toggled.emit(checked)