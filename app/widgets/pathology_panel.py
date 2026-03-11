from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QLabel, 
                             QListWidgetItem, QPushButton, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal

class PathologyPanel(QWidget):
    jump_to_position = pyqtSignal(int) # sample index

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Заголовок
        self.title = QLabel("Обнаруженные патологии")
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Список
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Запустить анализ")
        self.btn_export = QPushButton("Экспорт отчета")
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_export)
        
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.list_widget)
        self.layout.addLayout(btn_layout)
        
        self.pathologies = [] # Список кортежей (sample_idx, name)

    def set_analysis_results(self, results):
        """results: list of dict {'sample': int, 'label': str, 'probability': float}"""
        self.list_widget.clear()
        self.pathologies = results
        
        for res in results:
            time_sec = res['sample'] / res.get('fs', 360)
            item_text = f"[{time_sec:.2f}s] {res['label']} ({res['probability']:.2f}%)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, res['sample']) # Сохраняем позицию
            self.list_widget.addItem(item)

    def _on_item_double_clicked(self, item):
        sample_idx = item.data(Qt.UserRole)
        self.jump_to_position.emit(sample_idx)