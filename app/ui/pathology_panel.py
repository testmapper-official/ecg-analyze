from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
                             QPushButton, QComboBox, QHBoxLayout, QTreeWidgetItemIterator)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap, QPainter, QPen

DANGER_MAP = {
    'r': 3, 'VT': 3, 'VF': 3, 'Couplet': 3,
    'B': 2, 'L': 2, 'R': 2, 'SVT': 2, 'SB': 2, 'Bigeminy': 2, 'Trigeminy': 2,
    'V': 1, 'M': 1, 'P': 1, 'i': 1, 'A': 1, 'F': 1, 'E': 1, 'e': 1
}

SYMBOL_NAMES = {
    'N': 'Нормальный комплекс (N)',
    'V': 'Преждевременная желудочковая (V)',
    'M': 'Мономорфная ПЖС (M)',
    'P': 'Полиморфная ПЖС (P)',
    'i': 'Интерполированная ПЖС (i)',
    'A': 'Преждевременная наджелудочковая (A)',
    'B': 'Блокада (B)', 
    'L': 'Блокада левая (L)',
    'R': 'Блокада правая (R)',
    'r': 'R-on-T (r)',
    'E': 'Эктопический (E)',
    'F': 'Сливной (F)',
    'e': 'Эктопический (e)',
    'VT': 'Желудочковая тахикардия (VT)',
    'SVT': 'Наджелудочковая тахикардия (SVT)', 
    'SB': 'Стабильная блокада (SB)',
    'Bigeminy': 'Бигеминия (Bigeminy)', 
    'Trigeminy': 'Тригеминия (Trigeminy)',
    'Couplet': 'Парная экстрасистола (Couplet)',
    'VF': 'Фибрилляция (VF)'
}


class PathologyPanel(QWidget):
    jump_to_position = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Панель фильтров
        filter_layout = QHBoxLayout()
        
        self.type_filter = QComboBox()
        self.type_filter.addItems(["Все", "1 - Безопасный", "2 - Риск", "3 - Опасный"])
        
        # Кнопка применения фильтра с иконкой лупы
        self.btn_apply_filter = QPushButton()
        self.btn_apply_filter.setIcon(self._create_search_icon())
        self.btn_apply_filter.setToolTip("Применить фильтр")
        self.btn_apply_filter.setFixedSize(30, 30)
        self.btn_apply_filter.clicked.connect(self._update_tree)
        
        filter_layout.addWidget(self.type_filter, stretch=1)
        filter_layout.addWidget(self.btn_apply_filter)
        self.layout.addLayout(filter_layout)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Классификация", "Начало", "Конец"])
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.layout.addWidget(self.tree)
        self.all_results = []

    def _create_search_icon(self):
        """Программное создание иконки лупы"""
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("black"), 2)
        painter.setPen(pen)
        painter.drawEllipse(3, 3, 12, 12) # Линза
        painter.drawLine(13, 13, 20, 20) # Ручка
        painter.end()
        return QIcon(pixmap)

    def set_analysis_results(self, results, view_auto_mode=False):
        self.all_results = results
        self.view_auto_mode = view_auto_mode
        self._update_tree()

    def _update_tree(self):
        self.tree.clear()
        
        # Определяем выбранный класс опасности (0 = Все)
        selected_text = self.type_filter.currentText()
        allowed_danger = 0
        if "1" in selected_text: allowed_danger = 1
        elif "2" in selected_text: allowed_danger = 2
        elif "3" in selected_text: allowed_danger = 3

        rhythms = [r for r in self.all_results if r.get('is_rhythm')]
        singles = [r for r in self.all_results if not r.get('is_rhythm')]

        # --- Ритмы ---
        rhythm_parent = QTreeWidgetItem(self.tree, ["Ритмы", "", ""])
        rhythm_parent.setExpanded(True)
        for res in rhythms:
            sym = res.get('symbol', 'Rhythm')
            display_name = SYMBOL_NAMES.get(sym, sym)
            danger = DANGER_MAP.get(sym, 0)
            
            # Фильтрация
            if allowed_danger > 0 and danger != allowed_danger:
                continue
                
            color = self._get_color_for_danger(danger); brush = QBrush(color)
            start_sec = res['start_sample'] / 360.0; end_sec = res['end_sample'] / 360.0
            child = QTreeWidgetItem(rhythm_parent, [display_name, f"{start_sec:.2f} с", f"{end_sec:.2f} с"])
            child.setForeground(0, brush); child.setData(0, Qt.UserRole, res['start_sample'])

        # Удаляем категорию, если она пустая после фильтрации
        if rhythm_parent.childCount() == 0:
            index = self.tree.indexOfTopLevelItem(rhythm_parent)
            self.tree.takeTopLevelItem(index)

        # --- Одиночные ---
        single_parent = QTreeWidgetItem(self.tree, ["Одиночные QRS-комплексы", "", ""])
        single_parent.setExpanded(True)
        groups = {}
        for res in singles:
            if self.view_auto_mode:
                sym = res.get('auto_symbol') if res.get('auto_symbol') else res.get('manual_symbol', 'Unknown')
            else:
                sym = res.get('manual_symbol') if res.get('manual_symbol') else res.get('auto_symbol', 'Unknown')
                
            danger = DANGER_MAP.get(sym, 0)
            
            # Фильтрация
            if allowed_danger > 0 and danger != allowed_danger:
                continue
                
            if DANGER_MAP.get(sym, 0) == 0: continue
            if sym not in groups: groups[sym] = []
            groups[sym].append(res)
            
        for sym, items in groups.items():
            display_name = SYMBOL_NAMES.get(sym, sym)
            danger = DANGER_MAP.get(sym, 0); color = self._get_color_for_danger(danger); brush = QBrush(color)
            type_item = QTreeWidgetItem(single_parent, [display_name, f"({len(items)})", ""])
            type_item.setForeground(0, brush)
            for res in items:
                start_sec = res['sample'] / 360.0
                child = QTreeWidgetItem(type_item, [display_name, f"{start_sec:.2f} с", ""])
                child.setForeground(0, brush); child.setData(0, Qt.UserRole, res['sample'])

        # Удаляем категорию, если она пустая после фильтрации
        if single_parent.childCount() == 0:
            index = self.tree.indexOfTopLevelItem(single_parent)
            self.tree.takeTopLevelItem(index)

    def _get_color_for_danger(self, level):
        if level == 3: return QColor(255, 0, 0)
        elif level == 2: return QColor(255, 165, 0)
        elif level == 1: return QColor(0, 150, 0)
        return QColor(0, 0, 0)

    def _on_item_double_clicked(self, item, column):
        sample = item.data(0, Qt.UserRole)
        if sample is not None: self.jump_to_position.emit(sample)

    def highlight_item(self, sample_360, is_rhythm=False):
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, Qt.UserRole) == sample_360:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                return
            iterator += 1