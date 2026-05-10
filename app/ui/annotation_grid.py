from PyQt5.QtWidgets import QWidget, QMenu, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
import numpy as np

DANGER_MAP = {'r': 3, 'Couplet': 3, 'B': 2, 'L': 2, 'R': 2, 'V': 1, 'A': 1, 'F': 1, 'E': 1}

# ИСПРАВЛЕНО: Человекочитаемые метки (Название, Символ)
LABEL_CHOICES = [
    ('Нормальный комплекс (N)', 'N'),
    ('Преждевременная желудочковая систолия (V)', 'V'),
    ('Преждевременная наджелудочковая систолия (A)', 'A'),
    ('Блокада (B)', 'B'),
    ('Блокада левая (L)', 'L'),
    ('Блокада правая (R)', 'R'),
    ('R-on-T (r)', 'r'),
    ('Эктопический (E)', 'E'),
    ('Сливной (F)', 'F')
]

class AnnotationGrid(QWidget):
    manual_label_changed = pyqtSignal(int, str) # sample_360, new_label

    def __init__(self):
        super().__init__()
        self.setFixedHeight(60)
        self.orig_fs = 360
        self.current_annotations = []
        self.current_start = 0
        self.current_end = 0
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.placeholder_label = QLabel("Масштаб слишком мелкий для редактирования сетки")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-style: italic;")
        self.layout.addWidget(self.placeholder_label)
        self.placeholder_label.hide()

    def update_grid(self, start_sample_orig, end_sample_orig, annotations_360, orig_fs, plot_width_px, view_auto_mode=False):
        self.orig_fs = orig_fs
        self.current_start = start_sample_orig
        self.current_end = end_sample_orig
        self.plot_width_px = plot_width_px
        self.view_auto_mode = view_auto_mode # Сохраняем флаг
        self.setFixedWidth(plot_width_px)

        visible_annotations = []
        ratio_orig_to_360 = 360.0 / self.orig_fs
        for ann in annotations_360:
            if ann.get('is_rhythm'): continue
            s_orig = int(ann['sample'] / ratio_orig_to_360)
            if start_sample_orig <= s_orig < end_sample_orig:
                visible_annotations.append(ann)

        if len(visible_annotations) > 10:
            self.placeholder_label.show(); self.current_annotations = []
        else:
            self.placeholder_label.hide(); self.current_annotations = visible_annotations
        self.update()

    def paintEvent(self, event):
        if self.placeholder_label.isVisible() or not self.current_annotations: return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            view_duration = self.current_end - self.current_start
            font_manual = QFont("Arial", 11, QFont.Bold)
            font_auto = QFont("Arial", 9)
            pen = QPen(QColor(200, 200, 200), 1)

            for ann in self.current_annotations:
                ratio_orig_to_360 = 360.0 / self.orig_fs
                s_orig = int(ann['sample'] / ratio_orig_to_360)
                x_center = ((s_orig - self.current_start) / view_duration) * self.width()
                qrs_samples_120ms = int(0.120 * self.orig_fs)
                qrs_width_px = (qrs_samples_120ms / view_duration) * self.width()
                x_start = x_center - qrs_width_px / 2
                
                auto_sym = ann.get('auto_symbol', '')
                manual_sym = ann.get('manual_symbol', '')

                # Логика активного символа для цвета
                if self.view_auto_mode:
                    active_sym = auto_sym if auto_sym else manual_sym
                    text_manual = manual_sym if manual_sym else "—"
                    text_auto = auto_sym
                else:
                    active_sym = manual_sym if manual_sym else auto_sym
                    text_manual = manual_sym if manual_sym else "—"
                    text_auto = auto_sym

                danger_level = DANGER_MAP.get(active_sym, 0)
                bg_color = self._get_danger_bg(danger_level)
                
                rect = QRectF(x_start, 0, qrs_width_px, self.height())
                painter.fillRect(rect, bg_color)
                painter.setPen(pen)
                painter.drawRect(rect)
                painter.drawLine(int(x_start), self.height()//2, int(x_start + qrs_width_px), self.height()//2)
                
                # ВЕРХНЯЯ СТРОКА (Врач / Главная метка)
                painter.setPen(QColor("blue"))
                painter.setFont(font_manual)
                painter.drawText(rect.adjusted(0, 2, 0, -self.height()//2), Qt.AlignCenter, text_manual)
                
                # НИЖНЯЯ СТРОКА (Классификатор) - ИСПРАВЛЕНО: Рисуем всегда, если есть авто-метка!
                if text_auto:
                    painter.setPen(QColor("gray"))
                    painter.setFont(font_auto)
                    painter.drawText(rect.adjusted(0, self.height()//2, 0, -2), Qt.AlignCenter, text_auto)
        finally:
            painter.end()

    def mousePressEvent(self, event):
        if self.placeholder_label.isVisible() or not self.current_annotations: return
        view_duration = self.current_end - self.current_start
        click_x = event.pos().x()
        ratio_orig_to_360 = 360.0 / self.orig_fs
        qrs_samples_120ms = int(0.120 * self.orig_fs)
        qrs_width_px = (qrs_samples_120ms / view_duration) * self.width()

        for ann in self.current_annotations:
            s_orig = int(ann['sample'] / ratio_orig_to_360)
            x_center = ((s_orig - self.current_start) / view_duration) * self.width()
            x_start = x_center - qrs_width_px / 2
            if x_start <= click_x <= x_start + qrs_width_px:
                menu_pos = self.mapToGlobal(QPoint(int(x_center), self.height()))
                self._show_label_menu(ann['sample'], menu_pos)
                break

    def _show_label_menu(self, sample_360, global_pos):
        menu = QMenu(self)
        for text, sym in LABEL_CHOICES:
            action = menu.addAction(text)
            # Передаем короткий символ (sym), а показываем длинный текст (text)
            action.triggered.connect(lambda checked, s=sample_360, sy=sym: self.manual_label_changed.emit(s, sy))
        menu.exec_(global_pos)

    def _get_danger_bg(self, level):
        if level == 3: return QColor(255, 0, 0, 40)
        elif level == 2: return QColor(255, 165, 0, 30)
        elif level == 1: return QColor(0, 255, 0, 20)
        return QColor(255, 255, 255, 0)