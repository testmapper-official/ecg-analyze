import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal, Qt, QPointF

class BoundedLinearRegionItem(pg.LinearRegionItem):
    """
    Полностью кастомный ползунок.
    Вычисляет позицию самостоятельно, не давая выйти за границы,
    и корректно отправляет сигнал обновления.
    """
    def __init__(self, bounds_func, **kwargs):
        super().__init__(**kwargs)
        self.bounds_func = bounds_func
        
        # Состояние перетаскивания
        self._is_dragging = False
        self._drag_mode = None  # 'body', 'start', 'end'
        self._last_mouse_pos = QPointF(0, 0)

    def mousePressEvent(self, ev):
        """Запоминаем, за что ухватились (начало, конец или середину)"""
        super().mousePressEvent(ev)
        self._is_dragging = True
        self._last_mouse_pos = ev.pos()
        
        # Определяем режим перетаскивания по позиции клика
        mx = ev.pos().x()
        # Координаты линий
        sx = self.lines[0].pos().x()
        ex = self.lines[1].pos().x()
        
        # Порог срабатывания (20 пикселей или единиц координат)
        threshold = 20
        
        if abs(mx - sx) < threshold:
            self._drag_mode = 'start'
        elif abs(mx - ex) < threshold:
            self._drag_mode = 'end'
        else:
            self._drag_mode = 'body'

    def mouseDragEvent(self, ev):
        """
        ВАЖНО: Мы НЕ блокируем сигналы здесь, чтобы NavigationPlot получал обновления.
        Мы считаем дельту сами и применяем с жесткими ограничениями.
        """
        if not self._is_dragging:
            return

        # Вычисляем смещение мыши
        delta = ev.pos() - self._last_mouse_pos
        dx = delta.x()
        
        # Текущие границы
        start, end = self.getRegion()
        width = end - start
        min_width = 10 # Минимальная ширина окна
        
        # Границы сигнала
        min_x, max_x = self.bounds_func()
        
        new_start = start
        new_end = end

        # Логика движения в зависимости от режима
        if self._drag_mode == 'body':
            # Двигаем всё окно
            new_start += dx
            new_end += dx
            
            # Жесткое ограничение ("Стены")
            if new_end > max_x:
                new_end = max_x
                new_start = max_x - width
            elif new_start < min_x:
                new_start = min_x
                new_end = min_x + width
                
        elif self._drag_mode == 'start':
            # Двигаем только левую границу
            new_start += dx
            
            if new_start < min_x:
                new_start = min_x
            
            if (new_end - new_start) < min_width:
                new_start = new_end - min_width
                
        elif self._drag_mode == 'end':
            # Двигаем только правую границу
            new_end += dx
            
            if new_end > max_x:
                new_end = max_x
            
            if (new_end - new_start) < min_width:
                new_end = new_start + min_width

        # Применяем вычисленные координаты
        # УБРАНО blockSignals, чтобы сработал sigRegionChanged
        self.setRegion([new_start, new_end])
        
        # Обновляем последнюю позицию мыши для следующего шага
        self._last_mouse_pos = ev.pos()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        """Завершаем перетаскивание"""
        super().mouseReleaseEvent(ev)
        self._is_dragging = False
        self._drag_mode = None


class NavigationPlot(QWidget):
    range_changed = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMaximumHeight(150)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Разрешаем зум карты
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setMenuEnabled(False)
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=1))
        
        # Используем наш кастомный класс
        self.roi = BoundedLinearRegionItem(bounds_func=self.get_bounds, brush=pg.mkBrush(100, 100, 255, 50))
        
        # Подключаем сигнал изменения ROI к обновлению главного окна
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.plot_widget.addItem(self.roi)
        
        self.layout.addWidget(self.plot_widget)
        
        self.full_signal = None
        self.fs = 360
        self.max_samples = 0
        self._updating = False
        
        self._last_start = 0
        self._last_end = 0

    def get_bounds(self):
        """Возвращает границы сигнала [0, max_samples]"""
        return 0, self.max_samples

    def set_signal(self, signal_data, fs, lead_idx=0):
        self.full_signal = signal_data[:, lead_idx]
        self.fs = fs
        self.max_samples = len(signal_data)
        
        self._auto_scale_y()
        
        step = max(1, len(self.full_signal) // 2000)
        x = np.arange(0, len(self.full_signal), step)
        y = self.full_signal[::step]
        self.curve.setData(x, y)
        
        # Фиксируем карту
        self.plot_widget.setXRange(0, self.max_samples, padding=0)
        self.plot_widget.setLimits(xMin=0, xMax=self.max_samples)

        init_range = int(10 * fs)
        self.roi.setRegion([0, min(init_range, self.max_samples)])
        
        s, e = self.roi.getRegion()
        self._last_start = s
        self._last_end = e

    def _auto_scale_y(self):
        if self.full_signal is None or len(self.full_signal) == 0:
            self.plot_widget.setYRange(-5, 5)
            return

        min_val = np.min(self.full_signal)
        max_val = np.max(self.full_signal)
        margin = max((max_val - min_val) * 0.1, 1.0)
        self.plot_widget.setYRange(min_val - margin, max_val + margin)

    def _on_roi_changed(self):
        """Обработка изменения ROI и отправка сигнала в главное окно"""
        if self._updating:
            return

        self._updating = True
        start, end = self.roi.getRegion()
        
        # Округление
        start, end = int(start), int(end)
        
        self._last_start = start
        self._last_end = end
        
        # Теперь этот сигнал будет вызываться корректно
        self.range_changed.emit(start, end)
        self._updating = False
        
    def update_roi_from_external(self, start_sample, end_sample):
        """Обновление ROI из кода (например, при клике на патологию)"""
        self.roi.blockSignals(True)
        self.roi.setRegion([start_sample, end_sample])
        self.roi.blockSignals(False)
        self._last_start, self._last_end = start_sample, end_sample