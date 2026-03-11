import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import numpy as np
from PyQt5.QtCore import pyqtSignal

class NavigationPlot(QWidget):
    range_changed = pyqtSignal(int, int) # start_sample, end_sample

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMaximumHeight(150)
        
        # Данные для отрисовки (уменьшенная копия для скорости)
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=1))
        
        # Region Selector (ROI)
        self.roi = pg.LinearRegionItem(brush=pg.mkBrush(100, 100, 255, 50))
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.plot_widget.addItem(self.roi)
        
        self.layout.addWidget(self.plot_widget)
        
        self.full_signal = None
        self.fs = 360
        self.max_samples = 0

    def set_signal(self, signal_data, fs, lead_idx=0):
        self.full_signal = signal_data[:, lead_idx]
        self.fs = fs
        self.max_samples = len(signal_data)
        
        # Даунсэмплинг для быстрой отрисовки полного сигнала
        step = max(1, len(self.full_signal) // 2000)
        x = np.arange(0, len(self.full_signal), step)
        y = self.full_signal[::step]
        self.curve.setData(x, y)
        
        # Установка начального ROI (первые 10 секунд)
        init_range = int(10 * fs)
        self.roi.setRegion([0, min(init_range, self.max_samples)])

    def _on_roi_changed(self):
        start, end = self.roi.getRegion()
        # Округляем до сэмплов
        start_sample = int(start)
        end_sample = int(end)
        self.range_changed.emit(start_sample, end_sample)
        
    def update_roi_from_external(self, start_sample, end_sample):
        """Обновление ROI извне (если нужно)"""
        self.roi.setRegion([start_sample, end_sample])