import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import numpy as np

class ECGViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Используем PlotWidget для быстрой отрисовки
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude', units='mV')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('k', width=1.5))
        
        # Аннотации (маркеры)
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('r'))
        self.plot_widget.addItem(self.scatter)
        
        self.layout.addWidget(self.plot_widget)
        
        self.fs = 360
        self.current_lead_idx = 0
        self.full_signal = None

    def set_signal(self, signal_data, fs, lead_idx=0):
        self.full_signal = signal_data
        self.fs = fs
        self.current_lead_idx = lead_idx

    def update_view(self, start_sample, end_sample):
        if self.full_signal is None:
            return
            
        segment = self.full_signal[start_sample:end_sample, self.current_lead_idx]
        time = np.arange(len(segment)) / self.fs
        
        self.curve.setData(time, segment)
        
        # Обновление заголовка оси X
        start_time = start_sample / self.fs
        self.plot_widget.setXRange(0, len(segment)/self.fs, padding=0.0)
        
    def set_annotations(self, annotations, start_sample, end_sample):
        # Фильтрация аннотаций в пределах видимого диапазона
        points = []
        for sample, symbol in annotations:
            if start_sample <= sample < end_sample:
                # Нужно найти амплитуду в этой точке
                local_idx = sample - start_sample
                if local_idx < len(self.full_signal):
                    y_val = self.full_signal[sample, self.current_lead_idx]
                    x_val = (sample - start_sample) / self.fs
                    points.append({'pos': (x_val, y_val), 'brush': 'r', 'symbol': 'o'})
        
        self.scatter.setData(points)