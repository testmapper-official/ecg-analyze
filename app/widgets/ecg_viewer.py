import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import numpy as np

class ECGViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude', units='mV')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('k', width=1.5))
        
        # Аннотации
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('r'))
        self.plot_widget.addItem(self.scatter)
        
        self.layout.addWidget(self.plot_widget)
        
        self.fs = 360
        self.current_lead_idx = 0
        self.current_start_sample = 0 # Для отслеживания позиции

    def update_view(self, start_sample, end_sample, signal_data, fs, lead_idx):
        self.fs = fs
        self.current_lead_idx = lead_idx
        self.current_start_sample = start_sample
        
        if signal_data is None: return
            
        segment = signal_data[start_sample:end_sample, lead_idx]
        
        if len(segment) == 0:
            self.curve.clear()
            self.scatter.clear()
            return

        # Вычисляем абсолютное время для оси X
        # Время начала сегмента в секундах
        start_time_sec = start_sample / self.fs
        
        # Массив времени для графика: [0, 1, 2...] / fs + start_time_sec
        # Это дает нам массив [140.0, 140.0027, 140.0055...]
        time_array = (np.arange(len(segment)) / self.fs) + start_time_sec
        
        self.curve.setData(time_array, segment)
        
        # Автоматическое масштабирование Y под текущий сегмент
        min_val = np.min(segment)
        max_val = np.max(segment)
        margin = max((max_val - min_val) * 0.1, 0.5) 
        self.plot_widget.setYRange(min_val - margin, max_val + margin)
        
        # Устанавливаем видимый диапазон оси X точно по границам сегмента
        self.plot_widget.setXRange(time_array[0], time_array[-1], padding=0.0)
        
    def set_annotations(self, annotations, start_sample, end_sample):
        points = []
        for item in annotations:
            if isinstance(item, tuple):
                sample, symbol = item
            elif isinstance(item, dict):
                sample = item.get('sample')
                symbol = item.get('symbol', '?')
            else:
                continue
                
            # Проверяем, попадает ли аннотация в текущий диапазон
            if start_sample <= sample < end_sample:
                # ВАЖНО: Теперь x_val - это абсолютное время аннотации
                x_val = sample / self.fs
                
                # Для корректного отображения Y, нам нужно знать амплитуду в этой точке.
                # Поскольку у нас нет прямого доступа к данным отрисовки внутри метода (если мы не храним их),
                # мы можем попытаться найти ближайшую точку или просто рисовать на оси Y=0
                # Для профессионального вида лучше найти значение амплитуды:
                
                # Так как мы не передали сам массив segment сюда, сделаем упрощение:
                # Просто рисуем маркер на оси X. 
                # Чтобы он был виден, ставим его посередине видимого диапазона Y (приблизительно 0, если сигнал отнормирован, или просто 0)
                
                points.append({'pos': (x_val, 0), 'brush': 'r', 'symbol': 't1', 'data': symbol})
        
        self.scatter.setData(points)