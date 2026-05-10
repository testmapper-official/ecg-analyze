import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal, Qt, QPointF

class BoundedLinearRegionItem(pg.LinearRegionItem):
    def __init__(self, bounds_func, **kwargs):
        super().__init__(**kwargs)
        self.bounds_func = bounds_func
        self._is_dragging = False
        self._drag_mode = None
        self._last_mouse_pos = QPointF(0, 0)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        self._is_dragging = True
        self._last_mouse_pos = ev.pos()
        mx = ev.pos().x()
        sx = self.lines[0].pos().x()
        ex = self.lines[1].pos().x()
        threshold = 20
        if abs(mx - sx) < threshold: self._drag_mode = 'start'
        elif abs(mx - ex) < threshold: self._drag_mode = 'end'
        else: self._drag_mode = 'body'

    def mouseDragEvent(self, ev):
        if not self._is_dragging: return
        delta = ev.pos() - self._last_mouse_pos
        dx = delta.x()
        start, end = self.getRegion()
        width = end - start
        min_width = 10
        min_x, max_x = self.bounds_func()
        new_start, new_end = start, end

        if self._drag_mode == 'body':
            new_start += dx; new_end += dx
            if new_end > max_x: new_end = max_x; new_start = max_x - width
            elif new_start < min_x: new_start = min_x; new_end = min_x + width
        elif self._drag_mode == 'start':
            new_start += dx
            if new_start < min_x: new_start = min_x
            if (new_end - new_start) < min_width: new_start = new_end - min_width
        elif self._drag_mode == 'end':
            new_end += dx
            if new_end > max_x: new_end = max_x
            if (new_end - new_start) < min_width: new_end = new_start + min_width

        self.setRegion([new_start, new_end])
        self._last_mouse_pos = ev.pos()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self._is_dragging = False; self._drag_mode = None

def min_max_downsample(data, target_len=4000):
    if len(data) <= target_len: return data
    bin_size = len(data) // target_len
    data_reshaped = data[:bin_size * target_len].reshape(target_len, bin_size)
    mins = np.min(data_reshaped, axis=1)
    maxs = np.max(data_reshaped, axis=1)
    downsampled = np.empty(target_len * 2, dtype=data.dtype)
    downsampled[0::2] = mins
    downsampled[1::2] = maxs
    return downsampled


class NavigationPlot(QWidget):
    range_changed = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMaximumHeight(150)
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setMenuEnabled(False)
        
        self.x_axis = self.plot_widget.getAxis('bottom')
        self.x_axis.setStyle(tickLength=-5)
        self.plot_widget.getAxis('left').setVisible(False)
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=1))
        self.roi = BoundedLinearRegionItem(bounds_func=self.get_bounds, brush=pg.mkBrush(100, 100, 255, 50))
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.plot_widget.addItem(self.roi)
        
        self.layout.addWidget(self.plot_widget)
        self.full_signal = None
        self.fs = 360
        self.max_samples = 0
        self._updating = False
        
        # Флаг визуальной стандартизации
        self.visual_standardize = False

        self.plot_widget.getViewBox().sigXRangeChanged.connect(self._update_time_axis)

    def get_bounds(self):
        return 0, self.max_samples

    def set_visual_standardize(self, enabled):
        self.visual_standardize = enabled
        self._update_plot_data()

    def set_signal(self, signal_data, fs, lead_idx=0):
        is_first_load = self.full_signal is None
        saved_region = self.roi.getRegion() if not is_first_load else None
        
        self.full_signal = signal_data[:, lead_idx]
        self.fs = fs
        self.max_samples = len(signal_data)
        
        self._update_plot_data()
        
        if is_first_load:
            self.plot_widget.setLimits(xMin=0, xMax=self.max_samples)
            self.plot_widget.setXRange(0, self.max_samples, padding=0)
            init_range = int(20 * fs)
            self.roi.setRegion([0, min(init_range, self.max_samples)])
        else:
            self.roi.setRegion(saved_region)

    def _update_plot_data(self):
        """Перерисовка кривой навигации с учетом флага стандартизации"""
        if self.full_signal is None: return
        
        sig_to_plot = self.full_signal
        if self.visual_standardize:
            sig_to_plot = sig_to_plot - np.mean(sig_to_plot)
            
        downsampled_sig = min_max_downsample(sig_to_plot)
        x = np.linspace(0, self.max_samples, len(downsampled_sig))
        self.curve.setData(x, downsampled_sig)
        self._auto_scale_y(sig_to_plot)

    def _auto_scale_y(self, data=None):
        if data is None: data = self.full_signal
        if data is None or len(data) == 0:
            self.plot_widget.setYRange(-5, 5); return
            
        min_val = np.min(data)
        max_val = np.max(data)
        
        # ИСПРАВЛЕНО: Симметричный масштаб при центрировании
        if self.visual_standardize:
            max_abs = max(abs(min_val), abs(max_val), 1.0)
            margin = max_abs * 0.1
            self.plot_widget.setYRange(-max_abs - margin, max_abs + margin)
        else:
            margin = max((max_val - min_val) * 0.1, 1.0)
            self.plot_widget.setYRange(min_val - margin, max_val + margin)

    def _update_time_axis(self, *args):
        view_range = None
        for arg in args:
            if isinstance(arg, (tuple, list, np.ndarray)) and len(arg) == 2:
                try:
                    float(arg[0]); float(arg[1])
                    view_range = arg
                    break
                except:
                    pass
        
        if view_range is None:
            try:
                view_range = self.plot_widget.viewRange()[0]
            except:
                return

        min_x, max_x = view_range
        max_time = (max_x - min_x) / self.fs
        if max_time <= 0: return
        
        step = max_time / 10.0
        mag = 10 ** np.floor(np.log10(step))
        residual = step / mag
        if residual <= 1.5: nice_step = 1 * mag
        elif residual <= 3.5: nice_step = 2 * mag
        elif residual <= 7.5: nice_step = 5 * mag
        else: nice_step = 10 * mag

        start_tick_time = np.floor(min_x / self.fs / nice_step) * nice_step
        ticks_time = np.arange(start_tick_time, max_x / self.fs, nice_step)
        
        tick_items = [list(zip(ticks_time * self.fs, [f"{t:.1f}s" for t in ticks_time]))]
        self.x_axis.setTicks(tick_items)

    def _on_roi_changed(self):
        if self._updating: return
        self._updating = True
        start, end = self.roi.getRegion()
        start, end = int(start), int(end)
        self.range_changed.emit(start, end)
        self._updating = False
        
    def update_roi_from_external(self, start_sample, end_sample):
        self._updating = True
        self.roi.setRegion([start_sample, end_sample])
        self._updating = False
        self.range_changed.emit(start_sample, end_sample)