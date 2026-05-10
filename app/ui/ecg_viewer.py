import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor
import numpy as np

DANGER_MAP = {
    'r': 3, 'VT': 3, 'VF': 3, 'Couplet': 3, 
    'B': 2, 'L': 2, 'R': 2, 'SVT': 2, 'SB': 2, 'Bigeminy': 2, 'Trigeminy': 2, 
    'V': 1, 'M': 1, 'P': 1, 'i': 1, 'A': 1, 'F': 1, 'E': 1, 'e': 1   
}

class ClickableRegionItem(pg.LinearRegionItem):
    clicked = pyqtSignal(dict)
    def __init__(self, data_dict, **kwargs):
        super().__init__(**kwargs)
        self.data_dict = data_dict
        self.setMovable(False)
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton: self.clicked.emit(self.data_dict)
        ev.accept()

class ECGViewer(QWidget):
    zoom_requested = pyqtSignal(int)
    annotation_clicked = pyqtSignal(dict) 

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude', units='mV')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('k', width=1.5))
        self.r_peak_scatter = pg.ScatterPlotItem(pxMode=True, size=7, brush=pg.mkBrush('r'), pen=pg.mkPen(None))
        self.plot_widget.addItem(self.r_peak_scatter)
        self.layout.addWidget(self.plot_widget)
        
        self.fs = 360
        self.orig_fs = 360
        self.current_lead_idx = 0
        self.current_start_sample = 0 
        self.current_segment = None
        self.avg_rr_samples = 360 
        self.annotation_items = []
        
        # Флаг визуальной стандартизации
        self.visual_standardize = False

    def set_visual_standardize(self, enabled):
        self.visual_standardize = enabled

    def update_view(self, start_sample_orig, end_sample_orig, signal_data, orig_fs, lead_idx, avg_rr):
        self.orig_fs = orig_fs
        self.current_lead_idx = lead_idx
        self.current_start_sample = start_sample_orig
        self.avg_rr_samples = avg_rr
        
        if signal_data is None: return
        segment = signal_data[start_sample_orig:end_sample_orig, lead_idx]
        if len(segment) == 0: return

        # Визуальное центрирование (смещение к нулю)
        if self.visual_standardize:
            segment = segment - np.mean(segment)
            
        self.current_segment = segment

        start_time_sec = start_sample_orig / self.orig_fs
        time_array = (np.arange(len(segment)) / self.orig_fs) + start_time_sec
        
        self.curve.setData(time_array, segment)
        min_val, max_val = np.min(segment), np.max(segment)
        
        # ИСПРАВЛЕНО: Симметричный масштаб по Y при центрировании
        if self.visual_standardize:
            max_abs = max(abs(min_val), abs(max_val), 0.5)
            margin = max_abs * 0.15
            # Ось Y симметрична от -max до +max, ноль ровно по центру экрана
            self.plot_widget.setYRange(-max_abs - margin, max_abs + margin, padding=0)
        else:
            margin = max((max_val - min_val) * 0.15, 0.5) 
            self.plot_widget.setYRange(min_val - margin, max_val + margin, padding=0)
            
        self.plot_widget.setXRange(time_array[0], time_array[-1], padding=0.0)

    def set_annotations_and_highlights(self, annotations_list, start_sample_orig, end_sample_orig, view_auto_mode=False):
        for item in self.annotation_items:
            self.plot_widget.removeItem(item)
        self.annotation_items.clear()
        self.r_peak_scatter.setData([])

        ratio_orig_to_360 = 360.0 / self.orig_fs
        r_peak_points = []
        singles = [ann for ann in annotations_list if not ann.get('is_rhythm')]

        for ann in singles:
            s360 = ann['sample']
            if view_auto_mode:
                sym = ann.get('auto_symbol', '')
            else:
                sym = ann.get('manual_symbol') if ann.get('manual_symbol') else ann.get('auto_symbol', '')
                
            s_orig = int(s360 / ratio_orig_to_360)
            
            if start_sample_orig <= s_orig < end_sample_orig:
                x_val = s_orig / self.orig_fs
                idx_in_seg = s_orig - start_sample_orig
                if 0 <= idx_in_seg < len(self.current_segment):
                    y_val = self.current_segment[idx_in_seg]
                    r_peak_points.append({'pos': (x_val, y_val)})
                else:
                    y_val = 0
                
                danger = DANGER_MAP.get(sym, 0)
                if danger > 0:
                    half_win_sec = 0.06
                    brush_color = self._get_brush_for_danger(danger, alpha_mult=1.0)
                    region = ClickableRegionItem(data_dict=ann, values=[x_val - half_win_sec, x_val + half_win_sec], brush=brush_color)
                    region.lines[0].setPen(pg.mkPen(None)); region.lines[1].setPen(pg.mkPen(None))
                    region.setZValue(-10)
                    region.clicked.connect(self.annotation_clicked.emit)
                    self.plot_widget.addItem(region); self.annotation_items.append(region)
                
                text_color = 'k' if ann.get('manual_symbol') and not view_auto_mode else 'gray'
                text_item = pg.TextItem(text=sym, color=text_color, anchor=(0.5, 1.2))
                text_item.setPos(x_val, y_val)
                text_item.setFont(pg.QtGui.QFont('Arial', 12, pg.QtGui.QFont.Bold))
                self.plot_widget.addItem(text_item); self.annotation_items.append(text_item)

        self.r_peak_scatter.setData(r_peak_points)

    def _get_brush_for_danger(self, level, alpha_mult=1.0):
        alpha = int(60 * alpha_mult)
        if level == 3: return pg.mkBrush(255, 0, 0, alpha)
        elif level == 2: return pg.mkBrush(255, 165, 0, alpha)
        elif level == 1: return pg.mkBrush(0, 255, 0, alpha)
        return None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0: self.zoom_requested.emit(-self.avg_rr_samples)
        else: self.zoom_requested.emit(self.avg_rr_samples)
        event.accept()