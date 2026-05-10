from PyQt5.QtWidgets import QToolBar, QAction, QComboBox, QLabel, QStyle, QCheckBox
from PyQt5.QtCore import QSize, pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont, QColor

class AppToolBar(QToolBar):
    save_project_clicked = pyqtSignal()
    detect_r_clicked = pyqtSignal()
    analyze_clicked = pyqtSignal()
    export_wfdb_clicked = pyqtSignal()
    lead_changed = pyqtSignal(int)
    view_auto_toggled = pyqtSignal(bool) 
    visual_std_toggled = pyqtSignal(bool) # Новый сигнал стандартизации

    def __init__(self):
        super().__init__("Main Toolbar")
        self.setMovable(False)
        self.setIconSize(QSize(24, 24))

        # 1. Экспорт WFDB
        self.act_exp_wfdb = QAction(self.style().standardIcon(QStyle.SP_DriveFDIcon), "Экспорт сигнала (WFDB)", self)
        self.act_exp_wfdb.setToolTip("Экспортировать сигнал и аннотации в формат WFDB")
        self.act_exp_wfdb.triggered.connect(self.export_wfdb_clicked)
        self.addAction(self.act_exp_wfdb)

        self.addSeparator()

        # 2. Детекция R-пиков
        r_icon = self._create_text_icon("R", QColor("white"), QColor("darkred"))
        self.act_r_peak = QAction(r_icon, "Детекция R-пиков", self)
        self.act_r_peak.setToolTip("Найти R-пики (NeuroKit2)")
        self.act_r_peak.triggered.connect(self.detect_r_clicked)
        self.addAction(self.act_r_peak)

        # 3. Анализ (Классификация)
        self.act_analyze = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Классификация", self)
        self.act_analyze.setToolTip("Запуск каскадного классификатора")
        self.act_analyze.triggered.connect(self.analyze_clicked)
        self.addAction(self.act_analyze)

        self.addSeparator()

        # 4. Выбор отведения
        self.addWidget(QLabel(" Канал: "))
        self.combo_lead = QComboBox()
        self.combo_lead.currentIndexChanged.connect(self.lead_changed)
        self.addWidget(self.combo_lead)

        # 5. Сохранить проект
        self.act_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Сохранить проект", self)
        self.act_save.setToolTip("Сохранить проект и разметку")
        self.act_save.triggered.connect(self.save_project_clicked)
        self.addAction(self.act_save)

        self.addSeparator()

        # 6. Просмотр автоматической классификации
        self.cb_view_auto = QCheckBox("Просмотр авто-классификации")
        self.cb_view_auto.setToolTip("Если отмечено, отображаются только метки классификатора (метки врача игнорируются)")
        self.cb_view_auto.toggled.connect(self.view_auto_toggled)
        self.addWidget(self.cb_view_auto)

        self.addSeparator()

        # 7. Визуальное центрирование изолинии
        self.cb_visual_std = QCheckBox("Центрирование")
        self.cb_visual_std.setToolTip("Визуально выровнять изолинию (сместить к нулю)")
        self.cb_visual_std.toggled.connect(self.visual_std_toggled)
        self.addWidget(self.cb_visual_std)

        self.addSeparator()

    def set_leads(self, leads):
        self.combo_lead.clear()
        self.combo_lead.addItems(leads)

    def _create_text_icon(self, text, text_color, bg_color):
        pixmap = QPixmap(24, 24)
        painter = QPainter(pixmap)
        painter.fillRect(0, 0, 24, 24, bg_color)
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        painter.setPen(text_color)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
        painter.end()
        return QIcon(pixmap)