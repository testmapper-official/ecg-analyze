from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QFont

class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(35)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)
        
        self.title = QLabel("Holter Monitor")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        # Кнопки управления
        self.minimize_btn = QPushButton("—")
        self.maximize_btn = QPushButton("□")
        self.close_btn = QPushButton("×")
        
        btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                font-weight: bold;
                font-size: 14px;
                padding: 0px 5px;
            }
            QPushButton:hover {
                background-color: rgba(200, 200, 200, 0.3);
            }
        """
        
        for btn in [self.minimize_btn, self.maximize_btn, self.close_btn]:
            btn.setFixedSize(25, 25)
            btn.setStyleSheet(btn_style)
        
        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        self.maximize_btn.clicked.connect(self._toggle_maximize)
        self.close_btn.clicked.connect(self.parent.close)
        
        self.layout.addWidget(self.title)
        self.layout.addStretch()
        self.layout.addWidget(self.minimize_btn)
        self.layout.addWidget(self.maximize_btn)
        self.layout.addWidget(self.close_btn)
        
        self.start = QPoint(0, 0)
        self.pressing = False

    def _toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.type() == event.MouseButtonDblClick:
            self._toggle_maximize()
        else:
            self.start = self.mapToGlobal(event.pos())
            self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            end = self.mapToGlobal(event.pos())
            movement = end - self.start
            self.parent.move(self.parent.pos() + movement)
            self.start = end

    def mouseReleaseEvent(self, event):
        self.pressing = False