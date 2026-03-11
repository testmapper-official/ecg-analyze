import sys
import os
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow
from app.utils.style_loader import StyleLoader

def main():
    # Создание приложения
    app = QApplication(sys.argv)
    
    # Установка базовых параметров
    app.setApplicationName("Holter Monitor")
    
    # Загрузка стилей (по умолчанию белый)
    style_loader = StyleLoader()
    style_loader.apply_style(app, "white")
    
    # Создание и показ главного окна
    window = MainWindow(style_loader)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()