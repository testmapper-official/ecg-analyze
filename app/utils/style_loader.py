import os
from PyQt5.QtWidgets import QApplication
from app import STYLES_DIR

class StyleLoader:
    def __init__(self):
        self.available_styles = self._discover_styles()
        
    def _discover_styles(self):
        """Автоматически находит все .qss файлы в папке styles."""
        styles = {}
        if os.path.exists(STYLES_DIR):
            for file in os.listdir(STYLES_DIR):
                if file.endswith('.qss'):
                    name = file.split('.')[0]
                    styles[name] = os.path.join(STYLES_DIR, file)
        return styles

    def apply_style(self, app: QApplication, style_name: str):
        if style_name in self.available_styles:
            path = self.available_styles[style_name]
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
                    app.setStyleSheet(stylesheet)
            except Exception as e:
                print(f"Error loading style {style_name}: {e}")
        else:
            print(f"Style {style_name} not found. Available: {list(self.available_styles.keys())}")

    def get_available_styles(self):
        return list(self.available_styles.keys())