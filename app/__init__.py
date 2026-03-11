import os

# Корневая директория проекта (Holter)
# __file__ это .../Holter/app/__init__.py
# dirname(__file__) -> .../Holter/app
# dirname(dirname(__file__)) -> .../Holter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ИСПРАВЛЕНО: Указываем путь через resources
STYLES_DIR = os.path.join(BASE_DIR, 'resources', 'styles')
MODELS_DIR = os.path.join(BASE_DIR, 'app', 'models')
DB_DIR = os.path.join(BASE_DIR, 'DB')