import os

# Базовый путь к проекту
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
STYLES_DIR = os.path.join(RESOURCES_DIR, 'styles')

# Частота дискретизации по умолчанию
DEFAULT_SAMPLING_RATE = 360