import os
import warnings
warnings.filterwarnings("ignore")

from app.training.blk.dataset import DatasetBuilder
from app.training.blk.model import TCNClassifier
from app.training.blk.trainer import Trainer

def main():
    DB_ROOT = 'DB'
    CURRENT_CHANNEL = 'BLK' # Уникальное имя папки для сохранения
    
    print(f"--- ЭТАП 1: Подготовка данных (Канал: II, Классы: Normal vs Blockade) ---")
    builder = DatasetBuilder(db_root=DB_ROOT)
    X_train, y_train, sym_train, X_val, y_val, sym_val, X_test, y_test, sym_test = builder.build()
    
    if len(X_train) == 0:
        print("ОШИБКА: Нет данных для обучения!")
        return
        
    print(f"Train: {X_train.shape[0]} выборок")
    print(f"Val: {X_val.shape[0]} выборок")
    print(f"Test: {X_test.shape[0]} выборок")
    
    print("\n--- ЭТАП 2: Инициализация модели ---")
    model = TCNClassifier(num_classes=2)
    
    print("\n--- ЭТАП 3: Обучение ---")
    trainer = Trainer(channel_name=CURRENT_CHANNEL)
    model, history = trainer.train(model, X_train, y_train, X_val, y_val, epochs=120)
    
    print("\n--- ЭТАП 4: Тестирование и визуализация ---")
    trainer.evaluate_and_plot(model, X_test, y_test, sym_test, history, builder)
    print(f"Готово! Результаты сохранены в папку 'results/{CURRENT_CHANNEL}/'")

if __name__ == "__main__":
    main()