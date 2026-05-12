import warnings
warnings.filterwarnings("ignore")
import numpy as np
from app.training.parametric.dataset import ParametricDatasetBuilder
from app.training.parametric.model import MLPClassifier
from app.training.parametric.trainer import ParametricTrainer

def main():
    DB_ROOT = 'DB'
    print("--- ЭТАП 1: Подготовка параметрических данных ---")
    builder = ParametricDatasetBuilder(db_root=DB_ROOT)
    X_train, y_train, X_val, y_val, X_test, y_test, sym_test = builder.build()
    
    print(f"Train: {X_train.shape[0]} выборок")
    print(f"Val: {X_val.shape[0]} выборок")
    print(f"Test: {X_test.shape[0]} выборок")
    
    input_dim = X_train.shape[1]
    
    print("\n--- ЭТАП 2: Инициализация модели ---")
    model = MLPClassifier(input_dim=input_dim, num_classes=2)
    
    print("\n--- ЭТАП 3: Обучение ---")
    trainer = ParametricTrainer(models_dir='models/parametric', results_dir='results/parametric')
    model = trainer.train(model, X_train, y_train, X_val, y_val, epochs=150)
    
    print("\n--- ЭТАП 4: Тестирование и визуализация ---")
    print("Генерация зашумленных тестовых данных...")
    X_noisy, y_noisy, sym_noisy = builder.generate_noisy_test_set(X_test, y_test, sym_test)
    
    trainer.evaluate_and_plot(model, X_test, y_test, sym_test, X_noisy, y_noisy, sym_noisy)
    print("Готово! Модель и результаты сохранены в models/parametric/ и results/parametric/")

if __name__ == "__main__":
    main()