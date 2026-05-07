import os
import random
import wfdb
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# --- НАСТРОЙКИ ---
SRC_DIR = 'DB'
MYDB_PATH = os.path.join(SRC_DIR, 'mydb')
RECORDS_FILE = os.path.join(MYDB_PATH, 'RECORDS')

# Символы аннотаций WFDB, которые мы ищем
SYMBOL_MAP = {
    'N': 'Normal QRS',
    'L': 'LBBB (Left Bundle Branch Block)',
    'R': 'RBBB (Right Bundle Branch Block)',
    'V': 'PVC (Premature Ventricular Contraction)',
    'E': 'Ventricular Escape',
    'A': 'APC (Atrial Premature Contraction)',
    'F': 'Fusion of normal and PVC'
}

# Сколько примеров каждого класса показать
EXAMPLES_TO_SHOW = 5
# Сколько примеров хранить в памяти для случайной выборки (чтобы не переполнить память для класса N)
MAX_EXAMPLES_STORED = 200 

def main():
    if not os.path.exists(RECORDS_FILE):
        print(f"Файл RECORDS не найден в {MYDB_PATH}")
        return
        
    with open(RECORDS_FILE, 'r') as f:
        records = [line.strip() for line in f if line.strip()]
            
    print(f"Анализ базы mydb ({len(records)} путей в RECORDS). Это может занять некоторое время...")
    
    total_stats = Counter()
    processed_ids = set()
    skipped_duplicates = 0
    
    # Словарь для хранения примеров: { символ : [(путь_к_записи, индекс_удара), ...] }
    examples_dict = defaultdict(list)
    
    # --- ШАГ 1: Сбор статистики и путей к примерам ---
    for record_path in records:
        parts = record_path.split('/')
        if len(parts) < 3:
            continue
            
        record_id = parts[1]
        if record_id in processed_ids:
            skipped_duplicates += 1
            continue
            
        processed_ids.add(record_id)
        full_record_path = os.path.join(MYDB_PATH, record_path)
        
        try:
            ann = wfdb.rdann(full_record_path, 'atr')
            symbol_counts = Counter(ann.symbol)
            
            for sym in SYMBOL_MAP.keys():
                if sym in symbol_counts:
                    total_stats[sym] += symbol_counts[sym]
                    
                    # Сохраняем примеры, если не достигли лимита
                    if len(examples_dict[sym]) < MAX_EXAMPLES_STORED:
                        # Находим все индексы этого символа в текущей записи
                        idxs = [i for i, s in enumerate(ann.symbol) if s == sym]
                        for idx in idxs:
                            if len(examples_dict[sym]) < MAX_EXAMPLES_STORED:
                                examples_dict[sym].append((full_record_path, ann.sample[idx]))
            print(f"Обработана уникальная запись: {record_id}", end='\r')

        except Exception as e:
            continue

    print("\n\nСбор статистики завершен!")
    print(f"Уникальных записей обработано: {len(processed_ids)}")
    print(f"Дубликатов пропущено: {skipped_duplicates}")
    print(f"{'='*60}")
    print(f"ОБЩАЯ СТАТИСТИКА ПО БАЗЕ mydb (БЕЗ ДУБЛИРОВАНИЯ)")
    print(f"{'='*60}")
    for sym, name in SYMBOL_MAP.items():
        if total_stats[sym] > 0:
            print(f"  {name} [{sym}]: {total_stats[sym]}")
            
    # --- ШАГ 2: Выбор случайных примеров и отрисовка ---
    print(f"\nЗапуск визуализации. Будет показано по {EXAMPLES_TO_SHOW} случайных примеров каждого класса.")
    print("Закрывайте окно графика, чтобы увидеть следующий пример.")
    input("Нажмите Enter для начала показа графиков...")

    for sym, name in SYMBOL_MAP.items():
        if not examples_dict[sym]:
            continue
            
        # Выбираем случайные примеры из сохраненных
        chosen_examples = random.sample(examples_dict[sym], min(EXAMPLES_TO_SHOW, len(examples_dict[sym])))
        
        for i, (rec_path, sample_idx) in enumerate(chosen_examples, 1):
            try:
                # Читаем сам сигнал ЭКГ
                sig, fields = wfdb.rdsamp(rec_path)
                fs = fields['fs'] # Частота дискретизации
                
                # Определяем, какой канал отрисовывать (II или V5) на основе пути
                channel_name = 'II' if '\\II\\' in rec_path or '/II/' in rec_path else 'V5'
                channel_idx = 0
                # Ищем индекс нужного канала в заголовке
                sig_names_lower = [s.lower() for s in fields['sig_name']]
                if channel_name.lower() in sig_names_lower:
                    channel_idx = sig_names_lower.index(channel_name.lower())
                elif 'ii' in sig_names_lower or 'mlii' in sig_names_lower:
                    channel_idx = sig_names_lower.index([s for s in sig_names_lower if s in ['ii', 'mlii']][0])
                
                # Вырезаем окно сигнала: 0.4 сек до пика и 0.6 сек после
                before_samples = int(0.4 * fs)
                after_samples = int(0.6 * fs)
                start = max(0, sample_idx - before_samples)
                end = min(sig.shape[0], sample_idx + after_samples)
                
                signal_slice = sig[start:end, channel_idx]
                
                # Отрисовка
                plt.figure(figsize=(10, 4))
                plt.plot(signal_slice, color='blue', label=f'ECG Lead ({fields["sig_name"][channel_idx]})')
                
                # Отмечаем сам удар (R-пик)
                peak_pos_in_slice = sample_idx - start
                plt.axvline(x=peak_pos_in_slice, color='red', linestyle='--', label='Аннотация (R-пик)')
                
                plt.title(f"Класс: {name} [{sym}]\nЗапись: {os.path.basename(rec_path)} | Индекс сэмпла: {sample_idx}")
                plt.xlabel('Отсчеты (Samples)')
                plt.ylabel('Амплитуда (mV)')
                plt.legend()
                plt.grid(True)
                
                # Показываем график (скрипт остановится, пока окно не закроют)
                plt.show()
                
            except Exception as e:
                print(f"Ошибка при отрисовке {rec_path}: {e}")

if __name__ == '__main__':
    main()