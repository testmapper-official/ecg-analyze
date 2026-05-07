import os
import shutil

# --- НАСТРОЙКИ ---
SRC_DIR = 'DB'         # Корневая папка с исходными базами
DEST_DIR = os.path.join('DB', 'mydb')  # Папка mydb создается внутри DB

# Словарь баз данных: ключ - имя базы, значение - путь относительно SRC_DIR
DB_MAP = {
    'mitdb': 'mitdb',
    'sddb': 'sddb',
    'cudb': 'cudb',
    'ptbdb': 'ptbdb',
    'incartdb': 'incartdb/files',
    'edb': 'edb'
}

# --- ПРАВИЛА ФИЛЬТРАЦИИ ---
# Каналы, которые относятся ко второму отведению
LEAD_II_VARIANTS = ['ii', 'mlii']
MIN_DURATION_SEC = 300  # Минимальная длина записи в секундах (5 минут)
EXTENSIONS = ['dat', 'hea', 'xyz', 'atr', 'hea-', 'xws']

def parse_hea_file(hea_path):
    """Парсит .hea файл для получения частоты дискретизации, кол-ва отсчетов и имен каналов."""
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        header_line = lines[0].strip().split()
        n_sig = int(header_line[1])
        fs = float(header_line[2])
        n_samples = int(header_line[3])
        
        duration = n_samples / fs if fs > 0 else 0
        
        lead_names = []
        for i in range(1, n_sig + 1):
            if i < len(lines):
                sig_line = lines[i].strip().split()
                lead_names.append(sig_line[-1].lower())
                
        return duration, lead_names
    except Exception as e:
        print(f"Ошибка чтения {hea_path}: {e}")
        return 0, []

def copy_record_files(src_db_path, record_name, dest_folder_path, base_folder_name):
    """Копирует все существующие расширения записи в целевую папку."""
    os.makedirs(dest_folder_path, exist_ok=True)
    for ext in EXTENSIONS:
        src_file = os.path.join(src_db_path, f"{record_name}.{ext}")
        dst_file = os.path.join(dest_folder_path, f"{base_folder_name}.{ext}")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)

def main():
    # Создаем папки назначения
    os.makedirs(DEST_DIR, exist_ok=True)
    
    accepted_records = []
    used_folder_names = {} # Словарь для отслеживания дубликатов имен папок

    for db_name, db_rel_path in DB_MAP.items():
        db_path = os.path.join(SRC_DIR, db_rel_path)
        records_file = os.path.join(db_path, 'RECORDS')
        
        if not os.path.exists(records_file):
            print(f"Файл RECORDS не найден в {db_path}, пропуск...")
            continue
            
        with open(records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
            
        print(f"\nОбработка базы: {db_name} ({len(records)} записей)")
        
        for record_name in records:
            src_hea = os.path.join(db_path, f"{record_name}.hea")
            
            if not os.path.exists(src_hea):
                continue
                
            duration, leads = parse_hea_file(src_hea)
            
            if duration < MIN_DURATION_SEC:
                continue
            
            # Проверка наличия нужных каналов
            has_lead_ii = any(lead in LEAD_II_VARIANTS for lead in leads)
            has_v5 = 'v5' in leads
            
            # Если нет ни II/MLII, ни V5 - пропускаем запись
            if not (has_lead_ii or has_v5):
                continue
            
            # --- УСЛОВИЯ ВЫПОЛНЕНЫ ---
            
            # Формируем базовое имя папки для записи
            base_folder_name = record_name.replace('/', '_').replace('\\', '_')
            
            # Обработка дубликатов имен
            if base_folder_name in used_folder_names:
                used_folder_names[base_folder_name] += 1
                new_folder_name = f"{base_folder_name}_{used_folder_names[base_folder_name]}"
            else:
                used_folder_names[base_folder_name] = 1
                new_folder_name = base_folder_name
            
            # --- КОПИРОВАНИЕ И РАСПРЕДЕЛЕНИЕ ---
            dest_statuses = []
            
            if has_lead_ii:
                # Путь: DB/mydb/II/имя_папки/
                dest_folder_path = os.path.join(DEST_DIR, 'II', new_folder_name)
                copy_record_files(db_path, record_name, dest_folder_path, base_folder_name)
                # Путь для RECORDS в формате wfdb: II/имя_папки/имя_файла
                accepted_records.append(f"II/{new_folder_name}/{base_folder_name}")
                dest_statuses.append("II")
                
            if has_v5:
                # Путь: DB/mydb/V5/имя_папки/
                dest_folder_path = os.path.join(DEST_DIR, 'V5', new_folder_name)
                copy_record_files(db_path, record_name, dest_folder_path, base_folder_name)
                # Путь для RECORDS в формате wfdb: V5/имя_папки/имя_файла
                accepted_records.append(f"V5/{new_folder_name}/{base_folder_name}")
                dest_statuses.append("V5")

            print(f"  + Добавлена: {record_name} -> {new_folder_name} (Каналы: {leads} | Папки: {', '.join(dest_statuses)})")

    # Запись файла RECORDS в mydb
    records_out_path = os.path.join(DEST_DIR, 'RECORDS')
    with open(records_out_path, 'w') as f:
        for rec in accepted_records:
            f.write(f"{rec}\n")
            
    print(f"\nГотово! Успешно собрано {len(accepted_records)} позиций в папку {DEST_DIR}.")

if __name__ == '__main__':
    main()