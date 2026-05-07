import os
import wfdb
import shutil
import time
import requests.exceptions

# Список кортежей: (Название, ID на PhysioNet, Имя папки в проекте)
DATABASES = [
    ("MIT-BIH Arrhythmia", "mitdb", "mitdb"),
    ("Sudden Cardiac Death Holter", "sddb", "sddb"),
    ("Creighton University Ventricular Tachyarrhythmia", "cudb", "cudb"),
    ("St. Petersburg Institute 12-lead Arrhythmia", "incartdb", "incartdb"),
    ("Brugada-HUCA Syndrome", "brugada-huca", "brugadahuca"),
    ("PTB Diagnostic ECG", "ptbdb", "ptbdb"),
    ("European ST-T", "edb", "edb")
]

def main():
    root_dir = "DB"
    
    # Создаем корневую папку
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(f"[INIT] Создана папка {root_dir}")

    total = len(DATABASES)
    print(f"Начинаю загрузку {total} баз данных через wfdb...\n")

    for i, (name, db_id, folder_name) in enumerate(DATABASES):
        idx = i + 1
        
        # Итоговый путь: DB/mitdb
        target_path = os.path.join(root_dir, folder_name)
        
        print(f"[{idx}/{total}] {name}")
        print(f"    [TARGET] Папка назначения: {target_path}")
        
        # Если папка уже есть и не пуста — пропускаем
        if os.path.exists(target_path) and os.listdir(target_path):
            print(f"    [SKIP] База уже скачана.\n")
            continue
            
        # --- НАЧАЛО ПРОЦЕССА СКАЧИВАНИЯ ---
        start = time.time()
        
        # Имя временной папки
        temp_dir_name = f"_temp_{db_id}"
        temp_dir_path = os.path.join(root_dir, temp_dir_name)
        
        try:
            # 1. Чистим мусор, если остался с прошлого раза
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path)
            
            # 2. Скачиваем во временную папку
            # wfdb ОБЯЗАТЕЛЬНО создаст внутри подпапку с именем db_id.
            # То есть файлы будут здесь: DB/_temp_mitdb/mitdb/файлы
            print(f"    [DOWNLOAD] Скачивание (ID: {db_id})...")
            wfdb.dl_database(db_id, dl_dir=temp_dir_path)
            
            # 3. Определяем, где лежат файлы
            downloaded_data_path = os.path.join(temp_dir_path, db_id)
            
            if not os.path.exists(downloaded_data_path):
                # Редкий случай, если wfdb изменил поведение
                downloaded_data_path = temp_dir_path

            # 4. Создаем итоговую папку (DB/mitdb)
            os.makedirs(target_path, exist_ok=True)
            
            # 5. Переносим файлы
            files_moved = 0
            for item in os.listdir(downloaded_data_path):
                src = os.path.join(downloaded_data_path, item)
                dst = os.path.join(target_path, item)
                shutil.move(src, dst)
                files_moved += 1
                
            # 6. Удаляем пустую временную папку
            shutil.rmtree(temp_dir_path)
            
            elapsed = time.time() - start
            print(f"    [SUCCESS] Скачано файлов: {files_moved}. Время: {elapsed:.1f} сек.")
            print(f"    [RESULT] Данные в: {target_path}\n")

        except requests.exceptions.ConnectionError:
            print(f"    [ERROR] Нет связи с сервером PhysioNet (Timeout).")
            print(f"    [HINT] Проверьте интернет или включите VPN.\n")
            # Удаляем хвосты
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path)
                
        except Exception as e:
            print(f"    [ERROR] Ошибка: {e}\n")
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    main()