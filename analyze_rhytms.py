import os

def analyze_rhythm_md(md_filepath='results/Cascade/validation_and_rhythms.md'):
    if not os.path.exists(md_filepath):
        print(f"Файл {md_filepath} не найден! Сначала запустите оценку каскада.")
        return

    current_rhythm = None
    total_episodes = 0
    confident_episodes = 0 # Достоверность > 50%
    
    rhythm_stats = {}

    with open(md_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Ищем заголовок ритма
            if line.startswith("### Ритм:"):
                current_rhythm = line.replace("### Ритм:", "").strip()
                if current_rhythm not in rhythm_stats:
                    rhythm_stats[current_rhythm] = {'total': 0, 'confident': 0}
                continue
            
            # Ищем строки таблицы (начинаются с |)
            if line.startswith("|") and current_rhythm:
                # Разбиваем по | и фильтруем пустые строки, возникающие из-за крайних |
                parts = [p.strip() for p in line.split("|") if p.strip()]
                
                # Валидная строка таблицы содержит 4 колонки: Запись, Начало, Конец, Достоверность
                if len(parts) >= 4:
                    try:
                        # Последний элемент в строке — достоверность
                        conf_str = parts[-1].replace('%', '').strip()
                        conf_val = float(conf_str)
                        
                        rhythm_stats[current_rhythm]['total'] += 1
                        total_episodes += 1
                        
                        if conf_val > 50.0:
                            rhythm_stats[current_rhythm]['confident'] += 1
                            confident_episodes += 1
                    except ValueError:
                        pass # Пропускаем заголовки и разделители таблицы

    # Вывод результатов
    print("\n" + "="*60)
    print("АНАЛИЗ ОТЧЕТА ПО РИТМАМ")
    print("="*60)
    
    if not rhythm_stats:
        print("Ритмы не найдены в файле.")
        return
        
    for rhy, stats in rhythm_stats.items():
        total = stats['total']
        conf = stats['confident']
        perc = (conf / total * 100) if total > 0 else 0
        print(f"Ритм: {rhy}")
        print(f"  Всего эпизодов найдено: {total}")
        print(f"  Уверенных эпизодов (>50%): {conf} ({perc:.1f}%)")
        print("-" * 30)
        
    print(f"\nИТОГО:")
    print(f"  Всего эпизодов найдено: {total_episodes}")
    print(f"  Уверенных эпизодов (>50%): {confident_episodes} ({(confident_episodes/total_episodes*100 if total_episodes else 0):.1f}%)")
    print("="*60)

if __name__ == "__main__":
    analyze_rhythm_md()