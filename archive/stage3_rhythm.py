# -*- coding: utf-8 -*-
"""ЭТАП 3: Конечный автомат анализа ритма (Бигеминия, Тригеминия, Тахикардия)"""

import archive.config as config

class RhythmAnalyzer:
    def __init__(self):
        self.sequence_buffer = [] # Храним последние 10 ударов: [{'morph': str, 'rr': int}]

    def process_beat(self, morph_class_str, rr_ms):
        """
        Добавляет удар в буфер и возвращает текущий статус ритма.
        """
        self.sequence_buffer.append({'morph': morph_class_str, 'rr': rr_ms})
        if len(self.sequence_buffer) > 10:
            self.sequence_buffer.pop(0)

        n = len(self.sequence_buffer)
        rhythm = config.RHYTHM_CLASSES[0] # По умолчанию 'Normal_Sinus'

        # 1. Проверка на Тахикардию (>= 3 широких комплексов подряд, RR < 600 мс)
        if n >= 3:
            last_3 = self.sequence_buffer[-3:]
            is_wide = all(b['morph'] != 'Normal' for b in last_3)
            if is_wide:
                avg_rr = sum(b['rr'] for b in last_3) / 3
                if avg_rr < config.VT_RATE_THRESHOLD:
                    rhythm = config.RHYTHM_CLASSES[3] # 'Ventricular_Tachycardia'
                    return rhythm

        # 2. Проверка на Бигеминию (N, PVC, N, PVC)
        if n >= 4:
            seq = [b['morph'] for b in self.sequence_buffer[-4:]]
            # Допускаем любые ПЖС подтипы как паттерн
            pat1 = [seq[0] == 'Normal', seq[1] != 'Normal']
            pat2 = [seq[2] == 'Normal', seq[3] != 'Normal']
            if all(pat1) and all(pat2):
                rhythm = config.RHYTHM_CLASSES[1] # 'Bigeminy'

        # 3. Проверка на Тригеминию (N, N, PVC, N, N, PVC)
        if n >= 6:
            seq = [b['morph'] for b in self.sequence_buffer[-6:]]
            pat1 = [seq[0] == 'Normal', seq[1] == 'Normal', seq[2] != 'Normal']
            pat2 = [seq[3] == 'Normal', seq[4] == 'Normal', seq[5] != 'Normal']
            if all(pat1) and all(pat2):
                rhythm = config.RHYTHM_CLASSES[2] # 'Trigeminy'

        return rhythm
        
    def reset_state(self):
        self.sequence_buffer = []