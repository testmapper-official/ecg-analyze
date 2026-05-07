# -*- coding: utf-8 -*-
"""ЭТАП 2: Rule-Based + V5 Верификация"""

import numpy as np
from scipy.spatial.distance import cosine
import archive.config as config

class MorphologyClassifier:
    def __init__(self):
        self.ref_morphology = None

    def process_beat(self, tcn_ii_class_str, v5_class_str, rr_prev, rr_mean, raw_morph):
        """
        Добавлен параметр v5_class_str (может быть None)
        """
        if tcn_ii_class_str == 'Normal':
            return config.MORPH_CLASSES.index('Normal')
        
        # Если II отведение сказало "Блокада" или "ПЖС"
        is_blockade = False
        
        if tcn_ii_class_str == 'Blockade':
            # ЛОГИКА ВЕРИФИКАЦИИ (ЭКСПЕРТНАЯ РЕКОМЕНДАЦИЯ)
            if v5_class_str is not None:
                # V5 имеет приоритет в дифференциации блокад!
                if v5_class_str == 'Blockade':
                    is_blockade = True # V5 подтвердило блокаду
                elif v5_class_str == 'PVC':
                    is_blockade = False # V5 перезаписало диагноз на ПЖС
                else: # V5 сказало Normal (странный случай, fallback)
                    rr_ratio = rr_prev / rr_mean if rr_mean > 0 else 1.0
                    is_blockade = (rr_ratio >= config.PREMATURE_THRESHOLD)
            else:
                # Если V5 НЕТ (базовый режим Холтера) - работаем по старым RR правилам
                rr_ratio = rr_prev / rr_mean if rr_mean > 0 else 1.0
                is_blockade = (rr_ratio >= config.PREMATURE_THRESHOLD)

        if is_blockade:
            return config.MORPH_CLASSES.index('Blockade')
        
        # Если это ПЖС (по II, по V5 или по OVERRID-у) -> классифицируем подтип
        return self._classify_pvc_subtype(rr_prev, rr_mean, raw_morph)

    def _classify_pvc_subtype(self, rr_prev, rr_mean, raw_morph):
        qt_est = 0.4 * np.sqrt(rr_mean / 1000) * 1000
        if rr_prev < (0.8 * qt_est):
            return config.MORPH_CLASSES.index('R_on_T')
        
        if self.ref_morphology is None:
            self.ref_morphology = raw_morph
            return config.MORPH_CLASSES.index('PVC_Monomorphic')
            
        sim = 1 - cosine(self.ref_morphology, raw_morph)
        if sim > 0.7: return config.MORPH_CLASSES.index('PVC_Monomorphic')
        else: return config.MORPH_CLASSES.index('PVC_Polymorphic')