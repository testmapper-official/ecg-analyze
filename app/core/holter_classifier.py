import os
import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import medfilt
from scipy.spatial.distance import cosine

from app.core.signal import Signal

# ==========================================
# АРХИТЕКТУРЫ МОДЕЛЕЙ (Оставьте ваши текущие, здесь они сокращены для читаемости)
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x): return self.conv(nn.functional.pad(x, (self.padding, 0)))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size(); y = self.squeeze(x).view(b, c); y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, dilation=1, use_se=True):
        super().__init__()
        self.conv1 = CausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_out, n_out, kernel_size, dilation)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.relu = nn.ReLU(); self.drop = nn.Dropout(0.3)
        self.se = SEBlock(n_out) if use_se else nn.Identity()
    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x))); out = self.conv2(out); out = self.se(out)
        return self.relu(out + self.downsample(x))

class TCN_PSS(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [64] * 5; dilations = [1, 2, 4, 8, 16]; layers = []
        for i in range(len(dilations)): layers.append(ResidualBlock(2 if i==0 else channels[i-1], channels[i], 5, dilations[i]))
        self.network = nn.Sequential(*layers); self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(channels[-1], 2)
    def forward(self, x): return self.fc(self.pool(self.network(x)).squeeze(-1))

class TCN_BLK(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [64] * 5; dilations = [1, 2, 4, 8, 16]; layers = []
        for i in range(len(dilations)): layers.append(ResidualBlock(1 if i==0 else channels[i-1], channels[i], 5, dilations[i]))
        self.network = nn.Sequential(*layers); self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1] + 1, 2)
    def forward(self, x):
        ecg = x[:, 0, :].unsqueeze(1); qrs_dur = x[:, 1, 0].unsqueeze(1)   
        features = self.pool(self.network(ecg)).squeeze(-1)
        combined = torch.cat([features, qrs_dur], dim=1)
        return self.fc(combined)


# ==========================================
# КЛАСС КАСКАДНОГО КЛАССИФИКАТОРА
# ==========================================
class HolterClassifier:
    def __init__(self, models_dir='models', device='auto'):
        if device == 'auto': self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = torch.device(device)
            
        self.pss_model = self._load_model(TCN_PSS, os.path.join(models_dir, 'TCN_PSS.pth'))
        self.blk_model = self._load_model(TCN_BLK, os.path.join(models_dir, 'TCN_BLK.pth'))
        
        self.pss_threshold = 0.5
        self.blk_threshold = 0.5
        self.last_pvc_vector = None

    def _load_model(self, ModelClass, path):
        model = ModelClass().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _prepare_window(self, signal_obj, peak_idx, rr_prev, target_samples=288):
        segment = signal_obj.get_segment(peak_idx, target_samples)
        if segment is None: return None, None, None
        
        # --- Вход для PSS (Чистый, без TTA) ---
        pss_input = np.zeros((2, target_samples))
        pss_input[0, :] = segment
        pss_input[1, :] = rr_prev / 288.0 
        
        # --- Вход для BLK (С дизерингом/TTA для сохранения зазубрин) ---
        # ФИКСИРУЕМ Зерно рандома по позиции пика! 
        # Это гарантирует, что при повторном прогоне той же записи шум будет идентичным.
        np.random.seed(peak_idx) 
        awgn_frag = np.random.normal(0, 1.0, target_samples)
        
        temp_sig = Signal(data=segment, fs=360)
        noisy_sig = temp_sig.add_noise(awgn_frag, snr_db_range=(25, 35))
        denoised_sig = noisy_sig.wavelet_denoise()
        denoised_sig.standardize()
        
        denoised_segment = denoised_sig.resampled_data
        qrs_dur = denoised_sig.get_qrs_duration_norm(denoised_segment) * 2.0 
        
        blk_input = np.zeros((2, target_samples))
        blk_input[0, :] = denoised_segment
        blk_input[1, :] = qrs_dur
        
        return torch.FloatTensor(pss_input).unsqueeze(0).to(self.device), \
            torch.FloatTensor(blk_input).unsqueeze(0).to(self.device), \
            qrs_dur

    def analyze_signal(self, signal_obj: Signal):
        if not signal_obj.annotations: return [], []
        QRS_SYMBOLS = ['N', 'L', 'R', 'B', 'V', 'E', 'r', 'A', 'a', 'J', 'j', 'S', 'F', 'e', '/', 'f']
        r_peaks = [ann['sample'] for ann in signal_obj.annotations if ann['symbol'] in QRS_SYMBOLS]
        rr_intervals = np.diff(r_peaks).tolist()
        rr_intervals.insert(0, int(0.8 * 360)) 
        
        raw_predictions = []
        for i, peak in enumerate(r_peaks):
            rr_prev = rr_intervals[i]
            pss_inp, blk_inp, qrs_dur = self._prepare_window(signal_obj, peak, rr_prev)
            
            if pss_inp is None:
                raw_predictions.append({'peak': peak, 'label': 'N', 'prob_pss': 0, 'prob_blk': 0, 'qrs_dur': 0, 'segment': None})
                continue
                
            with torch.no_grad():
                prob_pss = torch.softmax(self.pss_model(pss_inp), dim=1).cpu().numpy()[0, 1]
                prob_blk = torch.softmax(self.blk_model(blk_inp), dim=1).cpu().numpy()[0, 1]
                
            raw_predictions.append({
                'peak': peak, 'label': 'N', 'prob_pss': prob_pss, 'prob_blk': prob_blk,
                'qrs_dur': qrs_dur, 'segment': signal_obj.get_segment(peak, 288)
            })
        
        self.last_pvc_vector = None # Сброс референса для каждой записи
        
        # ПОРЯДОК АЛГОРИТМОВ
        raw_preds = self._apply_base_logic(raw_predictions, rr_intervals)
        raw_preds, rhythm_annotations = self._apply_rhythm_engine(raw_preds, rr_intervals)
        self._apply_isolated_blk_filter(raw_preds)
        self._apply_v_subclassification(raw_preds, rr_intervals)
        
        final_output = [{'sample': p['peak'], 'label': p['label']} for p in raw_preds]
        return final_output, rhythm_annotations

    # ---------------------------------------------------------
    # 1. БАЗОВАЯ ЛОГИКА (Исправлено пропуск V на чистом)
    # ---------------------------------------------------------
    def _apply_base_logic(self, raw_preds, rr_intervals):
        median_rr = np.median(rr_intervals) if len(rr_intervals) > 5 else 288
        all_qrs = [p['qrs_dur'] for p in raw_preds]
        median_qrs = np.median(all_qrs) if len(all_qrs) > 5 else 1.8
        wide_threshold = median_qrs * 1.15 
        
        for i, pred in enumerate(raw_preds):
            rr_prev = rr_intervals[i]
            rr_next = rr_intervals[i+1] if i+1 < len(rr_intervals) else median_rr
            qrs_dur = pred['qrs_dur']
            is_narrow = qrs_dur < wide_threshold 
            
            is_premature = rr_prev < (0.85 * median_rr)
            is_delayed = rr_prev > (1.80 * median_rr) 
            is_post_pvc = (i > 0 and raw_preds[i-1]['label'] in ['V', 'B', 'r', 'E'])
            
            # НОВОЕ: Проверка на интерполированность
            total_rr = rr_prev + rr_next
            is_interpolated = (1.7 * median_rr) <= total_rr <= (2.1 * median_rr)
            
            # 1. Сеть BLK доминирует
            if pred['prob_blk'] > self.blk_threshold and not is_premature:
                pred['label'] = 'B'; continue
                
            # 2. Сеть PSS. Стандартная преждевременная V
            if pred['prob_pss'] > self.pss_threshold and is_premature:
                pred['label'] = 'V'; continue
                
            # НОВОЕ: 2.1 Интерполированная V (Не преждевременная, но вставленная)
            # Требуем высокой уверенности сети И попадания в физиологический диапазон паузы
            if pred['prob_pss'] > 0.8 and is_interpolated:
                pred['label'] = 'V'; continue
                
            # 3. Escape
            thr = 2.20 * median_rr if is_post_pvc else 1.80 * median_rr
            if is_delayed and rr_prev > thr: 
                # Проверяем, что комплекс широкий (желудочковый escape)
                if not is_narrow:
                    pred['label'] = 'E'; continue
                    
            # 4. APC (Узкий + Преждевременный)
            if is_narrow and is_premature and rr_prev < (0.70 * median_rr): 
                pred['label'] = 'A'; continue
                
            pred['label'] = 'N'
            
        return raw_preds

    # ---------------------------------------------------------
    # 2. ДВИЖОК РИТМОВ (Скользящее окно 6 комплексов)
    # ---------------------------------------------------------
    def _apply_rhythm_engine(self, raw_preds, rr_intervals):
        # Подготовка данных для универсального движка
        seq = [{'sample': p['peak'], 'group': self._map_group(p['label'])} for p in raw_preds]
        
        # Вызов единого метода
        detected_rhythms = self.detect_rhythms(seq)
        
        # Форматирование обратно в оригинальный формат классификатора
        rhythm_annotations = []
        for r in detected_rhythms:
            rhythm_annotations.append({
                'start_sample': r['start_sample'],
                'end_sample': r['end_sample'],
                'rhythm': f"({r['type']}" # Сохраняем оригинальный формат со скобкой
            })
            
        return raw_preds, rhythm_annotations

    @staticmethod
    def _map_group(label):
        """Маппинг меток в группы для универсального движка"""
        # ДОБАВЛЕНА 'i' ДЛЯ ИНТЕРПОЛИРОВАННЫХ
        if label in ['V', 'F', 'r', 'i', 'M', 'P']: return 'V'
        if label in ['B', 'L', 'R']: return 'B'
        if label in ['A', 'a']: return 'A'
        if label == 'N': return 'N'
        return 'O'

    @staticmethod
    def detect_rhythms(seq):
        """
        Единый универсальный движок поиска ритмов.
        На вход: [{'sample': int, 'group': str (V, B, A, N)}]
        На выход: [{'start_sample': int, 'end_sample': int, 'type': str (VT, Couplet, SB, Bigeminy, Trigeminy)}]
        """
        MAX_GAP_SEC = 2.0
        rhythms = []
        n = len(seq)
        i = 0
        
        while i < n:
            # 1. ТАХИКАРДИЯ / ПАРНАЯ (2+ V подряд)
            if seq[i]['group'] == 'V':
                start = i
                while i < n and seq[i]['group'] == 'V': i += 1
                count = i - start
                
                if count >= 2:
                    r_type = 'VT' if count >= 3 else 'Couplet'
                    s_s, e_s = seq[start]['sample'], seq[i-1]['sample']
                    
                    # Слияние близких ритмов
                    if rhythms and rhythms[-1]['type'] == r_type:
                        prev_end = rhythms[-1]['end_sample']
                        if (s_s - prev_end) / 360.0 <= MAX_GAP_SEC:
                            rhythms[-1]['end_sample'] = e_s
                            continue
                            
                    rhythms.append({'start_sample': s_s, 'end_sample': e_s, 'type': r_type})
                continue
                
            # 2. СТАБИЛЬНАЯ БЛОКАДА (3+ B подряд)
            elif seq[i]['group'] == 'B':
                start = i
                while i < n and seq[i]['group'] == 'B': i += 1
                count = i - start
                
                if count >= 3:
                    s_s, e_s = seq[start]['sample'], seq[i-1]['sample']
                    if rhythms and rhythms[-1]['type'] == 'SB':
                        prev_end = rhythms[-1]['end_sample']
                        if (s_s - prev_end) / 360.0 <= MAX_GAP_SEC:
                            rhythms[-1]['end_sample'] = e_s
                            continue
                    rhythms.append({'start_sample': s_s, 'end_sample': e_s, 'type': 'SB'})
                continue
                
            # 3. БИГЕМИНИЯ И ТРИГЕМИНИЯ (Окно 6)
            elif i + 5 < n:
                window = [seq[j]['group'] for j in range(i, i+6)]
                is_b = (window == ['N', 'V', 'N', 'V', 'N', 'V'])
                is_t = (window == ['N', 'N', 'V', 'N', 'N', 'V'])
                
                if is_b or is_t:
                    r_type = 'Bigeminy' if is_b else 'Trigeminy'
                    s_s = seq[i]['sample']
                    j = i + 6
                    step = 2 if is_b else 3
                    
                    while j + step - 1 < n:
                        ch = [seq[k]['group'] for k in range(j, j+step)]
                        if (is_b and ch == ['N', 'V']) or (is_t and ch == ['N', 'N', 'V']): j += step
                        else: break
                    
                    e_s = seq[j-1]['sample']
                    if rhythms and rhythms[-1]['type'] == r_type:
                        prev_end = rhythms[-1]['end_sample']
                        if (s_s - prev_end) / 360.0 <= MAX_GAP_SEC:
                            rhythms[-1]['end_sample'] = e_s
                            i = j; continue
                            
                    rhythms.append({'start_sample': s_s, 'end_sample': e_s, 'type': r_type})
                    i = j; continue
            
            i += 1
        return rhythms

    # ---------------------------------------------------------
    # 3. Отмена одиночных Блокад
    # ---------------------------------------------------------
    def _apply_isolated_blk_filter(self, raw_preds):
        for i in range(len(raw_preds)):
            if raw_preds[i]['label'] == 'B':
                is_isolated = True
                if i > 0 and raw_preds[i-1]['label'] == 'B': is_isolated = False
                if i < len(raw_preds)-1 and raw_preds[i+1]['label'] == 'B': is_isolated = False
                
                if is_isolated:
                    if raw_preds[i]['prob_pss'] > self.pss_threshold:
                        raw_preds[i]['label'] = 'V'
                    else:
                        raw_preds[i]['label'] = 'N'

    # ---------------------------------------------------------
    # 4. Детализация ПЖС + ФИЛЬТР ЛОЖНЫХ (Мертвая зона)
    # ---------------------------------------------------------
    def _apply_v_subclassification(self, raw_preds, rr_intervals):
        median_rr = np.median(rr_intervals) if len(rr_intervals) > 5 else 288
        for i, pred in enumerate(raw_preds):
            if pred['label'] != 'V': continue
                
            rr_prev = rr_intervals[i]
            rr_next = rr_intervals[i+1] if i+1 < len(rr_intervals) else median_rr
            
            # A. R-on-T
            prev_rr_sec = rr_prev / 360.0
            qt_est_samples = 0.4 * np.sqrt(prev_rr_sec) * 360
            if rr_prev < (0.85 * qt_est_samples): pred['label'] = 'r'; continue
                
            # Б. Оценка достоверности ПЖС по компенсаторной паузе
            total_rr = rr_prev + rr_next
            has_pause = total_rr >= (2.15 * median_rr)
            is_interpolated = (1.8 * median_rr) <= total_rr < (2.15 * median_rr)
            
            if has_pause:
                current_vec = pred['segment']
                if current_vec is not None and self.last_pvc_vector is not None:
                    sim = 1 - cosine(self.last_pvc_vector, current_vec)
                    if sim > 0.7: pred['label'] = 'M'
                    else: pred['label'] = 'P'
                else: pred['label'] = 'M'
                if current_vec is not None: self.last_pvc_vector = current_vec
            elif is_interpolated:
                pred['label'] = 'i'
            else:
                # В. ПРОВАЛ ФИЗИОЛОГИЧЕСКОЙ ПРОВЕРКИ (Мертвая зона)
                if pred['prob_pss'] > 0.7:
                    pred['label'] = 'M' 
                else:
                    pred['label'] = 'N' # Отмена ложного V