import os
import copy
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy.signal import medfilt
from scipy.spatial.distance import cosine

from app.core.signal import Signal

# ==========================================
# АРХИТЕКТУРЫ TCN
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
# АРХИТЕКТУРЫ MLP
# ==========================================
class FeatureAttentionGate(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.excitation(x)

class MLP_PSS(nn.Module):
    def __init__(self, input_dim=25, num_classes=2, hidden_dims=[128, 64, 32], dropout=0.25, input_dropout=0.2):
        super().__init__()
        layers = [nn.Dropout(input_dropout), FeatureAttentionGate(input_dim, reduction=4)]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class MLP_BLK(nn.Module):
    def __init__(self, input_dim=34, num_classes=2, hidden_dims=[128, 64, 32], dropout=0.25, input_dropout=0.2):
        super().__init__()
        layers = [nn.Dropout(input_dropout), FeatureAttentionGate(input_dim, reduction=4)]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ==========================================
# КЛАСС КАСКАДНОГО КЛАССИФИКАТОРА
# ==========================================
class HolterClassifier:
    def __init__(self, models_dir='models', pss_model_type='TCN', blk_model_type='TCN', device='auto'):
        if device == 'auto': self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = torch.device(device)
            
        self.pss_model_type = pss_model_type
        self.blk_model_type = blk_model_type
        
        # Загрузка всех 4 моделей
        self.models = {
            'TCN_PSS': self._load_model(TCN_PSS, os.path.join(models_dir, 'TCN_PSS.pth')),
            'TCN_BLK': self._load_model(TCN_BLK, os.path.join(models_dir, 'TCN_BLK.pth')),
            'MLP_PSS': self._load_model(MLP_PSS, os.path.join(models_dir, 'MLP_PSS.pth')),
            'MLP_BLK': self._load_model(MLP_BLK, os.path.join(models_dir, 'MLP_BLK.pth'))
        }
        
        # Загрузка скейлеров для MLP
        self.scaler_pss = joblib.load(os.path.join(models_dir, 'parametric', 'scaler_pss.pkl'))
        self.scaler_blk = joblib.load(os.path.join(models_dir, 'parametric_blk', 'scaler_blk.pkl'))
        
        # Оптимальные пороги из обучения
        self.optimal_thresholds = {
            'TCN_PSS': 0.8236, 'MLP_PSS': 0.5422,
            'TCN_BLK': 0.0663, 'MLP_BLK': 0.8245
        }
        
        self.pss_threshold = self.optimal_thresholds[f'{pss_model_type}_PSS']
        self.blk_threshold = self.optimal_thresholds[f'{blk_model_type}_BLK']
        
        self.last_pvc_vector = None
        self.last_base_feats_blk = None # Для дельт MLP_BLK

    def _load_model(self, ModelClass, path):
        model = ModelClass().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    # --- Предобработка для MLP (Точно как в dataset.py) ---
    def _extract_pss_features(self, segment, rr_norm):
        sig_obj = Signal(data=segment, fs=360)
        feats = [
            sig_obj.get_R_amplitude(segment), sig_obj.get_Q_amplitude(segment), sig_obj.get_S_amplitude(segment),
            sig_obj.get_R_over_S_ratio(segment), sig_obj.get_total_swing(segment), sig_obj.get_Q_R_ratio(segment),
            sig_obj.get_QRS_duration_50(segment), sig_obj.get_asymmetry_ratio(segment), sig_obj.get_max_upstroke(segment),
            sig_obj.get_max_downstroke(segment), sig_obj.get_mean_slope_ratio(segment), sig_obj.get_zero_crossings_qrs(segment),
            sig_obj.get_energy_ratio(segment), sig_obj.get_kurtosis_qrs(segment), sig_obj.get_entropy_qrs(segment),
            sig_obj.get_qrs_duration_norm(segment), sig_obj.get_P_energy_ratio(segment), sig_obj.get_P_polarity(segment),
            sig_obj.get_QRSd_ratio_15_50(segment), sig_obj.get_R2_presence(segment), sig_obj.get_pathologic_Q(segment),
            sig_obj.get_ST_dev(segment), sig_obj.get_activation_time(segment), sig_obj.get_P_over_R(segment), rr_norm
        ]
        return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_blk_features(self, segment, rr_norm):
        sig_obj = Signal(data=segment, fs=360)
        feats = [
            sig_obj.get_R_amplitude(segment), sig_obj.get_Q_amplitude(segment), sig_obj.get_S_amplitude(segment),
            sig_obj.get_R_over_S_ratio(segment), sig_obj.get_total_swing(segment), sig_obj.get_Q_R_ratio(segment),
            sig_obj.get_QRS_duration_50(segment), sig_obj.get_asymmetry_ratio(segment), sig_obj.get_max_upstroke(segment),
            sig_obj.get_max_downstroke(segment), sig_obj.get_mean_slope_ratio(segment), sig_obj.get_zero_crossings_qrs(segment),
            sig_obj.get_energy_ratio(segment), sig_obj.get_kurtosis_qrs(segment), sig_obj.get_entropy_qrs(segment),
            sig_obj.get_qrs_duration_norm(segment), sig_obj.get_P_energy_ratio(segment), sig_obj.get_P_polarity(segment),
            sig_obj.get_QRSd_ratio_15_50(segment), sig_obj.get_R2_presence(segment), sig_obj.get_pathologic_Q(segment),
            sig_obj.get_ST_dev(segment), sig_obj.get_activation_time(segment), sig_obj.get_P_over_R(segment), rr_norm,
            sig_obj.get_qrs_velocity_changes(segment) # 26-й признак
        ]
        return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_raw_predictions(self, signal_obj: Signal):
        if not signal_obj.annotations: return [], []
        QRS_SYMBOLS = ['N', 'L', 'R', 'B', 'V', 'E', 'r', 'A', 'a', 'J', 'j', 'S', 'F', 'e', '/', 'f']
        r_peaks = [ann['sample'] for ann in signal_obj.annotations if ann['symbol'] in QRS_SYMBOLS]
        
        rr_intervals = np.diff(r_peaks).tolist()
        rr_intervals.insert(0, int(0.8 * 360)) 
        
        raw_predictions = []
        self.last_base_feats_blk = None # Сброс контекста для новой записи
        
        for i, peak in enumerate(r_peaks):
            rr_prev = rr_intervals[i]
            segment = signal_obj.get_segment(peak, 288)
            if segment is None:
                raw_predictions.append({
                    'peak': peak, 'qrs_dur': 0, 'segment': None,
                    'prob_pss': 0, 'prob_blk': 0,
                    'prob_TCN_PSS': 0, 'prob_MLP_PSS': 0, 'prob_TCN_BLK': 0, 'prob_MLP_BLK': 0
                })
                continue
                
            rr_norm = rr_prev / 288.0
            
            # --- 1. ВХОД ДЛЯ TCN PSS ---
            pss_tcn_inp = np.zeros((2, 288))
            pss_tcn_inp[0, :] = segment
            pss_tcn_inp[1, :] = rr_norm
            pss_tcn_tensor = torch.FloatTensor(pss_tcn_inp).unsqueeze(0).to(self.device)
            
            # --- 2. ВХОД ДЛЯ MLP PSS ---
            # Стандартизируем сегмент как при обучении
            pss_sig = Signal(data=segment, fs=360)
            pss_sig.standardize()
            raw_feats_pss = self._extract_pss_features(pss_sig.resampled_data, rr_norm)
            scaled_feats_pss = self.scaler_pss.transform(raw_feats_pss.reshape(1, -1))
            pss_mlp_tensor = torch.FloatTensor(scaled_feats_pss).to(self.device)
            
            # --- 3. ВХОД ДЛЯ TCN BLK (С TTA) ---
            np.random.seed(peak)
            awgn_frag = np.random.normal(0, 1.0, 288)
            temp_sig = Signal(data=segment, fs=360)
            noisy_sig = temp_sig.add_noise(awgn_frag, snr_db_range=(25, 35))
            denoised_sig_tcn = noisy_sig.wavelet_denoise()
            denoised_sig_tcn.standardize()
            
            denoised_segment_tcn = denoised_sig_tcn.resampled_data
            qrs_dur_tcn = denoised_sig_tcn.get_qrs_duration_norm(denoised_segment_tcn) * 2.0 
            
            blk_tcn_inp = np.zeros((2, 288))
            blk_tcn_inp[0, :] = denoised_segment_tcn
            blk_tcn_inp[1, :] = qrs_dur_tcn
            blk_tcn_tensor = torch.FloatTensor(blk_tcn_inp).unsqueeze(0).to(self.device)
            
            # --- 4. ВХОД ДЛЯ MLP BLK (БЕЗ TTA, как при тестировании) ---
            denoised_sig_mlp = temp_sig.wavelet_denoise()
            denoised_sig_mlp.standardize()
            denoised_segment_mlp = denoised_sig_mlp.resampled_data
            
            raw_base_feats_blk = self._extract_blk_features(denoised_segment_mlp, rr_norm)
            
            # Расчет дельт
            if self.last_base_feats_blk is None:
                delta_feats = np.zeros(5, dtype=np.float32)
            else:
                delta_feats = np.array([
                    raw_base_feats_blk[15] - self.last_base_feats_blk[15],
                    raw_base_feats_blk[0] - self.last_base_feats_blk[0],
                    raw_base_feats_blk[3] - self.last_base_feats_blk[3],
                    raw_base_feats_blk[25] - self.last_base_feats_blk[25],
                    raw_base_feats_blk[24] - self.last_base_feats_blk[24]
                ], dtype=np.float32)
            self.last_base_feats_blk = raw_base_feats_blk.copy()
            
            combined_raw = np.concatenate([raw_base_feats_blk, delta_feats])
            scaled_combined = self.scaler_blk.transform(combined_raw.reshape(1, -1))
            
            # VIP-признаки (qrs_dur[15], velocity[25], R/S[3])
            vip_feats = raw_base_feats_blk[[15, 25, 3]]
            final_feats = np.concatenate([scaled_combined.squeeze(), vip_feats])
            blk_mlp_tensor = torch.FloatTensor(final_feats).unsqueeze(0).to(self.device)
            
            qrs_dur_mlp = raw_base_feats_blk[15] # Используем норм. длительность из чистых фичей
            
            # --- ПРОГОН ВСЕХ 4 МОДЕЛЕЙ ---
            with torch.no_grad():
                prob_tcn_pss = torch.softmax(self.models['TCN_PSS'](pss_tcn_tensor), dim=1).cpu().numpy()[0, 1]
                prob_mlp_pss = torch.softmax(self.models['MLP_PSS'](pss_mlp_tensor), dim=1).cpu().numpy()[0, 1]
                prob_tcn_blk = torch.softmax(self.models['TCN_BLK'](blk_tcn_tensor), dim=1).cpu().numpy()[0, 1]
                prob_mlp_blk = torch.softmax(self.models['MLP_BLK'](blk_mlp_tensor), dim=1).cpu().numpy()[0, 1]
                
            # Выбор активной модели для каскада
            qrs_dur = qrs_dur_tcn if self.blk_model_type == 'TCN' else qrs_dur_mlp
            prob_pss = prob_tcn_pss if self.pss_model_type == 'TCN' else prob_mlp_pss
            prob_blk = prob_tcn_blk if self.blk_model_type == 'TCN' else prob_mlp_blk
                
            raw_predictions.append({
                'peak': peak, 'qrs_dur': qrs_dur, 'segment': segment,
                'prob_pss': prob_pss, 'prob_blk': prob_blk,
                'prob_TCN_PSS': prob_tcn_pss, 'prob_MLP_PSS': prob_mlp_pss,
                'prob_TCN_BLK': prob_tcn_blk, 'prob_MLP_BLK': prob_mlp_blk
            })
            
        return raw_predictions, rr_intervals

    def analyze_signal(self, signal_obj: Signal):
        raw_predictions, rr_intervals = self._prepare_raw_predictions(signal_obj)
        if not raw_predictions: return [], []
        
        self.last_pvc_vector = None
        
        # ПОРЯДОК АЛГОРИТМОВ
        raw_preds = self._apply_base_logic(raw_predictions, rr_intervals)
        raw_preds, rhythm_annotations = self._apply_rhythm_engine(raw_preds, rr_intervals)
        self._apply_isolated_blk_filter(raw_preds)
        self._apply_v_subclassification(raw_preds, rr_intervals)
        
        final_output = [{'sample': p['peak'], 'label': p['label']} for p in raw_preds]
        return final_output, rhythm_annotations

    # ---------------------------------------------------------
    # 1. БАЗОВАЯ ЛОГИКА
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
            
            total_rr = rr_prev + rr_next
            is_interpolated = (1.7 * median_rr) <= total_rr <= (2.1 * median_rr)
            
            if pred['prob_blk'] > self.blk_threshold and not is_premature:
                pred['label'] = 'B'; continue
                
            if pred['prob_pss'] > self.pss_threshold and is_premature:
                pred['label'] = 'V'; continue
                
            if pred['prob_pss'] > 0.8 and is_interpolated:
                pred['label'] = 'V'; continue
                
            thr = 2.20 * median_rr if is_post_pvc else 1.80 * median_rr
            if is_delayed and rr_prev > thr: 
                if not is_narrow:
                    pred['label'] = 'E'; continue
                    
            if is_narrow and is_premature and rr_prev < (0.70 * median_rr): 
                pred['label'] = 'A'; continue
                
            pred['label'] = 'N'
            
        return raw_preds

    # ---------------------------------------------------------
    # 2. ДВИЖОК РИТМОВ
    # ---------------------------------------------------------
    def _apply_rhythm_engine(self, raw_preds, rr_intervals):
        seq = [{'sample': p['peak'], 'group': self._map_group(p['label'])} for p in raw_preds]
        detected_rhythms = self.detect_rhythms(seq)
        rhythm_annotations = []
        for r in detected_rhythms:
            rhythm_annotations.append({
                'start_sample': r['start_sample'],
                'end_sample': r['end_sample'],
                'rhythm': f"({r['type']}"
            })
        return raw_preds, rhythm_annotations

    @staticmethod
    def _map_group(label):
        if label in ['V', 'F', 'r', 'i', 'M', 'P']: return 'V'
        if label in ['B', 'L', 'R']: return 'B'
        if label in ['A', 'a']: return 'A'
        if label == 'N': return 'N'
        return 'O'

    @staticmethod
    def detect_rhythms(seq):
        MAX_GAP_SEC = 2.0
        rhythms = []
        n = len(seq)
        i = 0
        while i < n:
            if seq[i]['group'] == 'V':
                start = i
                while i < n and seq[i]['group'] == 'V': i += 1
                count = i - start
                if count >= 2:
                    r_type = 'VT' if count >= 3 else 'Couplet'
                    s_s, e_s = seq[start]['sample'], seq[i-1]['sample']
                    if rhythms and rhythms[-1]['type'] == r_type:
                        prev_end = rhythms[-1]['end_sample']
                        if (s_s - prev_end) / 360.0 <= MAX_GAP_SEC:
                            rhythms[-1]['end_sample'] = e_s
                            continue
                    rhythms.append({'start_sample': s_s, 'end_sample': e_s, 'type': r_type})
                continue
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
    # 4. Детализация ПЖС
    # ---------------------------------------------------------
    def _apply_v_subclassification(self, raw_preds, rr_intervals):
        median_rr = np.median(rr_intervals) if len(rr_intervals) > 5 else 288
        for i, pred in enumerate(raw_preds):
            if pred['label'] != 'V': continue
            rr_prev = rr_intervals[i]
            rr_next = rr_intervals[i+1] if i+1 < len(rr_intervals) else median_rr
            
            prev_rr_sec = rr_prev / 360.0
            qt_est_samples = 0.4 * np.sqrt(prev_rr_sec) * 360
            if rr_prev < (0.85 * qt_est_samples): pred['label'] = 'r'; continue
                
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
                if pred['prob_pss'] > 0.7:
                    pred['label'] = 'M' 
                else:
                    pred['label'] = 'N'