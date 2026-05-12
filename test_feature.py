import numpy as np
from app.core.signal import Signal

def generate_test_complex(qrs_type='normal'):
    """
    Создаёт синтетический сегмент 288 отсчётов, 360 Гц.
    qrs_type: 'normal' – узкий нормальный комплекс,
              'wide'   – широкий эктопический.
    Во всех случаях R-пик в центре (144), стандартизация не применяется.
    """
    fs = 360
    t = np.arange(288) / fs
    seg = np.zeros(288)
    center = 144

    if qrs_type == 'normal':
        # Нормальный комплекс: Q=-0.2, R=1.0, S=-0.3 на протяжении ~20 отсчётов
        # Подъём от Q к R (5 отсчётов), спуск от R к S (6 отсчётов)
        seg[center-2:center+3] = 1.0          # R
        seg[center-7:center-2] = -0.2          # Q
        seg[center+3:center+9] = -0.3          # S
        # Добавим немного ВЧ шума
        noise = 0.02 * np.random.randn(288)
        seg += noise
    else:
        # Широкий желудочковый: низковольтный, зазубренный, длительность ~40 отсчётов
        seg[center-20:center+20] = 0.6
        seg[center-5:center+5] = 1.0
        seg[center-15:center-10] = -0.4
        seg[center+10:center+18] = -0.5
        noise = 0.05 * np.random.randn(288)
        seg += noise

    return seg

def main():
    # Создаём "пустой" объект Signal для вызова методов
    s = Signal(data=np.zeros(288), fs=360)

    print("=== Нормальный комплекс ===")
    seg_normal = generate_test_complex('normal')
    # Стандартизируем, чтобы методы работали как при обучении
    seg_normal = (seg_normal - np.mean(seg_normal)) / np.std(seg_normal)

    features = {}
    features['R_amp'] = s.get_R_amplitude(seg_normal)
    features['Q_amp'] = s.get_Q_amplitude(seg_normal)
    features['S_amp'] = s.get_S_amplitude(seg_normal)
    features['R/S'] = s.get_R_over_S_ratio(seg_normal)
    features['swing'] = s.get_total_swing(seg_normal)
    features['Q/R'] = s.get_Q_R_ratio(seg_normal)
    features['QRSd_50'] = s.get_QRS_duration_50(seg_normal)
    features['asymmetry'] = s.get_asymmetry_ratio(seg_normal)
    features['upstroke'] = s.get_max_upstroke(seg_normal)
    features['downstroke'] = s.get_max_downstroke(seg_normal)
    features['slope_ratio'] = s.get_mean_slope_ratio(seg_normal)
    features['zcr_qrs'] = s.get_zero_crossings_qrs(seg_normal)
    features['energy_ratio'] = s.get_energy_ratio(seg_normal)
    features['kurtosis'] = s.get_kurtosis_qrs(seg_normal)
    features['entropy'] = s.get_entropy_qrs(seg_normal)
    features['QRSd_norm'] = s.get_qrs_duration_norm(seg_normal)  # уже существующий

    for k, v in features.items():
        print(f"{k:<15}: {v:.4f}")

    print("\n=== Широкий эктопический комплекс ===")
    seg_wide = generate_test_complex('wide')
    seg_wide = (seg_wide - np.mean(seg_wide)) / np.std(seg_wide)

    features_w = {}
    features_w['R_amp'] = s.get_R_amplitude(seg_wide)
    features_w['Q_amp'] = s.get_Q_amplitude(seg_wide)
    features_w['S_amp'] = s.get_S_amplitude(seg_wide)
    features_w['R/S'] = s.get_R_over_S_ratio(seg_wide)
    features_w['swing'] = s.get_total_swing(seg_wide)
    features_w['Q/R'] = s.get_Q_R_ratio(seg_wide)
    features_w['QRSd_50'] = s.get_QRS_duration_50(seg_wide)
    features_w['asymmetry'] = s.get_asymmetry_ratio(seg_wide)
    features_w['upstroke'] = s.get_max_upstroke(seg_wide)
    features_w['downstroke'] = s.get_max_downstroke(seg_wide)
    features_w['slope_ratio'] = s.get_mean_slope_ratio(seg_wide)
    features_w['zcr_qrs'] = s.get_zero_crossings_qrs(seg_wide)
    features_w['energy_ratio'] = s.get_energy_ratio(seg_wide)
    features_w['kurtosis'] = s.get_kurtosis_qrs(seg_wide)
    features_w['entropy'] = s.get_entropy_qrs(seg_wide)
    features_w['QRSd_norm'] = s.get_qrs_duration_norm(seg_wide)

    for k, v in features_w.items():
        print(f"{k:<15}: {v:.4f}")

if __name__ == '__main__':
    main()