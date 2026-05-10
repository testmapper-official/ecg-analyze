import os
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.training.blk.dataset import DatasetBuilder
from app.core.signal import Signal

TARGET_FS = 360
SEGMENT_SAMPLES = 288

np.random.seed(42)

def normalize_rms(sig):
    rms = np.sqrt(np.mean(sig**2))
    return sig / rms if rms > 0 else sig

def get_noise_fragments(builder, length):
    noises = {}
    if 'em' in builder.noise_data:
        base = builder.noise_data['em']
        start = np.random.randint(0, len(base) - length)
        noises['EM (nstdb)'] = normalize_rms(base[start:start+length])
    if 'ma' in builder.noise_data:
        base = builder.noise_data['ma']
        start = np.random.randint(0, len(base) - length)
        noises['MA (nstdb)'] = normalize_rms(base[start:start+length])
        
    t = np.arange(length) / TARGET_FS
    noises['AWGN'] = np.random.normal(0, 1.0, length)
    noises['Powerline (50Hz)'] = np.sin(2 * np.pi * 50.0 * t)
    
    pop = np.zeros(length)
    pop_idx = length // 2
    pop[pop_idx] = 5.0
    pop[pop_idx+1:] = 5.0 * np.exp(-0.03 * np.arange(length - pop_idx - 1))
    noises['Electrode Pop'] = normalize_rms(pop)
    return noises

def plot_clean_signals(segments_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for class_name, segments in segments_dict.items():
        if not segments: continue
        fig, axes = plt.subplots(len(segments), 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f'Clean Signals: Class {class_name}', fontsize=14)
        for i, seg in enumerate(segments):
            axes[i].plot(seg, color='black', linewidth=0.8)
            axes[i].set_ylabel(f'#{i+1}', rotation=0, labelpad=15)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'clean_{class_name}.png'), dpi=150)
        plt.close()

def plot_noise_examples(noises, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(len(noises), 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Applied Noise Fragments', fontsize=14)
    for i, (n_name, n_data) in enumerate(noises.items()):
        axes[i].plot(n_data, color='red', linewidth=0.8)
        axes[i].set_ylabel(n_name, rotation=0, labelpad=50)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_fragments.png'), dpi=150)
    plt.close()

def plot_snr_variations(segments_dict, noises, save_dir):
    """Наложение 1 шума (EM) с разными SNR на 5 экземпляров"""
    os.makedirs(save_dir, exist_ok=True)
    noise_name = 'EM (nstdb)' if 'EM (nstdb)' in noises else list(noises.keys())[0]
    noise_data = noises[noise_name]
    snr_list = [6, 0, -12]
    
    for class_name, segments in segments_dict.items():
        if not segments: continue
        for i, seg in enumerate(segments):
            fig, axes = plt.subplots(len(snr_list) + 1, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'Class {class_name} - Instance #{i+1} (SNR Variations)', fontsize=14)
            
            # Чистый + дыхание
            base_sig = Signal(data=seg.copy(), fs=TARGET_FS).respiratory_modulation(max_depth=0.3)
            axes[0].plot(base_sig.resampled_data, color='green', linewidth=0.8)
            axes[0].set_ylabel('Clean +\nResp Mod', rotation=0, labelpad=40)
            axes[0].grid(True, alpha=0.3)
            
            for j, snr_db in enumerate(snr_list):
                sig_to_noise = Signal(data=base_sig.resampled_data.copy(), fs=TARGET_FS)
                noisy_sig = sig_to_noise.add_noise(noise_data, snr_db_range=(snr_db, snr_db))
                axes[j+1].plot(noisy_sig.resampled_data, color='blue', linewidth=0.8)
                axes[j+1].set_ylabel(f'SNR {snr_db} dB', rotation=0, labelpad=40)
                axes[j+1].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Samples')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'snr_{class_name}_{i+1}.png'), dpi=150)
            plt.close()

def plot_duo_combinations(segments_dict, noises, save_dir):
    """1 экземпляр каждой классификации + ВСЕ комбинации из 2 шумов"""
    os.makedirs(save_dir, exist_ok=True)
    noise_names = list(noises.keys())
    # Генерируем все уникальные пары (для 5 шумов будет 10 комбинаций)
    pairs = list(itertools.combinations(noise_names, 2))
    
    for class_name, segments in segments_dict.items():
        if not segments: continue
        
        # БЕРЕМ ТОЛЬКО 1 ЭКЗЕМПЛЯР
        seg = segments[0]
        
        # Базовый сигнал с модуляцией дыханием
        base_sig = Signal(data=seg.copy(), fs=TARGET_FS).respiratory_modulation(max_depth=0.3)
        
        fig, axes = plt.subplots(len(pairs) + 1, 1, figsize=(12, 2.5 * (len(pairs) + 1)))
        fig.suptitle(f'Class {class_name} - 1 Instance: All Duo Noise Combinations (0 dB)', fontsize=14)
        
        # Чистый + дыхание
        axes[0].plot(base_sig.resampled_data, color='green', linewidth=0.8)
        axes[0].set_ylabel('Clean +\nResp Mod', rotation=0, labelpad=50)
        axes[0].grid(True, alpha=0.3)
        
        # Комбинации шумов
        for i, (n1_name, n2_name) in enumerate(pairs):
            mixed_noise = normalize_rms(0.5 * noises[n1_name] + 0.5 * noises[n2_name])
            noisy_sig = Signal(data=base_sig.resampled_data.copy(), fs=TARGET_FS).add_noise(mixed_noise, snr_db_range=(0, 0))
            
            axes[i+1].plot(noisy_sig.resampled_data, color='blue', linewidth=0.8)
            ylabel = f"{n1_name}\n+\n{n2_name}"
            axes[i+1].set_ylabel(ylabel, rotation=0, labelpad=60, fontsize=9)
            axes[i+1].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'duo_combinations_{class_name}.png'), dpi=150)
        plt.close()

def plot_full_noise(segments_dict, noises, save_dir):
    """Наложение всех шумов сразу на 5 экземпляров"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Миксуем все шумы
    mixed = np.zeros(SEGMENT_SAMPLES)
    for n_data in noises.values():
        mixed += n_data
    mixed = normalize_rms(mixed)
    
    for class_name, segments in segments_dict.items():
        if not segments: continue
        for i, seg in enumerate(segments):
            fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
            fig.suptitle(f'Class {class_name} - Instance #{i+1}: Full Noise Mix (0 dB)', fontsize=14)
            
            base_sig = Signal(data=seg.copy(), fs=TARGET_FS).respiratory_modulation(max_depth=0.3)
            axes[0].plot(base_sig.resampled_data, color='green', linewidth=0.8)
            axes[0].set_ylabel('Clean +\nResp Mod', rotation=0, labelpad=40)
            axes[0].grid(True, alpha=0.3)
            
            noisy_sig = Signal(data=base_sig.resampled_data.copy(), fs=TARGET_FS).add_noise(mixed, snr_db_range=(0, 0))
            axes[1].plot(noisy_sig.resampled_data, color='blue', linewidth=0.8)
            axes[1].set_ylabel('All Noises\nMixed', rotation=0, labelpad=40)
            axes[1].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Samples')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'full_mix_{class_name}_{i+1}.png'), dpi=150)
            plt.close()

def main():
    DB_ROOT = 'DB'
    BASE_SAVE_DIR = 'results/signals'
    
    print("Загрузка данных...")
    builder = DatasetBuilder(db_root=DB_ROOT)
    patients_data = builder._collect_data()
    
    if not patients_data:
        print("ОШИБКА: Данные не собраны.")
        return

    segments_dict = {'N': [], 'V': [], 'B': []}
    patient_ids = list(patients_data.keys())
    np.random.shuffle(patient_ids)
    
    for pid in patient_ids:
        for item in patients_data[pid]:
            sym = item['original_symbol']
            if sym == 'N' and len(segments_dict['N']) < 5:
                segments_dict['N'].append(item['segment'])
            elif sym == 'V' and len(segments_dict['V']) < 5:
                segments_dict['V'].append(item['segment'])
            elif sym in ['L', 'R'] and len(segments_dict['B']) < 5:
                segments_dict['B'].append(item['segment'])
        if all(len(v) >= 5 for v in segments_dict.values()): break
            
    print(f"Собрано экземпляров: N={len(segments_dict['N'])}, V={len(segments_dict['V'])}, B={len(segments_dict['B'])}")
    noises = get_noise_fragments(builder, SEGMENT_SAMPLES)
    
    print("Генерация визуализаций...")
    plot_clean_signals(segments_dict, save_dir=os.path.join(BASE_SAVE_DIR, 'clean'))
    plot_noise_examples(noises, save_dir=os.path.join(BASE_SAVE_DIR, 'noise'))
    
    # 1. Вариации SNR (5 экземпляров)
    plot_snr_variations(segments_dict, noises, save_dir=os.path.join(BASE_SAVE_DIR, 'noise'))
    
    # 2. Комбинации 2 шумов (строго 1 экземпляр на класс, все пары)
    plot_duo_combinations(segments_dict, noises, save_dir=os.path.join(BASE_SAVE_DIR, 'noise_duo'))
    
    # 3. Полная смесь всех шумов (5 экземпляров)
    plot_full_noise(segments_dict, noises, save_dir=os.path.join(BASE_SAVE_DIR, 'noice_full'))
    
    print(f"Готово! Все графики сохранены в папку {BASE_SAVE_DIR}/")

if __name__ == "__main__":
    main()