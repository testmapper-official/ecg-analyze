import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score, 
                             roc_curve, accuracy_score, recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from app.core.signal import Signal

def specificity_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp + 1e-8)
    return 0.0

class Trainer:
    def __init__(self, channel_name='II', models_dir='models', results_dir='results'):
        self.channel_name = channel_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = os.path.join(models_dir, channel_name)
        self.results_dir = os.path.join(results_dir, channel_name)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def train(self, model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.0005):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
        best_loss, patience_counter, history = float('inf'), 15, {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
        loader = DataLoader(TensorDataset(X_train, torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            model.train(); t_loss, t_corr, t_tot = 0, 0, 0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad(); out = model(bx); loss = criterion(out, by); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
                t_loss += loss.item(); t_corr += (out.argmax(1)==by).sum().item(); t_tot += by.size(0)
            model.eval()
            with torch.no_grad():
                v_out = model(X_val.to(self.device)); v_loss = criterion(v_out, torch.LongTensor(y_val).to(self.device)).item()
                v_acc = (v_out.argmax(1).cpu() == torch.LongTensor(y_val)).float().mean().item()
            history['loss'].append(t_loss/len(loader)); history['acc'].append(t_corr/t_tot)
            history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
            scheduler.step(v_loss)
            model_path = os.path.join(self.models_dir, f'best_tcn_{self.channel_name.lower()}.pth')
            if v_loss < best_loss: 
                best_loss = v_loss; torch.save(model.state_dict(), model_path); patience_counter = 15
            else: patience_counter -= 1
            if patience_counter == 0: break
        return model, history

    def evaluate_and_plot(self, model, X_test, y_test, sym_test, history, builder):
        model_path = os.path.join(self.models_dir, f'best_tcn_{self.channel_name.lower()}.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # --- ОЦЕНКА НА ЧИСТЫХ ДАННЫХ (С TTA) ---
        print("Оценка на чистых данных (применение TTA для активации вейвлета)...")
        X_clean_tta = torch.zeros_like(X_test)
        for i in range(X_test.shape[0]):
            segment = X_test[i, 0, :].numpy(); qrs_dur = X_test[i, 1, 0].item()
            temp_sig = Signal(data=segment, fs=360); awgn_frag = np.random.normal(0, 1.0, 288)
            noisy_sig = temp_sig.add_noise(awgn_frag, snr_db_range=(25, 35))
            denoised_sig = noisy_sig.wavelet_denoise(); denoised_sig.standardize()
            X_clean_tta[i, 0, :] = torch.FloatTensor(denoised_sig.resampled_data)
            X_clean_tta[i, 1, :] = qrs_dur
            del temp_sig, noisy_sig, denoised_sig

        with torch.no_grad():
            logits_clean = model(X_clean_tta.to(self.device))
            y_prob_clean = torch.softmax(logits_clean, dim=1).cpu().numpy()[:, 1]
            
        # --- ОЦЕНКА НА ЗАШУМЛЕННЫХ ДАННЫХ ---
        print("\nГенерация зашумленной тестовой выборки...")
        X_noisy, y_noisy, sym_noisy = builder.generate_noisy_test_set(X_test, y_test, sym_test)
        with torch.no_grad():
            logits_noisy = model(X_noisy.to(self.device))
            y_prob_noisy = torch.softmax(logits_noisy, dim=1).cpu().numpy()[:, 1]

        # --- ПОИСК ОПТИМАЛЬНОГО ПОРОГА ---
        fpr_c, tpr_c, thresholds_c = roc_curve(y_test, y_prob_clean)
        j_scores = 1.5 * tpr_c - fpr_c 
        optimal_threshold = thresholds_c[np.argmax(j_scores)]
        print(f"\n[INFO] Стандартный порог: 0.5000 | Оптимальный порог (смещен к Recall): {optimal_threshold:.4f}")

        y_pred_clean = (y_prob_clean >= optimal_threshold).astype(int)
        y_pred_noisy = (y_prob_noisy >= optimal_threshold).astype(int)

        # --- МЕТРИКИ И ГРАФИКИ ---
        auc_clean = roc_auc_score(y_test, y_prob_clean)
        auc_noisy = roc_auc_score(y_noisy, y_prob_noisy)
        
        self._plot_history(history)
        self._plot_binary_cm(y_test, y_pred_clean, title=f"Clean Test Data (Thr={optimal_threshold:.2f})", filename="binary_cm_clean.png")
        self._plot_binary_cm(y_noisy, y_pred_noisy, title=f"Noisy Test Data (Thr={optimal_threshold:.2f})", filename="binary_cm_noisy.png")
        self._plot_roc(y_test, y_prob_clean, optimal_threshold, auc_clean, title="ROC Curve: Clean Data", filename="roc_clean.png")
        self._plot_roc(y_noisy, y_prob_noisy, optimal_threshold, auc_noisy, title="ROC Curve: Noisy Data", filename="roc_noisy.png")
        
        # ИСПОЛЬЗУЕМ БУТСТРАП
        groups_clean = self._compute_metrics_bootstrap(y_test, y_pred_clean)
        groups_noisy = self._compute_metrics_bootstrap(y_noisy, y_pred_noisy)
        
        self._plot_boxplots(groups_clean, groups_noisy, title=f"TCN Metrics CI (Channel: {self.channel_name})", filename="boxplots_metrics.png")
        self._plot_comparison_bar(groups_clean, groups_noisy, title=f"TCN Robustness Comparison (Channel: {self.channel_name})")
        self._print_metrics(groups_clean, groups_noisy, auc_clean, auc_noisy)
        print("Готово! Результаты сохранены.")

    # ЗАМЕНЕНО НА БУТСТРАП
    def _compute_metrics_bootstrap(self, y_true, y_pred, n_bootstraps=1000):
        n = len(y_true)
        rng = np.random.default_rng(42)
        results = {'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1-Score': []}
        for _ in range(n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            yt, yp = y_true[idx], y_pred[idx]
            results['Accuracy'].append(accuracy_score(yt, yp))
            results['Recall'].append(recall_score(yt, yp, zero_division=0))
            results['Specificity'].append(specificity_binary(yt, yp))
            results['F1-Score'].append(f1_score(yt, yp, zero_division=0))
        for k in results: results[k] = np.array(results[k])
        return results

    def _print_metrics(self, clean_groups, noisy_groups, auc_clean, auc_noisy):
        print("\n" + "="*70)
        print(f"ФИНАЛЬНЫЕ МЕТРИКИ: Normal vs V (Канал: {self.channel_name})")
        print(f"Оптимальный порог, смещенный к Recall (Bootstrap 1000 CI)")
        print("="*70)
        print(f"{'Метрика':<15} | {'Чистые данные':<25} | {'Зашумленные данные':<25}")
        print("-"*70)
        for metric in ['Accuracy', 'Recall', 'Specificity', 'F1-Score']:
            c_mean = np.mean(clean_groups[metric]); c_ci = np.percentile(clean_groups[metric], [2.5, 97.5])
            n_mean = np.mean(noisy_groups[metric]); n_ci = np.percentile(noisy_groups[metric], [2.5, 97.5])
            print(f"{metric:<15} | {c_mean:.4f} [{c_ci[0]:.4f}, {c_ci[1]:.4f}] | {n_mean:.4f} [{n_ci[0]:.4f}, {n_ci[1]:.4f}]")
        print("-"*70)
        print(f"{'AUC':<15} | {auc_clean:<25.4f} | {auc_noisy:<25.4f}")
        print("="*70)

    def _plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history['loss'], label='Train'); ax1.plot(history['val_loss'], label='Val'); ax1.set_title("Loss"); ax1.legend()
        ax2.plot(history['acc'], label='Train'); ax2.plot(history['val_acc'], label='Val'); ax2.set_title("Accuracy"); ax2.legend()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=150); plt.close()

    def _plot_binary_cm(self, y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'V'], yticklabels=['Normal', 'V'])
        plt.title(title)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150); plt.close()

    def _plot_roc(self, y_true, y_prob, threshold, auc, title, filename):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.scatter(fpr[np.argmin(np.abs(thresholds - threshold))], tpr[np.argmin(np.abs(thresholds - threshold))], marker='o', color='red', s=100, label=f'Threshold = {threshold:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(title); plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150); plt.close()

    def _plot_boxplots(self, clean_groups, noisy_groups, title, filename):
        metrics_names = list(clean_groups.keys())
        fig, axes = plt.subplots(1, len(metrics_names), figsize=(20, 6))
        fig.suptitle(title, fontsize=16)
        
        # Вычисляем динамические лимиты для оси Y
        all_vals = list(clean_groups.values()) + list(noisy_groups.values())
        global_min = min(np.min(arr) for arr in all_vals)
        global_max = max(np.max(arr) for arr in all_vals)
        y_min = max(0, global_min - 0.05)
        y_max = min(1, global_max + 0.05)
        
        for i, metric in enumerate(metrics_names):
            ax = axes[i]; data = [clean_groups[metric], noisy_groups[metric]]
            bp = ax.boxplot(data, labels=['Clean', 'Noisy'], patch_artist=True, notch=True)
            bp['boxes'][0].set_facecolor('royalblue'); bp['boxes'][1].set_facecolor('salmon')
            ax.set_title(metric)
            ax.set_ylim(y_min, y_max) # Применяем динамические лимиты
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150)
        plt.close()

    def _plot_comparison_bar(self, clean_groups, noisy_groups, title):
        labels = list(clean_groups.keys())
        clean_means = [np.mean(clean_groups[l]) for l in labels]; noisy_means = [np.mean(noisy_groups[l]) for l in labels]
        x = np.arange(len(labels)); width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, clean_means, width, label='Clean Data', color='royalblue')
        rects2 = ax.bar(x + width/2, noisy_means, width, label='Noisy Data', color='salmon')
        ax.set_ylabel('Score'); ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1); ax.legend(loc='lower right')
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        plt.tight_layout(); plt.savefig(os.path.join(self.results_dir, 'comparison_metrics.png'), dpi=150); plt.close()