import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score, 
                             roc_curve, accuracy_score)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
from app.core.signal import Signal

CLASS_NAMES = ['Normal', 'Blockade']

class Trainer:
    def __init__(self, channel_name='II_BLK', models_dir='models', results_dir='results'):
        self.channel_name = channel_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = os.path.join(models_dir, channel_name)
        self.results_dir = os.path.join(results_dir, channel_name)
        os.makedirs(self.models_dir, exist_ok=True); os.makedirs(self.results_dir, exist_ok=True)

    def train(self, model, X_train, y_train, X_val, y_val, epochs=120, batch_size=64, lr=0.0005):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # ВАШЕ УЛУЧШЕНИЕ
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
        best_loss, patience_counter, history = float('inf'), 20, {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
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
            history['loss'].append(t_loss/len(loader)); history['acc'].append(t_corr/t_tot); history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
            scheduler.step(v_loss)
            if v_loss < best_loss: 
                best_loss = v_loss; torch.save(model.state_dict(), os.path.join(self.models_dir, f'best_tcn_{self.channel_name.lower()}.pth')); patience_counter = 20
            else: patience_counter -= 1
            if patience_counter == 0: print(f"Early stopping на эпохе {epoch+1}"); break
        return model, history
    
    def evaluate_and_plot(self, model, X_test, y_test, sym_test, history, builder):
        model_path = os.path.join(self.models_dir, f'best_tcn_{self.channel_name.lower()}.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # --- ОЦЕНКА НА ЧИСТЫХ ДАННЫХ (С ДЕЛИКАТНОЙ ПОРЧЕЙ - TTA) ---
        print("Оценка на чистых данных (применение TTA для активации вейвлета)...")
        X_clean_tta = torch.zeros_like(X_test)
        
        for i in range(X_test.shape[0]):
            segment = X_test[i, 0, :].numpy()
            qrs_dur = X_test[i, 1, 0].item()
            
            temp_sig = Signal(data=segment, fs=360)
            awgn_frag = np.random.normal(0, 1.0, 288)
            noisy_sig = temp_sig.add_noise(awgn_frag, snr_db_range=(25, 35))
            denoised_sig = noisy_sig.wavelet_denoise()
            denoised_sig.standardize()
            
            X_clean_tta[i, 0, :] = torch.FloatTensor(denoised_sig.resampled_data)
            X_clean_tta[i, 1, :] = qrs_dur
            del temp_sig, noisy_sig, denoised_sig

        with torch.no_grad():
            logits_clean = model(X_clean_tta.to(self.device))
            y_pred_clean = np.argmax(logits_clean.cpu().numpy(), axis=1)
            y_prob_clean = torch.softmax(logits_clean, dim=1).cpu().numpy()[:, 1]
            
        metrics_clean = self._calculate_metrics(y_test, y_pred_clean, y_prob_clean)
        self._plot_history(history)
        self._plot_binary_cm(y_test, y_pred_clean, title="Clean Test Data", filename="cm_clean.png")
        
        # ВОЗВРАЩАЕМ ROC ДЛЯ ЧИСТЫХ ДАННЫХ
        self._plot_roc(y_test, y_prob_clean, title="ROC Curve: Clean Data", filename="roc_clean.png")

        # --- ОЦЕНКА НА ЗАШУМЛЕННЫХ ДАННЫХ ---
        print("\nГенерация зашумленной тестовой выборки...")
        X_noisy, y_noisy, sym_noisy = builder.generate_noisy_test_set(X_test, y_test, sym_test)
        
        with torch.no_grad():
            logits_noisy = model(X_noisy.to(self.device))
            
            # Если используете сдвиг порога (например, 0.35), раскомментируйте строку ниже:
            # y_pred_noisy = (torch.softmax(logits_noisy, dim=1).cpu().numpy()[:, 1] > 0.35).astype(int)
            
            # Если не используете сдвиг порога, оставьте эту строку:
            y_pred_noisy = np.argmax(logits_noisy.cpu().numpy(), axis=1)
            
            # Для ROC-кривой всегда нужны чистые вероятности, независимо от порога!
            y_prob_noisy = torch.softmax(logits_noisy, dim=1).cpu().numpy()[:, 1]
            
        metrics_noisy = self._calculate_metrics(y_noisy, y_pred_noisy, y_prob_noisy)
        self._plot_binary_cm(y_noisy, y_pred_noisy, title="Noisy Test Data", filename="cm_noisy.png")
        
        # ВОЗВРАЩАЕМ ROC ДЛЯ ЗАШУМЛЕННЫХ ДАННЫХ
        self._plot_roc(y_noisy, y_prob_noisy, title="ROC Curve: Noisy Data", filename="roc_noisy.png")
        
        self._plot_comparison(metrics_clean, metrics_noisy)
        
        print("\n" + "="*60)
        print(f"ФИНАЛЬНЫЕ МЕТРИКИ (Канал: {self.channel_name})")
        print("="*60)
        print(f"{'Метрика':<15} | {'Чистые данные':<15} | {'Зашумленные данные':<15}")
        print("-"*60)
        for key in metrics_clean.keys():
            print(f"{key:<15} | {metrics_clean[key]:<15.4f} | {metrics_noisy[key]:<15.4f}")
        print("="*60)

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        return {'Accuracy': acc, 'Specificity': specificity, 'F1-Score': f1, 'AUC': auc}

    def _plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history['loss'], label='Train'); ax1.plot(history['val_loss'], label='Val'); ax1.set_title("Loss"); ax1.legend()
        ax2.plot(history['acc'], label='Train'); ax2.plot(history['val_acc'], label='Val'); ax2.set_title("Accuracy"); ax2.legend()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=150); plt.close()

    def _plot_binary_cm(self, y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        # ИСПРАВЛЕНО: Подписи Normal и Blockade
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Blockade'], yticklabels=['Normal', 'Blockade'])
        plt.title(title)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150); plt.close()

    def _plot_roc(self, y_true, y_prob, title, filename):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150); plt.close()

    def _plot_comparison(self, metrics_clean, metrics_noisy):
        labels = list(metrics_clean.keys())
        clean_vals = list(metrics_clean.values())
        noisy_vals = list(metrics_noisy.values())
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, clean_vals, width, label='Clean Data', color='royalblue')
        rects2 = ax.bar(x + width/2, noisy_vals, width, label='Noisy Data (-12 to 6 dB)', color='salmon')
        
        ax.set_ylabel('Score')
        ax.set_title(f'Classifier Robustness Comparison (Channel: {self.channel_name})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')
        
        # Добавляем значения над столбцами
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
                            
        autolabel(rects1)
        autolabel(rects2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comparison_metrics.png'), dpi=150); plt.close()