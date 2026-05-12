import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, recall_score, 
                             confusion_matrix, roc_curve)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight, label_smoothing=0.05)
        pt = torch.exp(-ce_loss); return (((1 - pt) ** self.gamma) * ce_loss).mean()

def specificity_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel(); return tn / (tn + fp + 1e-8)
    return 0.0

class ParametricTrainer:
    def __init__(self, models_dir='models/parametric', results_dir='results/parametric'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = models_dir; self.results_dir = results_dir
        os.makedirs(models_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    def train(self, model, X_train, y_train, X_val, y_val, epochs=150, batch_size=128, lr=0.001, noise_std=0.15):
        model = model.to(self.device)
        class_weights = torch.tensor([1.0, 1.5]).to(self.device)
        criterion = FocalLoss(gamma=1.5, weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        best_loss, patience_counter, history = float('inf'), 15, {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        loader = DataLoader(TensorDataset(X_train, y_train.long()), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            model.train(); t_loss, t_corr, t_tot = 0, 0, 0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                bx_noisy = bx + (torch.randn_like(bx) * noise_std * bx.std()) if noise_std > 0 else bx
                optimizer.zero_grad(); out = model(bx_noisy); loss = criterion(out, by); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                t_loss += loss.item(); t_corr += (out.argmax(1) == by).sum().item(); t_tot += by.size(0)
            model.eval()
            with torch.no_grad():
                v_out = model(X_val.to(self.device)); v_loss = criterion(v_out, y_val.long().to(self.device)).item()
                v_acc = (v_out.argmax(1).cpu() == y_val).float().mean().item()
            history['train_loss'].append(t_loss/len(loader)); history['train_acc'].append(t_corr/t_tot)
            history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
            scheduler.step(v_loss)
            if v_loss < best_loss: best_loss = v_loss; torch.save(model.state_dict(), os.path.join(self.models_dir, 'best_mlp.pth')); patience_counter = 15
            else: patience_counter -= 1;
            if patience_counter == 0: break
        model.load_state_dict(torch.load(os.path.join(self.models_dir, 'best_mlp.pth')))
        self._plot_training_curves(history); return model
    
    def _plot_training_curves(self, history):
        epochs = range(1, len(history['train_loss']) + 1); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
        ax1.plot(epochs, history['train_loss'], label='Train Loss'); ax1.plot(epochs, history['val_loss'], label='Val Loss'); ax1.set_title('Loss'); ax1.legend()
        ax2.plot(epochs, history['train_acc'], label='Train Acc'); ax2.plot(epochs, history['val_acc'], label='Val Acc'); ax2.set_title('Accuracy'); ax2.legend()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=150); plt.close()

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

    def evaluate_and_plot(self, model, X_test, y_test, sym_test, X_noisy, y_noisy, sym_noisy):
        model.eval()
        with torch.no_grad():
            prob_clean = torch.softmax(model(X_test.to(self.device)), dim=1).cpu().numpy()
            prob_noisy = torch.softmax(model(X_noisy.to(self.device)), dim=1).cpu().numpy()
        y_true_c, y_true_n = y_test.numpy(), y_noisy.numpy()
        prob_c_v, prob_n_v = prob_clean[:, 1], prob_noisy[:, 1]

        fpr_c, tpr_c, thresholds_c = roc_curve(y_true_c, prob_c_v)
        j_scores = 1.5 * tpr_c - fpr_c 
        optimal_threshold = thresholds_c[np.argmax(j_scores)]
        print(f"\n[INFO] Стандартный порог: 0.5000 | Оптимальный порог (смещен к Recall): {optimal_threshold:.4f}")

        y_pred_c_opt, y_pred_n_opt = (prob_c_v >= optimal_threshold).astype(int), (prob_n_v >= optimal_threshold).astype(int)

        self._plot_cm(y_true_c, y_pred_c_opt, title=f"Binary CM: Clean (Thr={optimal_threshold:.2f})", filename="binary_cm_clean.png")
        self._plot_roc(y_true_c, prob_c_v, optimal_threshold, title="ROC Curve: Clean Data", filename="roc_clean.png")
        groups_clean = self._compute_metrics_bootstrap(y_true_c, y_pred_c_opt)
        
        self._plot_cm(y_true_n, y_pred_n_opt, title=f"Binary CM: Noisy (Thr={optimal_threshold:.2f})", filename="binary_cm_noisy.png")
        self._plot_roc(y_true_n, prob_n_v, optimal_threshold, title="ROC Curve: Noisy Data", filename="roc_noisy.png")
        groups_noisy = self._compute_metrics_bootstrap(y_true_n, y_pred_n_opt)
        
        self._plot_boxplots(groups_clean, groups_noisy, title="Metrics CI (Normal vs V)", filename="boxplots_binary.png")
        self._plot_comparison_bar(groups_clean, groups_noisy, title="Classifier Robustness Comparison")
        self._print_metrics(groups_clean, groups_noisy)

    def _print_metrics(self, clean_groups, noisy_groups):
        print(f"\n{'='*70}\nФИНАЛЬНЫЕ МЕТРИКИ: Normal vs V (Bootstrap 1000 CI)\n{'='*70}")
        print(f"{'Метрика':<15} | {'Чистые данные':<25} | {'Зашумленные данные':<25}\n{'-'*70}")
        for metric in ['Accuracy', 'Recall', 'Specificity', 'F1-Score']:
            c_mean = np.mean(clean_groups[metric]); c_ci = np.percentile(clean_groups[metric], [2.5, 97.5])
            n_mean = np.mean(noisy_groups[metric]); n_ci = np.percentile(noisy_groups[metric], [2.5, 97.5])
            print(f"{metric:<15} | {c_mean:.4f} [{c_ci[0]:.4f}, {c_ci[1]:.4f}] | {n_mean:.4f} [{n_ci[0]:.4f}, {n_ci[1]:.4f}]")
        print(f"{'='*70}")

    def _plot_cm(self, y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred); plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'V'], yticklabels=['Normal', 'V'])
        plt.title(title); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150); plt.close()

    def _plot_roc(self, y_true, y_prob, threshold, title, filename):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob); roc_auc = roc_auc_score(y_true, y_prob)
        plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.scatter(fpr[np.argmin(np.abs(thresholds - threshold))], tpr[np.argmin(np.abs(thresholds - threshold))], marker='o', color='red', s=100, label=f'Threshold = {threshold:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)'); plt.ylabel('True Positive Rate (Recall)')
        plt.title(title); plt.legend(loc="lower right"); plt.savefig(os.path.join(self.results_dir, filename), dpi=150); plt.close()

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
        labels = list(clean_groups.keys()); clean_means = [np.mean(clean_groups[l]) for l in labels]; noisy_means = [np.mean(noisy_groups[l]) for l in labels]
        x = np.arange(len(labels)); width = 0.35; fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, clean_means, width, label='Clean Data', color='royalblue')
        rects2 = ax.bar(x + width/2, noisy_means, width, label='Noisy Data', color='salmon')
        ax.set_ylabel('Score'); ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.1); ax.legend(loc='lower right')
        for rect in rects1 + rects2:
            height = rect.get_height(); ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.tight_layout(); plt.savefig(os.path.join(self.results_dir, 'comparison_metrics.png'), dpi=150); plt.close()