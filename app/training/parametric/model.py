import torch
import torch.nn as nn

class FeatureAttentionGate(nn.Module):
    """Аналог SE-блока для табличных признаков MLP.
    Учится подавлять зашумленные/ненадежные признаки и усиливать стабильные."""
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, 25]
        weights = self.excitation(x)  # Выпускаем веса от 0 до 1 для каждого признака
        return x * weights            # Масштабируем признаки (подавляем шумовые)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=25, num_classes=2, hidden_dims=[128, 64, 32], dropout=0.25, input_dropout=0.2):
        super().__init__()
        layers = []
        
        # 1. Слой случайного отключения входных признаков
        layers.append(nn.Dropout(input_dropout))
        
        # 2. ВАЖНО: Feature Attention Gate (аналог SE-блока)
        # Вставляем его сразу после дропаута, чтобы он оценивал "зашумленные" признаки
        layers.append(FeatureAttentionGate(input_dim, reduction=4))
        
        # 3. Основная сеть
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)