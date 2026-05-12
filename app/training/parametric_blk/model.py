import torch
import torch.nn as nn

class FeatureAttentionGate(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.excitation(x)

class MLPClassifier(nn.Module):
    # ВХОД 31 (26 базовых + 5 дельт)
    def __init__(self, input_dim=34, num_classes=2, hidden_dims=[128, 64, 32], dropout=0.25, input_dropout=0.2):
        super().__init__()
        layers = [nn.Dropout(input_dropout), FeatureAttentionGate(input_dim, reduction=4)]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)