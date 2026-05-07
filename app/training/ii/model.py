import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        
    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.padding, 0)))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, dilation=1, use_se=True):
        super().__init__()
        self.conv1 = CausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_out, n_out, kernel_size, dilation)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.se = SEBlock(n_out) if use_se else nn.Identity() # Добавляем SE
        
    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x)))
        out = self.conv2(out)
        out = self.se(out) # Применяем внимание
        return self.relu(out + self.downsample(x))

class TCNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        channels = [64] * 5
        dilations = [1, 2, 4, 8, 16]
        layers = []
        
        # ВНИМАНИЕ: Первый слой теперь принимает 2 канала (ЭКГ + RR)
        in_channels = 2 
        for i in range(len(dilations)):
            layers.append(ResidualBlock(in_channels if i==0 else channels[i-1], channels[i], 5, dilations[i]))
            
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def forward(self, x):
        return self.fc(self.pool(self.network(x)).squeeze(-1))