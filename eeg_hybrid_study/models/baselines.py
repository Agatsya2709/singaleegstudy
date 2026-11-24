import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.35):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(48, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        feat = x.unsqueeze(1)
        feat = self.temporal(feat)
        feat = self.spatial(feat).squeeze(2)  # [batch, 48, time]
        feat = F.adaptive_avg_pool1d(feat, output_size=1).squeeze(-1)
        feat = self.dropout(feat)
        return self.head(feat)


class LSTMBaseline(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_size: int = 96, dropout: float = 0.35):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time] -> [batch, time, channels]
        seq = x.transpose(1, 2)
        seq = self.norm(seq)
        out, _ = self.lstm(seq)
        feat = out[:, -1, :]
        feat = self.dropout(feat)
        return self.head(feat)


class CNNLSTMBaseline(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.35):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(48 + 192, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(160, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        feat = self.temporal(x.unsqueeze(1))
        feat = self.spatial(feat).squeeze(2)  # [batch, 48, time]

        cnn_feat = F.adaptive_avg_pool1d(feat, output_size=1).squeeze(-1)

        lstm_in = feat.transpose(1, 2)  # [batch, time, 48]
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = lstm_out[:, -1, :]

        fused = torch.cat([cnn_feat, lstm_feat], dim=1)
        fused = self.dropout(fused)
        return self.fusion(fused)


class EEGNetBaseline(nn.Module):
    """Compact EEGNet-style baseline for a fair EEG-specific reference model."""

    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.35, f1: int = 8, depth_multiplier: int = 2):
        super().__init__()
        f2 = f1 * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(f2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.temporal(x.unsqueeze(1))
        feat = self.spatial(feat)
        feat = self.separable(feat)
        feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        return self.head(feat)
