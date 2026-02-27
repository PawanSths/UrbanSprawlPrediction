# models_pokhara.py – ConvLSTM model architecture for Pokhara region

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell matching the Pokhara saved checkpoint structure"""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # Using 'gates' instead of 'conv' to match saved model
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class UrbanSprawlConvLSTM_Pokhara(nn.Module):
    """Pokhara-specific ConvLSTM architecture with BatchNorm and 3-channel output"""

    def __init__(self, input_channels=3, hidden_channels=64):
        super().__init__()
        # Encoder WITHOUT bias (matching saved checkpoint)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),  # encoder.0
            nn.BatchNorm2d(32),  # encoder.1
            nn.ReLU(),  # encoder.2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),  # encoder.3
            nn.BatchNorm2d(64),  # encoder.4
            nn.ReLU()  # encoder.5
        )

        # ConvLSTM: input=64 (from encoder), hidden=64
        self.clstm = ConvLSTMCell(64, hidden_channels)

        # Decoder - outputs 3 channels
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 64, 3, padding=1, bias=False),  # decoder.0
            nn.BatchNorm2d(64),  # decoder.1
            nn.ReLU(),  # decoder.2
            nn.Dropout2d(0.3),  # decoder.3
            nn.Conv2d(64, 32, 3, padding=1, bias=False),  # decoder.4
            nn.BatchNorm2d(32),  # decoder.5
            nn.ReLU(),  # decoder.6
            nn.Conv2d(32, 3, 1, padding=0),  # decoder.7 - 1x1 conv
            nn.Sigmoid()
        )

        self.hidden_channels = hidden_channels

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H, W).to(x.device)
        c = torch.zeros_like(h)

        for t in range(T):
            enc = self.encoder(x[:, t])
            h, c = self.clstm(enc, h, c)

        # Concatenate h and c before decoder (128 channels total)
        combined = torch.cat([h, c], dim=1)
        out = self.decoder(combined)
        return out