"""Four spatiotemporal architectures for DAS event detection.
Input tensor: (B, 1, C, T) with C = channels along the fibre (space) and T = time samples."""
import math
import torch
import torch.nn as nn


def conv_block(cin, cout, pool):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class CNN2D(nn.Module):
    """2-D CNN treating the (space x time) patch as an image."""
    def __init__(self, width=(16, 32, 48, 64), drop=0.3):
        super().__init__()
        layers, cin = [], 1
        for w in width:
            layers.append(conv_block(cin, w, 2)); cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(cin, 1))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(1)


class CNNLSTM(nn.Module):
    """CNN encodes space (and short time) per step; a bidirectional LSTM models the time sequence."""
    def __init__(self, width=(16, 32, 48), hidden=64, drop=0.3):
        super().__init__()
        layers, cin = [], 1
        for w in width:
            layers.append(conv_block(cin, w, (2, 2))); cin = w
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(cin, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        f = self.cnn(x).mean(dim=2)            # (B, F, T') : pool over space
        h, _ = self.lstm(f.transpose(1, 2))    # (B, T', 2H)
        return self.head(h.mean(dim=1)).squeeze(1)


class ConvLSTMCell1D(nn.Module):
    """ConvLSTM cell (Shi et al. 2015) with 1-D convolutions along the fibre axis."""
    def __init__(self, cin, hid, k=5):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv1d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Convolutional stem (shortens time), then a ConvLSTM that keeps the spatial (fibre) layout in its state."""
    def __init__(self, stem=(16, 32), hid=32, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(conv_block(1, stem[0], (2, 2)), conv_block(stem[0], stem[1], (2, 2)))
        self.cell = ConvLSTMCell1D(stem[1], hid)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid, 1))

    def forward(self, x):
        f = self.stem(x)                        # (B, F, C', T')
        B, F, C, T = f.shape
        h = f.new_zeros(B, self.cell.hid, C); c = torch.zeros_like(h)
        hs = []
        for t in range(T):
            h, c = self.cell(f[..., t], (h, c))
            hs.append(h.mean(dim=2))
        return self.head(torch.stack(hs, 1).mean(1)).squeeze(1)


class ViTDAS(nn.Module):
    """Vision-Transformer-style encoder on non-overlapping space-time patches."""
    def __init__(self, n_ch, n_t, patch=(8, 16), dim=64, depth=4, heads=4, drop=0.1):
        super().__init__()
        self.embed = nn.Conv2d(1, dim, patch, stride=patch)
        n_tok = (n_ch // patch[0]) * (n_t // patch[1])
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, drop, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t)[:, 0])).squeeze(1)


def build(name, n_ch, n_t):
    return {'CNN': lambda: CNN2D(), 'CNN-LSTM': lambda: CNNLSTM(), 'ConvLSTM': lambda: ConvLSTM(),
            'Transformer': lambda: ViTDAS(n_ch, n_t)}[name]()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
