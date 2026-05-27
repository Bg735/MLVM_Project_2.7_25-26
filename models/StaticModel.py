import torch
import torch.nn as nn

class StaticSafetyClassifier(nn.Module):
    def __init__(self, input_dimension, num_classes, train_mean, train_std, dropout_rate):
        super().__init__()

        self.train_mean = train_mean
        self.train_std = train_std
        self.dropout_rate = dropout_rate

        self.backbone = nn.Sequential(
            self._block(input_dimension, 128),
            self._block(128, 64),
            self._block(64, 32)
        )
        self._head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self._head(x)
        return x

    def infer(self, x):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            x = (torch.tensor(x, dtype=torch.float32) - torch.tensor(self.train_mean, dtype=torch.float32)) / (
                        torch.tensor(self.train_std, dtype=torch.float32) + 1e-7)
            x = x.unsqueeze(0).to(device)

            output = self(x)
            prediction = torch.argmax(output, dim=1).item()
        return prediction

    def _block(self, in_f, out_f):
        return nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.fc(x)
        out = self.ln(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual