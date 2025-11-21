# models/selfgated_hierarchial_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


class SelfGating(nn.Module):
    """Simple element wise self gating."""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class SelfGatedHierarchicalTransformerEncoder(nn.Module):
    """
    Self Gated Hierarchical Transformer Encoder for time series classification.
    """
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers_low=3,
        num_layers_high=3,
        dim_feedforward=128,
        dropout=0.001,
        pool_output_size=10,
        num_classes=21,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer_low = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_low = nn.TransformerEncoder(
            encoder_layer_low, num_layers=num_layers_low
        )

        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)
        self.self_gate = SelfGating(d_model)

        encoder_layer_high = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_high = nn.TransformerEncoder(
            encoder_layer_high, num_layers=num_layers_high
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        self.pool_output_size = pool_output_size
        self.d_model = d_model

    def forward_features(self, x):
        """
        Compute feature representation before the classifier.
        """
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        low_out = self.encoder_low(z)
        pooled = self.pool(low_out.transpose(1, 2)).transpose(1, 2)
        gated = self.self_gate(pooled)
        high_out = self.encoder_high(gated)
        feat = high_out.mean(dim=1)
        return feat

    def forward(self, x, return_gates: bool = False):
        B, T, F = x.shape

        z = self.input_proj(x)
        z = self.pos_encoder(z)
        low_out = self.encoder_low(z)
        pooled = self.pool(low_out.transpose(1, 2)).transpose(1, 2)

        with torch.no_grad():
            gate_linear = self.self_gate.gate[0]
            gate_sigmoid = self.self_gate.gate[1]
            raw_gates = gate_sigmoid(gate_linear(pooled.detach()))

        gated = self.self_gate(pooled)
        high_out = self.encoder_high(gated)

        logits_per_timestep = self.classifier(high_out)
        final_logits = logits_per_timestep.mean(dim=1)

        if return_gates:
            extras = {
                "gates": raw_gates,
                "W_proj": self.input_proj.weight.detach(),
                "pooled": pooled.detach(),
            }
            return final_logits, extras

        return final_logits
