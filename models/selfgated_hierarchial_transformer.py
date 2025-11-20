import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding

    This encoding injects information about the position of tokens in the sequence,
    enabling the Transformer to leverage order without recurrence or convolution.

    Args:
        d_model (int): The dimensionality of the embedding vector.
        max_len (int): The maximum length of input sequences to be encoded.

    Attributes:
        pe (Tensor): A (1, max_len, d_model) tensor containing positional encodings.

    Forward Input:
        x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

    Forward Output:
        Tensor of same shape as input with positional encodings added.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Eq. (2) and (3): sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Eq. (2): P_{t,2i}
        pe[:, 1::2] = torch.cos(position * div_term)  # Eq. (3): P_{t,2i+1}
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input embeddings with shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Input embeddings plus positional encoding, same shape as input.
        """
        return x + self.pe[:, :x.size(1)].to(x.device)


class SelfGating(nn.Module):
    """
    Implements a self-gating mechanism as a simple element-wise gating operation.

    The gating weights are computed by a sigmoid-activated linear layer 
    applied to the input features, enabling the model to emphasize or suppress
    information adaptively.

    Args:
        d_model (int): The dimensionality of the input feature vectors.

    Forward Input:
        x (Tensor): Input features of shape (batch_size, seq_len, d_model).

    Forward Output:
        Tensor of same shape with gated features.
    """
    def __init__(self, d_model):
        super().__init__()
        # Eq. (11): G = σ(H_pool W_g + b_g)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply self-gating on input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Element-wise gated tensor of the same shape.
        """
        return x * self.gate(x)


class SelfGatedHierarchicalTransformerEncoder(nn.Module):
 
    """
    Self-Gated Hierarchical Transformer Encoder for time series classification.

    The architecture consists of:
    - A linear projection layer to increase input dimensionality.
    - Positional encoding added to the input embeddings.
    - A local Transformer encoder block (multiple layers).
    - Adaptive average pooling to reduce temporal dimension.
    - Self-gating mechanism after pooling.
    - A high-level Transformer encoder block (multiple layers).
    - A classifier on top of the high-level encoder outputs.

    Args:
        input_dim (int): Number of input features per time step.
        d_model (int): Dimensionality of embedding space in Transformer.
        nhead (int): Number of attention heads.
        num_layers_low (int): Number of layers in local Transformer encoder.
        num_layers_high (int): Number of layers in high-level Transformer encoder.
        dim_feedforward (int): Dimension of feedforward layers in Transformer.
        dropout (float): Dropout rate for Transformer layers.
        pool_output_size (int): Output size of temporal dimension after pooling.
        num_classes (int): Number of output classes for classification.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Forward Output:
        If return_gates=False (default):
            Tensor: Output logits of shape (batch_size, num_classes).
        If return_gates=True:
            (logits, extras) where extras is a dict with:
                - 'gates': (B, S, d_model) raw gate coefficients σ(W_g z + b)
                - 'W_proj': (d_model, F) input projection weights (for attribution)
                - 'pooled': (B, S, d_model) pooled representations before gating
    """
    def __init__(self, input_dim, d_model=64, nhead=4,
                 num_layers_low=3, num_layers_high=3,
                 dim_feedforward=128, dropout=0.001,
                 pool_output_size=10, num_classes=21):
        super().__init__()

        # Eq. (1): Linear projection X W_proj + b_proj
        self.input_proj = nn.Linear(input_dim, d_model)

        # Eq. (4): Add positional encoding after projection
        self.pos_encoder = PositionalEncoding(d_model)

        # Eq. (5–9): Local Transformer encoder block (L_ell layers)
        encoder_layer_low = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder_low = nn.TransformerEncoder(encoder_layer_low, num_layers=num_layers_low)

        # Eq. (10): Adaptive average pooling to reduce time dimension
        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        # Eq. (11)–(12): Self-gating after pooling
        self.self_gate = SelfGating(d_model)

        # Eq. (13): High-level Transformer encoder block (L_h layers)
        encoder_layer_high = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder_high = nn.TransformerEncoder(encoder_layer_high, num_layers=num_layers_high)

        # Eq. (15)–(17): Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),  # Eq. (15): z W_c1 + b_c1
            nn.ReLU(),
            nn.Dropout(0.2),          # Eq. (16): Dropout
            nn.Linear(128, num_classes)  # Eq. (17): h_drop W_c2 + b_c2
        )

        self.pool_output_size = pool_output_size
        self.d_model = d_model

    def forward(self, x, return_gates: bool = False):
        """
        Forward pass of the hierarchical transformer encoder.

        Args:
            x (Tensor): Input tensor with shape (batch_size, seq_len, input_dim).
            return_gates (bool): If True, also return gate coefficients and aux tensors.

        Returns:
            See class docstring.
        """
        B, T, F = x.shape

        # Eq. (1): Linear projection
        z = self.input_proj(x)

        # Eq. (4): Add positional encoding
        z = self.pos_encoder(z)

        # Eq. (5–9): Local Transformer encoding
        low_out = self.encoder_low(z)

        # Eq. (10): Adaptive pooling: H_pool = AdaptiveAvgPool1d(H_ell^T)^T
        pooled = self.pool(low_out.transpose(1, 2)).transpose(1, 2)  # (B, S, d_model)

        # Compute raw gates explicitly for interpretability
        with torch.no_grad():
            gate_linear = self.self_gate.gate[0]   # Linear(d_model -> d_model)
            gate_sigmoid = self.self_gate.gate[1]  # Sigmoid
            raw_gates = gate_sigmoid(gate_linear(pooled.detach()))  # (B, S, d_model)

        # Eq. (11)–(12): Self-gating
        gated = self.self_gate(pooled)

        # Eq. (13): High-level Transformer encoding
        high_out = self.encoder_high(gated)

        # Eq. (15)–(17): Classifier on each timestep, then mean-pool logits
        logits_per_timestep = self.classifier(high_out)
        final_logits = logits_per_timestep.mean(dim=1)

        if return_gates:
            extras = {
                'gates': raw_gates,                        # (B, S, d_model)
                'W_proj': self.input_proj.weight.detach(), # (d_model, F)
                'pooled': pooled.detach()                  # (B, S, d_model)
            }
            return final_logits, extras

        return final_logits  # Final prediction: ŷ ∈ ℝ^{B × num_classes}
