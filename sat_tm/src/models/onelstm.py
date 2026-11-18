import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)
