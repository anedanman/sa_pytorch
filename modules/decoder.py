from torch import nn


class Decoder(nn.Module):
    """
    Decoder for autoencoder model
    """
    def __init__(self, num_channels=64):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
        ])
        self.final_module = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(2, 2), output_padding=0, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels, 4, 3, padding=(1,1), output_padding=0, stride=1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)[:, :, :-1, :-1]
        return self.final_module(x)
