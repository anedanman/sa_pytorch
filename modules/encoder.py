from torch import nn


class Encoder(nn.Module):
    """
    CNN encoder for set prediction task, which reduces the spatial resolution 4 times
    """
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, hidden_size, 5, padding=(2, 2)), nn.ReLU(),
            nn.ZeroPad2d((1, 3, 1, 3)),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(0, 0), stride=2), nn.ReLU(),
            nn.ZeroPad2d((1, 3, 1, 3)),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(0, 0), stride=2), nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(2, 2)), nn.ReLU()
        ])

    def forward(self, inputs):
        return self.layers(inputs)
