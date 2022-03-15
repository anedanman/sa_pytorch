import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from modules import Encoder, PosEmbeds, SlotAttention
from utils import spatial_flatten, hungarian_huber_loss


class SlotAttentionClassifier(pl.LightningModule):
    """
    Slot Attention based classifier for set prediction task
    """
    def __init__(self, resolution=(128, 128), num_slots=6, num_iters=3, in_channels=3, hidden_size=64, slot_size=64, lr=0.0004):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.slot_size = slot_size

        self.encoder_cnn = Encoder(in_channels=self.in_channels, hidden_size=hidden_size)
        self.encoder_pos = PosEmbeds(hidden_size, (resolution[0] // 4, resolution[1] // 4))

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )

        self.slot_attention = SlotAttention(num_slots=num_slots, iters=num_iters, dim=slot_size, hidden_dim=slot_size*2)

        self.mlp_classifier = nn.Sequential(
            nn.Linear(slot_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 19),
            nn.Sigmoid()
        )
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder_cnn(inputs)
        x = self.encoder_pos(x)
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))
        x = self.slot_attention(x)
        return self.mlp_classifier(x)

    def step(self, batch):
        images = batch['image']
        targets = batch['target']
        predictions = self(images)
        loss = hungarian_huber_loss(predictions, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('training loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('training loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        return optimizer
