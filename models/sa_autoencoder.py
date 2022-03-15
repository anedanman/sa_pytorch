import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from modules import Decoder, PosEmbeds, SlotAttention
from utils import spatial_broadcast, spatial_flatten


class SlotAttentionAE(pl.LightningModule):
    """
    Slot attention based autoencoder for object discovery task
    """
    def __init__(self, resolution=(128, 128), num_slots=6, num_iters=3, in_channels=3, slot_size=64, hidden_size=64, lr=0.00043):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in range(3)]
            )
        self.decoder_initial_size = (8, 8)

        self.decoder = Decoder()
        self.enc_emb = PosEmbeds(64, self.resolution)
        self.dec_emb = PosEmbeds(64, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )
        self.slot_attention = SlotAttention(num_slots=num_slots, iters=num_iters, dim=slot_size, hidden_dim=slot_size*2)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.enc_emb(x)

        x = spatial_flatten(x)
        x = self.layer_norm(x)
        x = self.mlp(x)

        slots = self.slot_attention(x)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        x = self.dec_emb(x)
        x = self.decoder(x)
        
        x = x.reshape(inputs.shape[0], self.num_slots, *x.shape[1:])
        recons, masks = torch.split(x, self.in_channels, dim=2)
        masks = F.softmax(masks, dim=1)
        recons = recons * masks
        result = torch.sum(recons, dim=1)
        return result, recons

    def step(self, batch):
        imgs = batch['image']
        result, _ = self(imgs)
        loss = F.mse_loss(result, imgs)
        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer
        loss = self.step(batch)
        self.log('training MSE', loss, on_step=True, on_epoch=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('validating MSE', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams["lr"], total_steps=300000, pct_start=0.2)
        return [optimizer], [scheduler]
