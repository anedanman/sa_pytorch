import torch
import wandb
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from modules import SlotAttention
from torch.optim import lr_scheduler


def spatial_broadcast(x, resolution):
    x = x.reshape(-1, x.shape[-1], 1, 1)
    x = x.expand(-1, -1, *resolution)
    return x


def spatial_flatten(x):
    x = torch.swapaxes(x, 1, -1)
    return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])


class SlotAttentionAE(pl.LightningModule):
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
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in range(4)]
            )
        self.decoder_initial_size = (8, 8)
        self.decoder = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=5, stride=(2, 2), padding=(2, 2)), nn.ReLU(),
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=5, stride=(2, 2), padding=(1, 1)), nn.ReLU()
                ) for _ in range(2)],
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=5, stride=(1, 1), padding=(1, 1)), nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size // 2, kernel_size=3, stride=(1, 1), padding=(0, 0)), nn.ReLU(),
            nn.ConvTranspose2d(hidden_size // 2, in_channels + 1, kernel_size=2, stride=(1, 1), padding=(0, 0))#, nn.Tanh()
            )
        self.pos_x = nn.Parameter(torch.randn(1, hidden_size, resolution[0], 1))
        self.pos_y = nn.Parameter(torch.randn(1, hidden_size, 1, resolution[1]))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, slot_size))
        self.slot_attention = SlotAttention(num_slots=num_slots, iters=num_iters, dim=slot_size, hidden_dim=slot_size*2)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder(inputs)   # B, C, H, W
        x += (self.pos_x + self.pos_y) / 2
        x = spatial_flatten(x)  # B, H*W, C
        x = self.layer_norm(x)
        x = self.mlp(x)

        slots = self.slot_attention(x)  # B, num_slots, slot_size

        x = spatial_broadcast(slots, self.decoder_initial_size)
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


class SlotAttentionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=4):
        super().__init__()
        self.val_samples = val_samples[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_samples = self.val_samples.to(device=pl_module.device)
        result, recons = pl_module(val_samples)
       # val_samples = val_samples.moveaxis(1, -1)
       # result = result.moveaxis(1, -1)
       # recons = recons.moveaxis(2, -1)
        trainer.logger.experiment.log({
            'images': [wandb.Image(x/2 + 0.5) for x in torch.clamp(val_samples, -1, 1)],
            'reconstructions': [wandb.Image(x/2 + 0.5) for x in torch.clamp(result, -1, 1)]
        })
        trainer.logger.experiment.log({
            f'{i} slot': [wandb.Image(x/2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]
             for i in range(pl_module.num_slots)
        })
