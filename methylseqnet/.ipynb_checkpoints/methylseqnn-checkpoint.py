import torch
import torch.nn as nn
import torch.optim as optim
import gin
import lightning as L

from methylseqnet.layers import *

@gin.configurable
class MethylSeqNN(L.LightningModule):
    def __init__(
        self, 
        in_channels=7,
        out_tracks=None,
        out_bins=None,
        learning_rate=0.005, 
        momentum=0.98, 
        pos_weight=100,
    ):
        super().__init__()
        if out_tracks is None:
            raise ValueError("MethylSeqNN requires out_tracks be specified in the gin config file or when instantiating the class.")
        
        self.in_channels = in_channels
        self.out_tracks = out_tracks
        self.out_bins = out_bins
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.pos_weight = torch.tensor([pos_weight])

        # Trunk
        self.conv_dna = ConvDNA(in_channels=self.in_channels)
        self.conv_tower = ConvTower()
        self.conv_block = ConvBlock()
        self.conv_dropout = ConvDropout()

        # Head
        self.conv_final = ConvFinal(filters=self.out_tracks)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x):
        x = self.conv_dna(x)
        x = self.conv_tower(x)
        x = self.conv_block(x)
        x = self.conv_dropout(x)
        x = self.conv_final(x)
        return x

    def training_step(self,batch,batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)  
        trim_off_targets = targets.shape[2] - self.out_bins
        targets = targets[:, :, trim_off_targets // 2:-trim_off_targets // 2]
        if mask is not None:
            mask = mask[:, :, trim_off_targets // 2:-trim_off_targets // 2]
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss  

    def validation_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)
        trim_off_targets = targets.shape[2] - self.out_bins
        targets = targets[:, :, trim_off_targets // 2:-trim_off_targets // 2]
        if mask is not None:
            mask = mask[:, :, trim_off_targets // 2:-trim_off_targets // 2]
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer
    
    def get_layer(self, layer_name):
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in the model")