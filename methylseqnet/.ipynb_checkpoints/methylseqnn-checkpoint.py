import torch
import torch.nn as nn
import torch.optim as optim
import gin
import lightning as L
from memory_profiler import profile
import gc
import torchmetrics
import zipfile

from methylseqnet.layers import *

@gin.configurable
class MethylSeqNN(L.LightningModule):
    def __init__(
        self, 
        layers,
        out_tracks=None,
        out_bins=None,
        regression=False,
        label_threshold_cts=5,
        learning_rate=0.005, 
        momentum=0.98, 
        pos_weight=100,
    ):
        super().__init__()
        if out_tracks is None:
            raise ValueError("MethylSeqNN requires out_tracks be specified in the gin config file or when instantiating the class.")
        
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(layer())
        # self.out_tracks = out_tracks
        self.out_bins = out_bins
        self.regression = regression
        self.label_threshold_cts = label_threshold_cts
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.pos_weight = torch.tensor([pos_weight])

        # Trunk
        # self.encoding_adjuster = EncodingAdjuster()
        # self.conv_dna = ConvDNA(in_channels = self.encoding_adjuster.channels)
        # self.conv_tower = ConvTower()
        # self.conv_block = ConvBlock()
        # self.conv_dropout = ConvDropout()

        # # Head
        # self.conv_final = ConvFinal(filters=self.out_tracks)

        if self.regression:
            self.softplus = nn.Softplus()
            self.criterion = nn.PoissonNLLLoss(log_input=False)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = self.encoding_adjuster(x)
        # x = self.conv_dna(x)
        # x = self.conv_tower(x)
        # x = self.conv_block(x)
        # x = self.conv_dropout(x)
        # x = self.conv_final(x)
        if self.regression:
            x = self.softplus(x)
        return x

    def training_step(self,batch,batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)  
        trim_off_targets = targets.shape[2] - self.out_bins
        targets = targets[:, :, trim_off_targets // 2:-trim_off_targets // 2]
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
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
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
        if mask is not None:
            mask = mask[:, :, trim_off_targets // 2:-trim_off_targets // 2]
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def on_test_epoch_start(self):
        self.test_targets_list = []
        self.test_outputs_list = []
        return
        
    def test_step(self, batch, batch_idx):
        inputs, targets, mask = batch

        outputs = self(inputs)
        trim_off_targets = targets.shape[2] - self.out_bins
        targets = targets[:, :, trim_off_targets // 2:-trim_off_targets // 2]
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
        if mask is not None:
            mask = mask[:, :, trim_off_targets // 2:-trim_off_targets // 2]
            outputs = outputs[mask]
            targets = targets[mask]

        loss = self.criterion(outputs, targets)
        self.test_targets_list.extend(targets.cpu().numpy().tolist())
        self.test_outputs_list.extend(outputs.cpu().numpy().tolist())
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer
    
    def get_layer(self, layer_name):
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in the model")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["operative_config_str"] = gin.operative_config_str()
        
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        # Load the checkpoint to extract the gin config
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        # Parse the gin configuration from the checkpoint
        operative_config_str = checkpoint["operative_config_str"]
        gin.parse_config(operative_config_str)
        del checkpoint
        gc.collect() 
        # Continue with the regular loading process
        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)