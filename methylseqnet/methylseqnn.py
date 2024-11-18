import torch
import torch.nn as nn
import torch.optim as optim
import gin
import lightning as L
from memory_profiler import profile
import gc
import torchmetrics
import zipfile
import pandas as pd
from io import StringIO

from methylseqnet.transforms import *
from methylseqnet.layers import *
from methylseqnet.losses import *

@gin.configurable
class MethylSeqNN(L.LightningModule):
    def __init__(
        self, 
        layers,
        out_tracks=None,
        regression=False,
        pad_all_layers=False,
        crop_off_final=0,
        label_threshold_cts=5,
        learning_rate=0.005, 
        momentum=0.98, 
        pos_weight=100,
        betas=(0.97,0.98),
    ):
        super().__init__()
        if out_tracks is None:
            raise ValueError("MethylSeqNN requires out_tracks be specified in the gin config file or when instantiating the class.")
        
        self.pad_all_layers = pad_all_layers
        self.crop_off_final = crop_off_final
        self.layers = nn.ModuleList()
        for layer in layers:
            try:
                self.layers.append(layer(pad=self.pad_all_layers))
            except:
                # print(f"Padding is not defined for {layer}, adding without specifying padding.")
                self.layers.append(layer())
        self.receptive_field,self.total_stride = self.calculate_receptive_field_and_stride()
        # print(f"receptive field calculated to be {self.receptive_field}")
        # self.out_bins = out_bins

        
        self.regression = regression
        self.label_threshold_cts = label_threshold_cts
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.pos_weight = torch.tensor([pos_weight])
        self.betas = betas

        self.io_mappings_str = ''

        if self.regression:
            self.softplus = nn.Softplus()
            self.criterion = CustomPoissonNLLLossLogTransformed()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.regression:
            x = self.softplus(x)
        if self.crop_off_final:
            x = x[:,:,self.crop_off_final:-self.crop_off_final]
        return x

    def training_step(self,batch,batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)  
        # Trimming is basically taking into account the receptive field to determine how many labels must be trimmed
        # off the end given there is no padding in the network, i.e. all predictions have full sequence information
        # and thus for bigger receptive field there are fewer prediction bins along the sequence
        trim_off_targets = 2*self.crop_off_final + (not self.pad_all_layers)*(targets.shape[2]-((inputs.shape[2]-self.receptive_field+self.total_stride)//self.total_stride))
        # raise ValueError(f"receptive field {self.receptive_field}, trim off {trim_off_targets}")
        targets = targets[:, :, trim_off_targets // 2:-trim_off_targets // 2]
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
        if mask is not None:
            mask = mask[:, :, trim_off_targets // 2:-trim_off_targets // 2]
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.criterion(outputs, targets)
        print(outputs.shape,targets.shape,mask.shape)
        print(loss)
        print(self.layers[0].channels)
        print(self.layers[1].conv.weight.grad)
        self.log("train_loss", loss)
        return loss  

    def validation_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)
        # Trimming is basically taking into account the receptive field to determine how many labels must be trimmed
        # off the end given there is no padding in the network, i.e. all predictions have full sequence information
        # and thus for bigger receptive field there are fewer prediction bins along the sequence
        trim_off_targets = 2*self.crop_off_final + (not self.pad_all_layers)*(targets.shape[2]-((inputs.shape[2]-self.receptive_field+self.total_stride)//self.total_stride))
        # raise ValueError(f"receptive field {self.receptive_field}, trim off {trim_off_targets}")
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
        # Trimming is basically taking into account the receptive field to determine how many labels must be trimmed
        # off the end given there is no padding in the network, i.e. all predictions have full sequence information
        # and thus for bigger receptive field there are fewer prediction bins along the sequence
        trim_off_targets = 2*self.crop_off_final + (not self.pad_all_layers)*(targets.shape[2]-((inputs.shape[2]-self.receptive_field+self.total_stride)//self.total_stride))
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
        # Define parameter groups based on the layer's weight decay
        param_groups = []
        for layer in self.layers:
            param_groups.append({'params': layer.parameters(), 'weight_decay': getattr(layer, 'weight_decay', 0)})
        if self.regression:
            optimizer = optim.Adam(param_groups, lr=self.learning_rate, betas=self.betas)
        else:
            optimizer = optim.SGD(param_groups, lr=self.learning_rate, momentum=self.momentum)
        return optimizer
    
    def get_layer(self, layer_name):
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in the model")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["operative_config_str"] = gin.operative_config_str()
        checkpoint["io_mappings_str"] = self.io_mappings_str

    def on_load_checkpoint(self, checkpoint):
        self.io_mappings_str = checkpoint.get("io_mappings_str","")
        
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        # Load the checkpoint to extract the gin config
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        # Parse the gin configuration from the checkpoint
        operative_config_str = checkpoint["operative_config_str"]
        gin.clear_config()
        gin.parse_config(operative_config_str)
        del checkpoint
        gc.collect() 
        # Continue with the regular loading process
        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

    def calculate_receptive_field_and_stride(self):
        """
        Calculate the receptive field based on the layers in the model.
        Each layer that modulates receptive field or stride is assumed to have one or more attributes:
            - kernel_size: the size of the convolution filter
            - stride: the stride of the layer
            - dilation: the dilation of the layer
            - pool_size: the size of a pooling layer after the convolution
            - repeat: how many times the layer inside gets repeated
    
        NOTE: this should be possible to adapt to more sophisticated dilation schemes or to transformer
        layers but that will require additional testing
        """
        # Initial receptive field size
        receptive_field = 1
        # Initial stride (the first input)
        total_stride = 1
        # Total pooling
        total_pooling = 1
    
        for layer in self.layers:
            kernel_size = getattr(layer, 'kernel_size', 1)
            pool_size = getattr(layer, 'pool_size', 1)
            stride = getattr(layer, 'stride', 1)
            dilation = getattr(layer, 'dilation', 1)
            repeat = getattr(layer, 'repeat', 1)
            rate_mult = getattr(layer, 'rate_mult', 1.0)
            
            # Update receptive field using the formula
            for _ in range(repeat):
                receptive_field += (kernel_size - 1) * total_stride * dilation
                total_stride *= stride  # Update the total stride
                receptive_field += (pool_size - 1) * total_stride
                total_stride *= pool_size
                dilation *= rate_mult
                dilation = round(dilation)
    
        return receptive_field,total_stride

    def get_io_mappings_df(self):
        io_mappings_df = pd.read_csv(StringIO(self.io_mappings_str),sep='\t',header=0)
        return io_mappings_df