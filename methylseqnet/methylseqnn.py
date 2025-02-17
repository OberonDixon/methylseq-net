import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import gin
import lightning as L
# from memory_profiler import profile
import gc
import torchmetrics
import zipfile
import pandas as pd
from io import StringIO

from methylseqnet.transforms import *
from methylseqnet.layers import *
from methylseqnet.losses import *
from methylseqnet.pretrained import *

gin.register(nn.Softplus)

@gin.configurable
class MethylSeqNN(L.LightningModule):
    def __init__(
        self, 
        layers,
        pretrained_seq_model_generator=None,
        pretrained_seq_model_weights=None,
        train_stages={
            0:{
                'mode':'full-model',
                'grad_dict':{
                    'layers':True,
                    'pretrained_seq_model':False,
                    'seq_input_head':True,
                    'seq_output_head':True,
                }
            }
        },
        seq_input_head=None,
        seq_output_head=None,
        model_merge_operation='multiply',
        out_tracks=None,
        regression=False,
        pad_all_layers=False,
        crop_off_final=None, # consider adjusted this name to be more clearly about how much is cropped off. Also, can't be zero??
        label_threshold_cts=5,
        learning_rate=0.005, 
        momentum=0.98, 
        pos_weight=100,
        betas=(0.97,0.98),
    ):
        """
        Args:
            layers: a list of nn.Modules that run sequentially to form the seq+methyl model
        """
        super().__init__()
        if out_tracks is None:
            raise ValueError("MethylSeqNN requires out_tracks be specified in the gin config file or when instantiating the class.")
        
        self.train_stages = train_stages
        # print(self.train_stages)
        self.mode = 'full-model'
        self.pad_all_layers = pad_all_layers
        self.crop_off_final = crop_off_final
        self.model_merge_operation = model_merge_operation
        # TODO: rename layers to something like residual_methylseq_model
        self.layers = nn.ModuleList()
        if layers:
            for layer in layers:
                try:
                    self.layers.append(layer(pad=self.pad_all_layers))
                except:
                    self.layers.append(layer())
        self.receptive_field,self.total_stride = self.calculate_receptive_field_and_stride()
        
        self.pretrained_seq_model = nn.ModuleList()
        self.seq_input_head = nn.ModuleList()
        self.seq_output_head = nn.ModuleList()
        if pretrained_seq_model_generator is not None:
            self.pretrained_seq_model = pretrained_seq_model_generator(pretrained_seq_model_weights)
            # if pretrained_seq_model_weights:
            #     self.pretrained_seq_model
            for param in self.pretrained_seq_model.parameters():
                param.requires_grad = False
            if seq_input_head:
                for layer in seq_input_head:
                    self.seq_input_head.append(layer())
            if seq_output_head:
                for layer in seq_output_head:
                    self.seq_output_head.append(layer())

        
        self.regression = regression
        self.label_threshold_cts = label_threshold_cts
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.pos_weight = torch.tensor([pos_weight])
        self.betas = betas

        self.io_mappings_str = ''

        if self.regression:
            # self.softplus = nn.Softplus()
            self.criterion = CustomPoissonNLLLossLogTransformed()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x):
        # TODO: add shape assertions here for dim 0, etc -> what do we expect as layers progress
        if self.layers and self.mode in ['full-model','residual-only']:
            x_methylseq = x
            for layer in self.layers:
                x_methylseq = layer(x_methylseq)
            # TODO: check that this cropping logic isn't busted in some cases - e.g. what if crop_off_final is zero??
            if self.crop_off_final:
                x_methylseq = x_methylseq[:,:,self.crop_off_final:-self.crop_off_final]
            if not self.pretrained_seq_model or self.mode=='residual-only':
                return x_methylseq
        if self.pretrained_seq_model and self.mode in ['full-model','pretrained-only']:
            x_seq = x
            if self.seq_input_head:
                for layer in self.seq_input_head:
                    x_seq = layer(x_seq)
            x_seq = self.pretrained_seq_model(x_seq)
            if self.seq_output_head:
                for layer in self.seq_output_head:
                    x_seq = layer(x_seq)
            if not self.layers or self.mode=='pretrained-only':
                return x_seq
        if self.layers and self.pretrained_seq_model and self.mode=='full-model':
            operations = {
                'multiply': torch.mul,  # Element-wise multiplication
                'add': torch.add        # Element-wise addition
            }
            
            if self.model_merge_operation in operations:
                x = operations[self.model_merge_operation](x_methylseq, x_seq)
            else:
                raise ValueError(f"Unsupported model_merge_operation: {self.model_merge_operation}")
            
        return x

    def training_step(self,batch,batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)  
        targets = self.trim_targets(inputs,targets)
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
        if mask is not None:
            mask = self.trim_targets(inputs,mask)
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss  

    def validation_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)
        targets = self.trim_targets(inputs,targets)
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
        if mask is not None:
            mask = self.trim_targets(inputs,mask)
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)
        targets = self.trim_targets(inputs,targets)
        if not self.regression:
            targets = (targets>torch.log10(torch.tensor(float(self.label_threshold_cts)) + 1)).float()
        if mask is not None:
            mask = self.trim_targets(inputs,mask)
            outputs = outputs[mask]
            targets = targets[mask]

        loss = self.criterion(outputs, targets)
        self.test_targets_list.extend(targets.cpu().numpy().tolist())
        self.test_outputs_list.extend(outputs.cpu().numpy().tolist())
        return loss
    
    def configure_optimizers(self):
        # Define parameter groups based on the layer's weight decay
        param_groups = []
        
        for attr_name in dir(self):  # Iterate over all attributes of the model
            if attr_name.startswith("_"):  # Skip private attributes
                continue
            
            module = getattr(self, attr_name)
            
            if isinstance(module, nn.Module):  # Check if it's an nn.Module or nn.ModuleList
                print(attr_name)
                params = [p for p in module.parameters() if p.requires_grad]  # Filter trainable params
                if params:  # Only add if there are trainable params
                    weight_decay = getattr(module, 'weight_decay', 0)
                    param_groups.append({'params': params, 'weight_decay': weight_decay})
        
        if not param_groups:
            raise ValueError("No trainable parameters found. Ensure at least one module has trainable parameters.")
        if self.regression:
            optimizer = [optim.Adam(param_groups, lr=self.learning_rate, betas=self.betas)]
        else:
            optimizer = [optim.SGD(param_groups, lr=self.learning_rate, momentum=self.momentum)]
        return optimizer
    
    def get_layer(self, layer_name):
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in the model")
        
    def set_module_requires_grad(self, grad_dict):
        """
        Sets requires_grad for entire modules in a model based on a dictionary.
    
        :param self: The PyTorch model
        :param grad_dict: Dictionary where keys are model attribute names (str) and values are booleans (True/False)
        """
        for attr_name, requires_grad in grad_dict.items():
            if hasattr(self, attr_name):  # Check if attribute exists
                module = getattr(self, attr_name)
                if isinstance(module, nn.Module):  # Ensure it's an nn.Module or nn.ModuleList
                    for param in module.parameters():
                        param.requires_grad = requires_grad
                        
    def on_train_epoch_start(self,*args,**kwargs):
        if self.current_epoch in self.train_stages:
            stage_dict = self.train_stages[self.current_epoch]
            if 'mode' in stage_dict:
                self.mode = stage_dict['mode']
            if 'grad_dict' in stage_dict:
                self.set_module_requires_grad(stage_dict['grad_dict'])
                # print("resetting optimizers maybe?")
                # self.trainer.strategy.setup_optimizers(self.trainer)
                # trainer.optimizers = self.configure_optimizers()

    def on_test_epoch_start(self):
        self.test_targets_list = []
        self.test_outputs_list = []
        return         

    def on_save_checkpoint(self, checkpoint):
        checkpoint["operative_config_str"] = gin.operative_config_str()
        checkpoint["io_mappings_str"] = self.io_mappings_str
        # Store metadata for reloading the external model
        if self.pretrained_seq_model is not None:
            external_model = self.pretrained_seq_model
            checkpoint["pretrained_model_class"] = external_model.__class__.__name__
            checkpoint["pretrained_model_module"] = external_model.__class__.__module__

    def on_load_checkpoint(self, checkpoint):
        self.io_mappings_str = checkpoint.get("io_mappings_str","")
     
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        # Load the checkpoint to extract the gin config
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        # Parse the gin configuration from the checkpoint
        operative_config_str = checkpoint["operative_config_str"]
        # TODO: verify that clearing config is necessary
        gin.clear_config()
        gin.parse_config(operative_config_str)
        # Reconstruct the pretrained model dynamically
        # pretrained_seq_model = None
        # if "pretrained_model_class" in checkpoint and "pretrained_model_module" in checkpoint:
        #     module_name = checkpoint["pretrained_model_module"]
        #     class_name = checkpoint["pretrained_model_class"]

        #     # Dynamically import the module and instantiate the model
        #     module = importlib.import_module(module_name)
        #     model_class = getattr(module, class_name)
        #     pretrained_seq_model = model_class()
        del checkpoint
        gc.collect() 
        # Continue with the regular loading process
        return super().load_from_checkpoint(
            checkpoint_path, 
            *args, 
            **kwargs
        )

    def trim_targets(self,inputs,targets):
        """
        Trim the targets (or the targets mask) to what the model will actually be able to output.
        Based on the calculated receptive field alongside the cropping of the final layer.

        Trimming is basically taking into account the receptive field to determine how many labels must be trimmed
        off the end given there is no padding in the network, i.e. all predictions have full sequence information
        and thus for bigger receptive field there are fewer prediction bins along the sequence

        Args:
            inputs: the input tensor provided to the model. This will be used to determine the input lengths
            targets: the targets (or targets mask) that needs to be trimmed based on the input and network
        """
        inputs_length = inputs.shape[2]
        targets_length = targets.shape[2]
        
        if self.layers:
            if not self.pad_all_layers:
                network_outputs_length = (inputs_length - self.receptive_field+self.total_stride)//self.total_stride
            else:
                network_outputs_length = inputs_length//self.total_stride
    
            trim_off_targets = 2*self.crop_off_final + targets_length - network_outputs_length
        else:
            trim_off_targets = 2*self.crop_off_final
        
        if trim_off_targets>1:
            return targets[:, :, trim_off_targets // 2:-trim_off_targets // 2]
        else:
            return targets
    
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