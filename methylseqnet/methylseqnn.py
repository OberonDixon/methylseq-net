import importlib
from collections import defaultdict

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
from methylseqnet.optimizers import *
from methylseqnet.pretrained import *
from methylseqnet.activations import *

gin.register(nn.Softplus)

@gin.configurable
class MethylSeqNN(L.LightningModule):
    def __init__(
        self, 
        # Core architecture
        layers,
        seq_input_head=None,
        seq_output_head=None,
        model_merge_operation='multiply',
        merged_output_head=None,
        out_tracks=None,
        
        # Pretrained model handling       
        pretrained_seq_model_generator=None,
        pretrained_seq_model_weights=None,
        concat_pretrained_embeddings_at={},

        # Cropping / padding behavior
        pad_all_layers=False,
        crop_off_sequence=None,
        crop_off_final=None, # consider adjusted this name to be more clearly about how much is cropped off. Also, can't be zero??
        # Training schedule and stage-specific config
        train_stages={},
        residual_activation_loss_weight=0,
        seq_only_loss_weight=0,
        prediction_criterion=PoissonLoss,
        seq_only_prediction_criterion=PoissonLoss,
        activation_criterion=LogL1Loss,
        label_threshold_cts=5,

        # Optimizer config
        optimizer_class=AdamOptimizer,
        learning_rate=None,
        betas=None,
        momentum=None,
        pos_weight=None,

        # Deprecated / legacy flags
        regression=False,
        pow=False, # temporarily brought back for backward compatibility; does nothing
    ):
        """
        Args:
            Core architecture:
             - layers: a list of nn.Modules that run sequentially to form the seq+methyl model
            Pretrained model handling:
             - pretrained_seq_model_generator: returns an nn.Module objects when called with the pretrained_seq_model_weights.
                 Which generator is provided here will determine the pretrained model architecture.
             - pretrained_seq_model_weights: specifier passed to pretrained_seq_model_generator to provide appropriate info
                 for the pretrained_seq_model. Expect str or Path. This should specify the pretrained weights not the
                 architecture.
             - concat_pretrained_embeddings_at: a dict for pretrained_seq_model embeddings injection into layers model. 
                Schema {layer_before_which_to_concat:relative_bin_size}; when relative bin size is >1 pooling will be
                used to pool embeddings down to size and when relative bin size is <1 interpolation will be used to get
                the embeddings up to size. In both cases, padding/cropping will be used to match up with the shape[-1]
                dimension of x after scaling.
            Cropping / padding behavior:
             - pad_all_layers
             - crop_off_sequence
             - crop_off_final
            Training schedule and stage-specific config:
             - train_stages: a dictionary providing at minimum a model `mode` and `epochs` count for a stage. May also provide
                 `grad_dict` to specify model submodules to train for this stage and `loss_dict` to specify loss function 
                 components.
           Optimizer config:

           Deprecated / legacy flags:
        """
        super().__init__()
        if out_tracks is None:
            raise ValueError("MethylSeqNN requires out_tracks be specified in the gin config file or when instantiating the class.")
        if not layers and not pretrained_seq_model_generator:
            raise ValueError("MethylSeqNN requires a defined methylseq model (self.layers) or a defined pretrained sequence model.")
        
        self.train_stages = train_stages
        self.mode = 'full-model' #'residual-w/-pretrained-embeddings'
        self.peak_subset_threshold = 0
        self.pad_all_layers = pad_all_layers
        self.crop_off_sequence = crop_off_sequence
        self.crop_off_final = crop_off_final
        self.concat_pretrained_embeddings_at = { # adjust negative indices to positive
            (i if i >= 0 else len(layers) + i): v
            for i, v in concat_pretrained_embeddings_at.items()
        }
        self.model_merge_operation = model_merge_operation
        # TODO: rename layers to something like residual_methylseq_model
        self.layers = nn.ModuleList()
        self.pretrained_seq_model = nn.ModuleList()
        self.seq_input_head = nn.ModuleList()
        self.seq_output_head = nn.ModuleList()
        self.merged_output_head = nn.ModuleList()
        if layers:
            for layer in layers:
                try:
                    self.layers.append(layer(pad=self.pad_all_layers))
                except:
                    self.layers.append(layer())
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
        if merged_output_head:
            for layer in merged_output_head:
                try:
                    self.merged_output_head.append(layer(pad=self.pad_all_layers))
                except:
                    self.merged_output_head.append(layer())             

        self.receptive_field,self.total_stride = self.calculate_receptive_field_and_stride()

        self.hooked_activations = {}
        self.residual_activation_loss_weight = residual_activation_loss_weight
        self.seq_only_loss_weight = seq_only_loss_weight
        if self.residual_activation_loss_weight!=0:
            self.layers[-1].register_forward_hook(self._capture_activations_hook)
        if self.seq_only_loss_weight!=0:
            self.seq_output_head[-1].register_forward_hook(self._capture_activations_hook)
            
        
        self.regression = regression
        self.label_threshold_cts = label_threshold_cts
        # self.learning_rate = learning_rate
        # self.momentum = momentum
        # self.pos_weight = torch.tensor([pos_weight])
        # self.betas = betas

        self.io_mappings_str = ''

        self.prediction_criterion = prediction_criterion()
        self.seq_only_prediction_criterion = seq_only_prediction_criterion()
        self.activation_criterion = activation_criterion()
        self.optimizer_class = optimizer_class
        self.start_epoch = 0

    def forward(self, x):
        """
        MethylSeqNN forward supports two types of inputs:
            standard input: a single x tensor (sample, channel, position) with 4 sequence channels and 3 methylation channels
            multimethyl input: a tensor with the same 4 sequence channels but 3*n methylation channels for n cell types

        These two input types exist to support either one-to-all mapping for methylation to activity or a more efficient shared-sequence 
        differential-methylation mode for multitask training or inference. In the latter case, the larger sequence-only model needs to run only once,
        while the methylseq residual model runs many times.

        TODO: this should be refactored into a match statement with a case per valid mode and duplicated code split into separate functions. It is
        too easy for me to get confused I can only imagine how anyone else feels reading this.
        """
        valid_modes = ( # the different ways to run the model forward pass
            'full-model', # run both the pretrained and residual models, including their outputs heads. Full prediction.
            'residual-only', # run only the residual model; outputs may not reflect true labels
            'pretrained-only', # run only the pretrained model, including its output head. Outputs still predict true labels.
            'pretrained-embeddings-only', # run only the pretrained model output head based on cached embeddings dataset.
            'residual-w/-pretrained-embeddings', # run full residual model combined with pretrained output head from cached embeddings.
            'residual-only-w/-pretrained-embeddings', # run only residual model, with cached embeddings available for concatenation
        )
        operations = { # the different operations that can be used to combine pretrained and residual models
            'multiply': torch.mul,  # Element-wise multiplication
            'add': torch.add        # Element-wise addition
        }
        
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid forward pass mode {self.mode}. Valid modes are {valid_modes}")
        if self.mode in ('pretrained-embeddings-only','residual-w/-pretrained-embeddings','residual-only-w/-pretrained-embeddings'):
            x, embeddings = x
        if x.shape[1]>7:
            multimethyl_input = True
        elif x.shape[1]==7:
            multimethyl_input = False
        else:
            raise ValueError(f"Forward passes for MethylSeqNN require that x have 7 or more channels; if using only DNA onehot you must pad up to 7 with zeros. Found shape was {x.shape[1]}")
        # TODO: add shape assertions here for dim 0, etc -> what do we expect as layers progress

        # this block runs if the pretrained_seq_model is defined and one of the following is true:
        # (1) the model is in a mode that runs the pretrained model
        # (2) the submodel in self.layers has pretrained embedding concatenation enabled, so we need the embeddings, and they aren't provided by the dataloader
        if self.pretrained_seq_model and (
            self.mode in ('full-model','pretrained-only') or (
                self.mode in ('residual-only') and self.concat_pretrained_embeddings_at
            )
        ):
            x_seq = x[:,0:7,:]
            if self.seq_input_head:
                for layer in self.seq_input_head:
                    x_seq = layer(x_seq)
            embeddings = self.pretrained_seq_model(x_seq)
            if self.seq_output_head:
                x_seq = embeddings
                for layer in self.seq_output_head:
                    x_seq = layer(x_seq)
            if not self.layers or self.mode in ('pretrained-only'):
                x = x_seq
                
        # this block runs if you are running with pretrained model embeddings rather than running forward passes on the model itself
        if self.mode in ('pretrained-embeddings-only','residual-w/-pretrained-embeddings'):
            x_seq = embeddings
            if self.seq_output_head:
                for layer in self.seq_output_head:
                    x_seq = layer(x_seq)
            if not self.layers or self.mode in ('pretrained-embeddings-only'): 
                x = x_seq
        
        # this block runs if the residual model layers are populated and if the model is in a mode that runs the residual model
        if self.layers and self.mode in ('full-model','residual-only','residual-w/-pretrained-embeddings'):
            if multimethyl_input:
                input_to_outputs_dict = defaultdict(list)
                for _,io_mappings_row in self.get_io_mappings_df().iterrows():
                    input_to_outputs_dict[int(io_mappings_row['cell_type'])].append(int(io_mappings_row['channel']))
                x_methylseq_allchannels = None
                x_pseudobatch_list = []
                for cell_type in input_to_outputs_dict.keys():
                    x_methylseq = torch.cat(
                        [
                            x[:,0:4,:],
                            x[:,4+3*cell_type:4+3*(cell_type+1),:],
                        ],
                        dim=1
                    )
                    if self.crop_off_sequence:
                        x_methylseq = x_methylseq[:,:,self.crop_off_sequence:-self.crop_off_sequence]
                    x_pseudobatch_list.append(x_methylseq)

                pseudobatch_scaleup = len(x_pseudobatch_list)
                x_pseudobatch = torch.cat(x_pseudobatch_list, dim=0)
                
                for layer_index,layer in enumerate(self.layers):
                    if layer_index in self.concat_pretrained_embeddings_at:
                        rbs = self.concat_pretrained_embeddings_at[layer_index]
                        x_pseudobatch = self._concat_pretrained_embeddings(
                            embeddings, 
                            rbs, 
                            x_pseudobatch, 
                            embeddings_pseudobatch_scaleup=pseudobatch_scaleup)
                    x_pseudobatch = layer(x_pseudobatch)
                if self.crop_off_final:
                    x_pseudobatch = x_pseudobatch[:,:,self.crop_off_final:-self.crop_off_final]
                
                x_methylseq_allchannels = x_pseudobatch.new_zeros(x.size(0), *x_pseudobatch.shape[1:])
                
                batch_size = x.size(0)
                for cell_type_idx, (cell_type, channels) in enumerate(input_to_outputs_dict.items()):
                    start = cell_type_idx*batch_size
                    end = (cell_type_idx+1)*batch_size
                    x_cell_type = x_pseudobatch[start:end]
                    x_methylseq_allchannels[:, channels, :] = x_cell_type[:, channels, :]
            else:
                if self.crop_off_sequence:
                    x_methylseq = x[:,:,self.crop_off_sequence:-self.crop_off_sequence]
                else:
                    x_methylseq = x
                for layer_index,layer in enumerate(self.layers):
                    if layer_index in self.concat_pretrained_embeddings_at:
                        rbs = self.concat_pretrained_embeddings_at[layer_index]
                        x_methylseq = self._concat_pretrained_embeddings(embeddings, rbs, x_methylseq)
                    x_methylseq = layer(x_methylseq)
                if self.crop_off_final:
                    x_methylseq_allchannels = x_methylseq[:,:,self.crop_off_final:-self.crop_off_final]
                else:
                    x_methylseq_allchannels = x_methylseq
            if not self.pretrained_seq_model or self.mode=='residual-only':
                x = x_methylseq_allchannels
                
        # this block runs if both pretrained and residual models are defined and if the full model is running
        if self.layers and self.pretrained_seq_model and self.mode in ('full-model','residual-w/-pretrained-embeddings'):
            if self.model_merge_operation in operations:
                x = operations[self.model_merge_operation](x_methylseq_allchannels, x_seq)
            else:
                raise ValueError(f"Unsupported model_merge_operation: {self.model_merge_operation}")
        # this block runs if there is a post-merge output head and if we aren't in a residual-only mode
        if self.merged_output_head and self.mode in ('full-model','pretrained-only','pretrained-embeddings-only','residual-w/-pretrained-embeddings'):
            for layer in self.merged_output_head:
                x = layer(x)
            
        return x

    def _concat_pretrained_embeddings(self, embeddings, rbs, x, embeddings_pseudobatch_scaleup=1):
        """
        resize the pretrained embeddings and concatenate with x.

        Args:
            embeddings: some pretrained embeddings of shape (N,C_pretrained,L_pretrained)
            rbs: the relative bin size. >1 means pooling must occur, <1 means interpolation must occur
            x: the embeddings to which the pretrained_seq embeddings must be concatenated. Shape (N, C_layer, L_layer)
            embeddings_pseudobatch_scaleup: duplicate embeddings to match pseudobatch. 1 means no scaleup. <1 not allowed.
        """  
        if embeddings_pseudobatch_scaleup < 1 or not isinstance(embeddings_pseudobatch_scaleup, int):
            raise ValueError(f"Invalid pseudobatch scaleup {embeddings_pseudobatch_scaleup}. Must be an int >= 1.")
            
        if embeddings_pseudobatch_scaleup > 1:
            embeddings = embeddings.repeat(embeddings_pseudobatch_scaleup, 1, 1)

        assert embeddings.shape[0] == x.shape[0], (
            f"Batch size mismatch: x has batch size {x.shape[0]}, "
            f"but pretrained embeddings have batch size {embeddings.shape[0]}"
        ) 
            
        if rbs == 1:
            embeddings_scaled = embeddings
        if rbs > 1:
            kernel = int(rbs)
            embeddings_scaled = F.avg_pool1d(embeddings,kernel_size=kernel,stride=kernel)
        elif rbs < 1:
            scale = int(round(1 / rbs))
            embeddings_scaled = F.interpolate(embeddings, scale_factor=scale, mode='linear', align_corners=False)
        else:
            ValueError(f"Invalid relative_bin_size {rbs}.")

        scaled_len = embeddings_scaled.shape[-1]
        expected_len = x.shape[-1]
        diff = scaled_len - expected_len
        diff_ratio = scaled_len / expected_len if expected_len > 0 else 1
        if diff_ratio > 2 or diff_ratio < 0.5:
            raise ValueError(
                f"Unreasonable length mistamch after scaling embeddings: "
                f"expected ~{expected_len}, got {scaled_len} (ratio {diff_ratio})."
                f"This suggests misconfigured relative_bin_size (value {rbs}) or incompatible model layers."
            )
        if diff > 0:
            # Crop embeddings symmetrically
            left = diff // 2
            right = diff - left
            embeddings_matched = embeddings_scaled[:, :, left:-right if right > 0 else None]
        elif diff < 0:
            # Pad embeddings symmetrically
            pad_left = (-diff) // 2
            pad_right = (-diff) - pad_left
            embeddings_matched = F.pad(embeddings_scaled, (pad_left, pad_right))
        else:
            embeddings_matched = embeddings_scaled

        x_concatenated = torch.cat([x, embeddings_matched], dim=1)
        return x_concatenated
        
    def _capture_activations_hook(self, module, inputs, outputs):
        # Store the output of the module
        # Use id(module) or module.__class__.__name__ to distinguish them
        self.hooked_activations[id(module)] = outputs
    
    def training_step(self,batch,batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)  
        targets = self.trim_targets(inputs,targets)
        if not self.regression:
            targets = (targets>self.label_threshold_cts).float()
        if self.peak_subset_threshold:
            active_pos_mask = (targets > self.peak_subset_threshold).any(dim=1)
            fraction_true = active_pos_mask.float().mean().item()
            self.log("train_sites",fraction_true)
        else:
            active_pos_mask = torch.full_like(targets, True, dtype=torch.bool)
        if mask is not None:
            mask = self.trim_targets(inputs,mask)
            outputs = outputs[mask & active_pos_mask]
            targets = targets[mask & active_pos_mask]
        else:
            outputs = outputs[active_pos_mask]
            targets = targets[active_pos_mask]  
        if mask is None or mask.any():
            loss = self.prediction_criterion(outputs, targets)
            self.log("train_prediction_loss",loss)
        else:
            print(f"Fully masked for batch {batch_idx}. No gradients to compute.")
            loss = sum(param.sum() * 0.0 for param in self.parameters() if param.requires_grad)
        if self.residual_activation_loss_weight!=0 and id(self.layers[-1]) in self.hooked_activations:
            residual_activations = self.hooked_activations[id(self.layers[-1])]
            residual_activations_loss = self.activation_criterion(residual_activations)
            loss = loss + self.residual_activation_loss_weight * residual_activations_loss
            self.log("train_residual_activations_loss",residual_activations_loss)
        if self.seq_only_loss_weight!=0 and id(self.seq_output_head[-1]) in self.hooked_activations:
            seq_only_predictions = self.hooked_activations[id(self.seq_output_head[-1])]
            if mask is not None:
                seq_only_predictions = seq_only_predictions[mask & active_pos_mask]
            else:
                seq_only_predictions = seq_only_predictions[active_pos_mask]
            if mask is None or mask.any():
                seq_only_prediction_loss = self.seq_only_prediction_criterion(seq_only_predictions,targets)
                loss = loss + self.seq_only_loss_weight * seq_only_prediction_loss
                self.log("train_seq_only_prediction_loss",seq_only_prediction_loss)          
        self.log("train_loss", loss)
        self.hooked_activations.clear()
        return loss  

    def validation_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)  
        targets = self.trim_targets(inputs,targets)
        if not self.regression:
            targets = (targets>self.label_threshold_cts).float()
        if self.peak_subset_threshold:
            active_pos_mask = (targets > self.peak_subset_threshold).any(dim=1)
            fraction_true = active_pos_mask.float().mean().item()
            self.log("val_sites",fraction_true)
        else:
            active_pos_mask = torch.full_like(targets, True, dtype=torch.bool)
        if mask is not None:
            mask = self.trim_targets(inputs,mask)
            outputs = outputs[mask & active_pos_mask]
            targets = targets[mask & active_pos_mask]
        else:
            outputs = outputs[active_pos_mask]
            targets = targets[active_pos_mask]
        if mask is None or mask.any():
            loss = self.prediction_criterion(outputs, targets)
            self.log("val_prediction_loss",loss)
        else:
            print(f"Fully masked for batch {batch_idx}. No gradients to compute.")
            loss = sum(param.sum() * 0.0 for param in self.parameters() if param.requires_grad)
        if self.residual_activation_loss_weight!=0 and id(self.layers[-1]) in self.hooked_activations:
            residual_activations = self.hooked_activations[id(self.layers[-1])]
            residual_activations_loss = self.activation_criterion(residual_activations)
            loss = loss + self.residual_activation_loss_weight * residual_activations_loss
            self.log("val_residual_activations_loss",residual_activations_loss)
        if self.seq_only_loss_weight!=0 and id(self.seq_output_head[-1]) in self.hooked_activations:
            seq_only_predictions = self.hooked_activations[id(self.seq_output_head[-1])]
            if mask is not None:
                seq_only_predictions = seq_only_predictions[mask & active_pos_mask]
            else:
                seq_only_predictions = seq_only_predictions[active_pos_mask]
            if mask is None or mask.any():
                seq_only_prediction_loss = self.seq_only_prediction_criterion(seq_only_predictions,targets)
                loss = loss + self.seq_only_loss_weight * seq_only_prediction_loss
                self.log("val_seq_only_prediction_loss",seq_only_prediction_loss)
        self.log("val_loss", loss)
        self.hooked_activations.clear()
        return loss
        
    def test_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)
        targets = self.trim_targets(inputs,targets)
        if not self.regression:
            targets = (targets>self.label_threshold_cts).float()
        if self.peak_subset_threshold:
            active_pos_mask = (targets > self.peak_subset_threshold).any(dim=1)
            fraction_true = active_pos_mask.float().mean().item()
            # self.log("test_sites",fraction_true)
        else:
            active_pos_mask = torch.full_like(targets, True, dtype=torch.bool)
        if mask is not None:
            mask = self.trim_targets(inputs,mask)
            outputs = outputs[mask & active_pos_mask]
            targets = targets[mask & active_pos_mask]
        else:
            outputs = outputs[active_pos_mask]
            targets = targets[active_pos_mask]
        if mask is None or mask.any():
            loss = self.prediction_criterion(outputs, targets)
        else:
            print(f"Fully masked for batch {batch_idx}. No gradients to compute.")
            loss = sum(param.sum() * 0.0 for param in self.parameters() if param.requires_grad)
        if self.residual_activation_loss_weight!=0 and id(self.layers[-1]) in self.hooked_activations:
            residual_activations = self.hooked_activations[id(self.layers[-1])]
            loss = loss + self.residual_activation_loss_weight * self.activation_criterion(residual_activations)
        if self.seq_only_loss_weight!=0 and id(self.seq_output_head[-1]) in self.hooked_activations:
            seq_only_predictions = self.hooked_activations[id(self.seq_output_head[-1])]
            if mask is not None:
                seq_only_predictions = seq_only_predictions[mask & active_pos_mask]
            else:
                seq_only_predictions = seq_only_predictions[active_pos_mask]
            if mask is None or mask.any():
                seq_only_prediction_loss = self.seq_only_prediction_criterion(seq_only_predictions,targets)
                loss = loss + self.seq_only_loss_weight * seq_only_prediction_loss
        self.hooked_activations.clear()
        return loss

    def predict_step(self, batch, batch_idx):
        inputs, targets, mask = batch
        outputs = self(inputs)
        self.hooked_activations.clear()
        return outputs
    
    def configure_optimizers(self):
        # Define parameter groups based on the layer's weight decay
        param_groups = []
        
        for attr_name in dir(self):  # Iterate over all attributes of the model
            if attr_name.startswith("_"):  # Skip private attributes
                continue
            
            module = getattr(self, attr_name)
            
            if isinstance(module, nn.Module) and attr_name!='pretrained_seq_model':  # Check if it's an nn.Module or nn.ModuleList. Exclude pretrained model
                params = [p for p in module.parameters()]  # Include all params to make trainability switchable
                if params:  # Only add if there are trainable params
                    weight_decay = getattr(module, 'weight_decay', 0)
                    param_groups.append({'params': params, 'weight_decay': weight_decay})
        
        if not param_groups:
            raise ValueError("No trainable parameters found. Ensure at least one module has trainable parameters.")
        else:
            return [self.optimizer_class(param_groups)]
    
    def get_layer(self, layer_name):
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in the model")
        
    def set_requires_grad(self, grad_dict):
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
    
    # def on_train_epoch_start(self,*args,**kwargs):
    #     if self.current_epoch in self.train_stages:
    #         stage_dict = self.train_stages[self.current_epoch]
    #         if 'mode' in stage_dict:
    #             self.mode = stage_dict['mode']
    #         if 'grad_dict' in stage_dict:
    #             self.set_module_requires_grad(stage_dict['grad_dict'])
    #             # print("resetting optimizers maybe?")
    #             # self.trainer.strategy.setup_optimizers(self.trainer)
    #             # trainer.optimizers = self.configure_optimizers()

    def on_train_start(self):
        if self.start_epoch > 0:
            # this lets us start at a specified epoch (relevant especially for epoch-based stage-wise training)
            self.trainer.fit_loop.epoch_progress.current.completed = self.start_epoch
            self.trainer.fit_loop.epoch_progress.current.processed = self.start_epoch            
    
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
        if self.mode in ('pretrained-embeddings-only','residual-w/-pretrained-embeddings'):
            x,_ = inputs
        else:
            x = inputs
        inputs_length = x.shape[2] - (2*self.crop_off_sequence if self.crop_off_sequence else 0)
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
    
        for layer in self.layers+self.merged_output_head:
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