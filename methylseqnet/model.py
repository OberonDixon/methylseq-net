import importlib
from collections import defaultdict
import inspect
import math
import logging
import warnings
from typing import Callable, Any, Set, Type, Literal
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gin
import lightning as L
import gc
import torchmetrics
import zipfile
import pandas as pd
from io import StringIO

from methylseqnet.layers import ActivationCapture
from methylseqnet.losses import MaskedLoss, PoissonLoss, LogL1Loss, BCELoss, OrthogonalityLoss, MSELoss
from methylseqnet.pretrained import basenji2_pytorch, borzoi_pytorch
from methylseqnet.tensor_ops import FEATURE_MODULATION_OPS
from methylseqnet.hub import release_checkpoint_path, DEFAULT_REPO, DEFAULT_BASE, DEFAULT_VERSION

gin.register(nn.Softplus)
gin.register(nn.Sigmoid)
gin.register(nn.Hardtanh)
gin.register(nn.Conv1d)

gin.external_configurable(optim.Adam, module='torch.optim')
gin.external_configurable(optim.AdamW, module='torch.optim')
gin.external_configurable(optim.SGD, module='torch.optim')

@gin.configurable
class ConditionedSeqNN(L.LightningModule):
    def __init__(
        self, 

        # Input encoders
        sequence_encoder: list[Type[nn.Module] | Callable] = [],
        conditioning_state_encoder: list[Type[nn.Module] | Callable] = [],
        concat_pretrained_embeddings_at: dict[int: int]={},

        # Conditioning head
        embeddings_to_unconditional_seq_rep: list[Type[nn.Module] | Callable] = [],
        embeddings_to_conditional_seq_rep: list[Type[nn.Module] | Callable] = [],
        embeddings_to_conditioning_state_rep: list[Type[nn.Module] | Callable] = [],
        output_head: list[Type[nn.Module] | Callable] = [],

        conditioning_operation: str = 'multiply',
        true_conditioning_state_weight: float = 1.0,
        interpolate_conditioning_state_location: str = 'representation', 

        # Cropping
        crop_off_conditioning_input: int = 0,
        crop_off_output: int = 0,

        # Task details
        out_tracks: int | None = None,
        total_stride: int = 128,
        data_types_subset: int | str = None,
        regression: bool = True,
        label_threshold_cts: int | None = None,
        
        # Training stages
        train_stages: dict[str, dict[str: Any]] = {},

        # Optimization config
        optimizer_class: Type[torch.optim.Optimizer] = optim.Adam,

        conditioning_state_rep_loss_weight: float = 0.0,
        seq_reps_orthogonality_loss_weight: float = 0.0,

        prediction_criterion: Type[MaskedLoss] = PoissonLoss,
        conditioning_state_rep_criterion: Type[MaskedLoss] = BCELoss,
        seq_reps_orthogonality_criterion: Type[MaskedLoss] = OrthogonalityLoss,
        seq_reps_to_conditioning_state_criterion: Type[MaskedLoss] = MSELoss,

        # Predict time config
        supplemental_predict_outputs: Set = set(),
    ):
        """
        ConditionedSeqNN forward pass diagram. 

        DNA sequence:         ┌────► [unconditional_seq_rep] ──────────┐       output:
        (N,4,L)               │                                        ▼       (N,out_tracks,L)
        [sequence_encoder] ─┬─┤                                      concat ─► [output_head]
                            │ │                                        ▲
                            │ └────► [conditional_seq_rep] ──► ⊗ ──────┘
                            │                                  ▲ [conditioning_operation]
                            │    ┌─► [conditioning_state_rep] ─┘      pseudobatch across
        conditioning_state: │    │                                    conditioning states
        (N,states,C,L)      ▼    │
        [conditioning_state_encoder] (optional; can be imputed from sequence)
        """
        
        super().__init__()
        # Check config validity
        if true_conditioning_state_weight>1 or true_conditioning_state_weight<0:
            raise ValueError("ConditionedSeqNN true_conditioning_state_weight must be between 0 and 1.")
        if out_tracks is None:
            raise ValueError("ConditionedSeqNN out_tracks must be specified.")
        if not regression:
            if label_threshold_cts is None:
                raise ValueError("ConditionedSeqNN label_threshold_cts must be specified for classification tasks.")
            elif label_threshold_cts < 0:
                raise ValueError("ConditionedSeqNN label_threshold_cts must be non-negative.")
        
        # Set up encoders
        self.sequence_encoder = self._sequential_from_constructors(sequence_encoder)
        self.conditioning_state_encoder = self._modulelist_from_constructors(conditioning_state_encoder)
        if len(self.conditioning_state_encoder)>0:
            self.concat_pretrained_embeddings_at = { # adjust negative indices to positive
                (i if i >= 0 else len(conditioning_state_encoder) + i): v
                for i, v in concat_pretrained_embeddings_at.items()
            }
        else:
            if concat_pretrained_embeddings_at:
                raise ValueError("ConditionedSeqNN concat_pretrained_embeddings_at provided but no conditioning_state_encoder, nowhere to concatenate.")

        # Set up conditioning head modules
        self.embeddings_to_unconditional_seq_rep = self._sequential_from_constructors(embeddings_to_unconditional_seq_rep)
        self.embeddings_to_conditional_seq_rep = self._sequential_from_constructors(embeddings_to_conditional_seq_rep)
        self.embeddings_to_conditioning_state_rep = self._sequential_from_constructors(embeddings_to_conditioning_state_rep)
        self.output_head = self._sequential_from_constructors(output_head)

        # Conditioning logic configuration
        self.conditioning_operation = conditioning_operation
        self.true_conditioning_state_weight = true_conditioning_state_weight
        self.interpolate_conditioning_state_location = interpolate_conditioning_state_location

        # Activation captures for auxiliary losses and predict time outputs
        self.capture_true_conditioning_state_rep = ActivationCapture()
        self.capture_imputed_conditioning_state_rep = ActivationCapture()
        self.capture_unconditional_seq_rep = ActivationCapture()
        self.capture_conditional_seq_rep = ActivationCapture()

        # Set up cropping
        self.crop_off_conditioning_input = crop_off_conditioning_input
        self.crop_off_output = crop_off_output
        
        # Set task details
        self.out_tracks = out_tracks
        self.total_stride = total_stride
        self.data_types_subset = data_types_subset if data_types_subset is None else set(data_types_subset)
        self.regression = regression
        self.label_threshold_cts = label_threshold_cts
        
        # Training stages
        self.train_stages = train_stages
        self.current_stage_name = None

        self.hooked_activations = {}
        self.conditioning_state_rep_loss_weight = conditioning_state_rep_loss_weight
        self.seq_reps_orthogonality_loss_weight = seq_reps_orthogonality_loss_weight

        self.capture_imputed_conditioning_state_rep.register_forward_hook(self._capture_activations_hook)
        self.capture_true_conditioning_state_rep.register_forward_hook(self._capture_activations_hook)
        self.capture_unconditional_seq_rep.register_forward_hook(self._capture_activations_hook)
        self.capture_conditional_seq_rep.register_forward_hook(self._capture_activations_hook)

        self.io_mappings_str = ''
        
        self.prediction_criterion = prediction_criterion()
        self.conditioning_state_rep_criterion = conditioning_state_rep_criterion()
        self.seq_reps_orthogonality_criterion = seq_reps_orthogonality_criterion()
        self.seq_reps_to_conditioning_state_criterion = seq_reps_to_conditioning_state_criterion()
        self.optimizer_class = optimizer_class
        self.start_epoch = 0

        self.supplemental_predict_outputs = supplemental_predict_outputs
        if 'cpg_density' in self.supplemental_predict_outputs:
            warnings.warn("CpG density supplemental output is hardcoded to 128bp bins with no cropping.")
        self.hooked_supplemental_outputs = {}

    def _modulelist_from_constructors(self, modulelist_constructors):
        modulelist = nn.ModuleList()
        for module_constructor in modulelist_constructors:
            try:
                modulelist.append(module_constructor(pad=True))
            except TypeError:
                modulelist.append(module_constructor())
        return modulelist

    def _sequential_from_constructors(self, modulelist_constructors):
        return nn.Sequential(*self._modulelist_from_constructors(modulelist_constructors))

    def forward(self, sequence, conditioning_state, dataset_key="all"):
        if conditioning_state.shape[1] > 1 and conditioning_state.shape[1]!=self.num_states[dataset_key]:
            raise ValueError(
                f"ConditionedSeqNN residual forward received conditioning_state input with {conditioning_state.shape[1]} cell types."+
                f"Expected either 1 (shared conditioning_state) or {self.num_states[dataset_key] if dataset_key in self.num_states else self.num_states}."
                )
        assert sequence.shape[0] == conditioning_state.shape[0], f"Batch size mismatch between sequence and conditioning_state: {sequence.shape[0]} vs {conditioning_state.shape[0]}"
        assert sequence.shape[-1] == conditioning_state.shape[-1], f"Sequence length mismatch between sequence and conditioning_state: {sequence.shape[-1]} vs {conditioning_state.shape[-1]}"
        assert dataset_key in self.dataset_keys, f"Dataset key {dataset_key} not found in model dataset_keys {self.dataset_keys}. Used 'all' to run all tasks. Check if io_mappings_str is set correctly and contains the dataset_key."
        embeddings = self._sequence_encoder_forward(sequence)
        x = self._conditioning_forward(sequence, conditioning_state, embeddings, dataset_key)
        return x
    
    def training_step(self,batch,batch_idx):
        return self._shared_step(batch,batch_idx,"train") 

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,"val")  
        
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,None)  

    def predict_step(self, batch, batch_idx):
        sequence_all_variants = batch['sequence']
        conditioning_state_all_variants = batch['conditioning_state']
        specifiers = batch['specifier']
        io_mappings_df = self.get_io_mappings_df()
        dataset_key = batch['dataset_key'][0]
        output_tracks_slice = (
            io_mappings_df['absolute_channel'].tolist()
            if dataset_key=='all'
            else io_mappings_df[io_mappings_df['dataset_key']==dataset_key]['absolute_channel'].tolist()
        )

        outputs_list = []
        for variant_idx in range(sequence_all_variants.shape[1]):
            sequence = sequence_all_variants[:,variant_idx]
            conditioning_state = conditioning_state_all_variants[:,variant_idx]
            outputs = self(sequence, conditioning_state, dataset_key)
            if outputs.shape[1] > len(output_tracks_slice):
                outputs = outputs[:,output_tracks_slice,:]
            outputs_list.append(outputs.unsqueeze(1))
        self.hooked_activations.clear()
        # for predictions writing to hdf5, everything in this dictionary gets saved
        prediction_dict = {
            "predictions":torch.cat(outputs_list,dim=1),
            "specifier":specifiers,
            **self.hooked_supplemental_outputs,
        }
        self.hooked_supplemental_outputs.clear()
        return prediction_dict
    
    def _shared_step(self, batch, batch_idx, log_descriptor):
        sequence_all_variants = batch['sequence']
        conditioning_state_all_variants = batch['conditioning_state']
        targets_all_variants = self.targets_from_batch(batch)
        mask_all_variants = self.mask_from_batch(batch)

        io_mappings_df = self.get_io_mappings_df()
        dataset_key = batch['dataset_key'][0]
        output_tracks_slice = io_mappings_df[io_mappings_df['dataset_key']==dataset_key]['absolute_channel'].tolist()

        all_variants_loss_terms = []
        outputs_list = []

        dataset_log_descriptor = log_descriptor + "/" + dataset_key if log_descriptor else None
        for variant_idx in range(sequence_all_variants.shape[1]):
            sequence = sequence_all_variants[:,variant_idx]
            conditioning_state = conditioning_state_all_variants[:,variant_idx]
            targets = targets_all_variants[:,variant_idx]
            mask = mask_all_variants[:,variant_idx]

            outputs = self(sequence, conditioning_state, dataset_key)
            if outputs.shape[1] > len(output_tracks_slice):
                outputs = outputs[:,output_tracks_slice,:]
            outputs_list.append(outputs.unsqueeze(1))

            # additionally mask out tasks that aren't in the data_types_subset, if provided
            if self.data_types_subset is not None:
                subset_indices = io_mappings_df[io_mappings_df['data_type'].isin(self.data_types_subset)]['channel'].tolist()
                subset_mask = torch.zeros_like(targets,dtype=torch.bool)
                subset_mask[:,subset_indices,:] = 1
                if mask is not None:
                    mask = mask & subset_mask
                else:
                    mask = subset_mask
                
            if not self.regression:
                targets = (targets>self.label_threshold_cts).float()

            all_variants_loss_terms.extend(
                [
                    self._apply_masked_loss(
                        self.prediction_criterion,
                            (outputs,targets),
                            mask=mask,
                            weight=1,
                            log_name=f"{dataset_log_descriptor}/prediction_loss" if dataset_log_descriptor else None,
                        )
                ]
                + self._calculate_auxiliary_losses(dataset_log_descriptor, mask, targets)
            )

        loss = sum(all_variants_loss_terms)
        if log_descriptor:
            self.log(f"{log_descriptor}/loss", loss, sync_dist=True)
        self.hooked_activations.clear()
        predictions = torch.cat(outputs_list,dim=1)
        return {
            "loss":loss,
            "predictions":predictions,
        }

    def _apply_masked_loss(self,loss_fn,args,mask=None,weight=1,log_name=None):
        if weight==0 or (mask is not None and not mask.any()):
            loss = sum(param.sum() * 0.0 for param in self.parameters() if param.requires_grad)
        else:
            loss = loss_fn(*args,mask=mask)
        if log_name:
            self.log(log_name,loss,sync_dist=False)            
        return weight*loss

    def _calculate_auxiliary_losses(self, log_descriptor, mask, targets):
        auxiliary_losses = []
        if self.conditioning_state_rep_loss_weight>0:
            if id(self.capture_imputed_conditioning_state_rep) in self.hooked_activations and id(self.capture_true_conditioning_state_rep) in self.hooked_activations:
                imputed_conditioning_state_rep = self.hooked_activations[id(self.capture_imputed_conditioning_state_rep)]
                true_conditioning_state_rep = self.hooked_activations[id(self.capture_true_conditioning_state_rep)]
                auxiliary_losses.append(
                    self._apply_masked_loss(
                        self.conditioning_state_rep_criterion,
                        (imputed_conditioning_state_rep, true_conditioning_state_rep),
                        mask=None,
                        weight=self.conditioning_state_rep_loss_weight,
                        log_name=f"{log_descriptor}/conditioning_state_rep_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("conditioning_state_rep_loss not calculated; no hooked activations found.")
        if self.seq_reps_orthogonality_loss_weight>0:
            if id(self.capture_unconditional_seq_rep) in self.hooked_activations and id(self.capture_conditional_seq_rep) in self.hooked_activations:
                unconditional_seq_rep = self.hooked_activations[id(self.capture_unconditional_seq_rep)]
                conditional_seq_rep = self.hooked_activations[id(self.capture_conditional_seq_rep)]
                auxiliary_losses.append(
                    self._apply_masked_loss(
                        self.seq_reps_orthogonality_criterion,
                        (unconditional_seq_rep, conditional_seq_rep),
                        mask=None,
                        weight=self.seq_reps_orthogonality_loss_weight,
                        log_name=f"{log_descriptor}/seq_reps_orthogonality_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("seq_reps_orthogonality_loss not calculated; no hooked activations found.")
        return auxiliary_losses
    
    def configure_optimizers(self):
        # Define parameter groups based on the layer's weight decay
        param_groups = []
        
        for attr_name in dir(self):  # Iterate over all attributes of the model
            if attr_name.startswith("_"):  # Skip private attributes
                continue
            
            module = getattr(self, attr_name)
            
            if isinstance(module, nn.Module): # Check if it's an nn.Module or nn.ModuleList.
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

    def apply_current_stage(self):
        if self.train_stages:
            stage_dict = self.train_stages[self.current_stage_name]
            self.set_requires_grad(stage_dict['grad_dict'])
            self.true_conditioning_state_weight = stage_dict.get('true_conditioning_state_weight', self.true_conditioning_state_weight)
            self.interpolate_conditioning_state_location = stage_dict.get('interpolate_conditioning_state_location', self.interpolate_conditioning_state_location)
            self.conditioning_operation = stage_dict.get('conditioning_operation', self.conditioning_operation)
        
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
            else:
                raise ValueError(f"Module {attr_name} not found in the model")

    def on_train_start(self):
        if self.start_epoch > 0:
            # this lets us start at a specified epoch (relevant especially for epoch-based stage-wise training)
            self.train.fit_loop.epoch_progress.current.completed = self.start_epoch
            self.train.fit_loop.epoch_progress.current.processed = self.start_epoch      

    def on_save_checkpoint(self, checkpoint):
        checkpoint["operative_config_str"] = gin.operative_config_str()
        checkpoint["io_mappings_str"] = self.io_mappings_str

    def on_load_checkpoint(self, checkpoint):
        self.set_io_mappings(checkpoint.get("io_mappings_str",""))
     
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        # lazy imports to avoid circular dependencies while giving the gin config what it needs
        from methylseqnet import callbacks
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
    
    @classmethod
    def from_release(
        cls,
        base: Literal["borzoi-rep0"] = DEFAULT_BASE,
        version: str = DEFAULT_VERSION,
        *,
        repo_id: str = DEFAULT_REPO,
        filename: str | None = None,
        **kwargs,
    ):
        ckpt_path = release_checkpoint_path(base, version, repo_id=repo_id, filename=filename)
        return cls.load_from_checkpoint(ckpt_path, **kwargs)

    @classmethod
    def from_local(
        cls, run_id,
        *,
        checkpoints_dir,
        **kwargs,
    ):
        ckpt_path = max(Path(checkpoints_dir, run_id, "checkpoints").glob("best*.ckpt"),
                   key=lambda p: p.stat().st_mtime)
        return cls.load_from_checkpoint(ckpt_path, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        *args,
        **kwargs,
    ):
        return cls.from_release(*args,**kwargs)

    def set_io_mappings(self,io_mappings_str):
        self.io_mappings_str = io_mappings_str
        subsets = self.get_loss_subsets_from_io_mappings()
        self.apply_subsets_to_losses(subsets)
        io_mappings_df = self.get_io_mappings_df()
        self.num_states = {}
        self.cell_type_list_per_dataset = {}
        cell_type_idx_offset = 0
        for dataset_key in io_mappings_df['dataset_key'].unique():
            input_to_outputs_dict_dataset = self.get_input_to_outputs_dict(dataset_key)
            self.num_states[dataset_key] = len(input_to_outputs_dict_dataset)
            self.cell_type_list_per_dataset[dataset_key] = [cell_type_idx+cell_type_idx_offset for cell_type_idx in input_to_outputs_dict_dataset.keys()]
            cell_type_idx_offset += self.num_states[dataset_key]
        self.num_states['all'] = sum([cell_types for cell_types in self.num_states.values()])
        self.cell_type_list_per_dataset['all'] = [cell_type for cell_types in self.cell_type_list_per_dataset.values() for cell_type in cell_types]
        self.dataset_keys = set(self.num_states.keys())
        assert max(io_mappings_df['absolute_channel']) == self.out_tracks - 1, f"Absolute channel counts from io_mappings {max(io_mappings_df['absolute_channel'])+1} does not match out_tracks {self.out_tracks} passed to __init__"
    
    def get_io_mappings_df(self):
        io_mappings_df = pd.read_csv(StringIO(self.io_mappings_str),sep='\t',header=0)
        return io_mappings_df
    
    def get_input_to_outputs_dict(
        self,
        dataset_key,
        absolute_and_relative_cell_types=False,
        absolute_and_relative_channels=False,
        ):
        input_to_outputs_dict = defaultdict(list)
        for _,io_mappings_row in self.get_io_mappings_df().iterrows():
            if dataset_key == 'all':
                if absolute_and_relative_cell_types:
                    cell_type = (
                        int(io_mappings_row['cell_type']),
                        int(io_mappings_row['absolute_cell_type']),
                        )
                else:
                    cell_type = int(io_mappings_row['absolute_cell_type'])
                if absolute_and_relative_channels:
                    channel = (
                        int(io_mappings_row['channel']),
                        int(io_mappings_row['absolute_channel']),
                        )
                else:
                    channel = int(io_mappings_row['absolute_channel'])
                input_to_outputs_dict[cell_type].append(channel)
            elif io_mappings_row['dataset_key']==dataset_key:
                if absolute_and_relative_cell_types:
                    cell_type = (
                        int(io_mappings_row['cell_type']),
                        int(io_mappings_row['absolute_cell_type']),
                        )
                else:
                    cell_type = int(io_mappings_row['cell_type'])
                if absolute_and_relative_channels:
                    channel = (
                        int(io_mappings_row['channel']),
                        int(io_mappings_row['absolute_channel']),
                        )
                else:
                    channel = int(io_mappings_row['channel'])
                input_to_outputs_dict[cell_type].append(channel)
        return input_to_outputs_dict

    def apply_subsets_to_losses(self, subsets: list[list[int]]):
        """
        Find all PoissonMultinomialLoss modules in the model and assign `subsets`
        if `subsetted` is True.
        """
        for name, module in self.named_modules():
            if module.__class__.__name__ == "PoissonMultinomialLoss":
                if getattr(module, "subsetted", False):
                    module.subsets = subsets
                    logger.debug(f"Assigned subsets to PoissonMultinomialLoss module: {name}")
                else:
                    logger.debug(f"Skipped {name}: subsetted is False")
    
    def get_loss_subsets_from_io_mappings(self):
        io_mappings_df = self.get_io_mappings_df()
        subsets = []
        for data_type in io_mappings_df["data_type"].unique():
            channels = (
                io_mappings_df.loc[io_mappings_df["data_type"] == data_type, "channel"]
                .astype(int)  # ensure it's int not object
                .tolist()
            )
            subsets.append(channels)
        return subsets
    
    def _sequence_encoder_forward(self, sequence):
        if 'cpg_density' in self.supplemental_predict_outputs:
            bin_size = self.total_stride
            cpgs = (sequence[:,1,:-1].bool() & sequence[:,2,1:].bool()).float()
            cpgs = torch.nn.functional.pad(cpgs, (0, 1), value=0)
            num_bins = sequence.shape[2] // bin_size
            cpgs_binned = cpgs[:, :num_bins*bin_size].reshape(cpgs.shape[0], num_bins, bin_size)
            cpg_density = cpgs_binned.sum(dim=2) / bin_size
            self.hooked_supplemental_outputs['cpg_density'] = self.crop_targets(cpg_density)
        embeddings = self.sequence_encoder(sequence)
        return embeddings

    def _conditioning_forward(self, sequence, conditioning_state, embeddings, dataset_key):
        """
        Get conditioning state representation from conditioning_state_encoder, then combine with sequence representations and apply output head to get final predictions.
        """
        unconditional_seq_rep = self._embeddings_to_unconditional_seq_rep_forward(embeddings)
        conditional_seq_rep = self._embeddings_to_conditional_seq_rep_forward(embeddings)
        true_conditioning_state_rep = self._conditioning_state_encoder_forward(sequence, conditioning_state, embeddings, dataset_key)
        # TODO: add shape assertions for the representations to make sure they are what we expect
        if math.isclose(self.true_conditioning_state_weight,1.0) and self.conditioning_state_rep_loss_weight==0:
            imputed_conditioning_state_rep = torch.zeros_like(true_conditioning_state_rep)
        else:
            imputed_conditioning_state_rep = self._embeddings_to_conditioning_state_rep_forward(embeddings, dataset_key)
        if "sequence" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['sequence'] = sequence
        if "conditioning_state" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['conditioning_state'] = conditioning_state
        if "sequence_embedding" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['sequence_embedding'] = embeddings
        if "unconditional_seq_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['unconditional_seq_rep'] = self.crop_outputs(unconditional_seq_rep)
        if "conditional_seq_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['conditional_seq_rep'] = self.crop_outputs(conditional_seq_rep)
        if "true_conditioning_state_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['true_conditioning_state_rep'] = self.crop_outputs(true_conditioning_state_rep)
        if "imputed_conditioning_state_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['imputed_conditioning_state_rep'] = self.crop_outputs(imputed_conditioning_state_rep)
        match self.interpolate_conditioning_state_location:
            case 'output':
                if math.isclose(self.true_conditioning_state_weight,1.0):
                    return self._split_reps_to_output_forward(unconditional_seq_rep, conditional_seq_rep, true_conditioning_state_rep, dataset_key)
                elif math.isclose(self.true_conditioning_state_weight,0.0):
                    return self._split_reps_to_output_forward(unconditional_seq_rep, conditional_seq_rep, imputed_conditioning_state_rep, dataset_key)
                else:
                    x_true_component = self.true_conditioning_state_weight * self._split_reps_to_output_forward(unconditional_seq_rep, conditional_seq_rep, true_conditioning_state_rep, dataset_key)
                    x_imputed_component = (1 - self.true_conditioning_state_weight) * self._split_reps_to_output_forward(unconditional_seq_rep, conditional_seq_rep, imputed_conditioning_state_rep, dataset_key)
                    return x_true_component + x_imputed_component
            case 'representation':
                interpolated_rep = self.true_conditioning_state_weight * true_conditioning_state_rep + (1 - self.true_conditioning_state_weight) * imputed_conditioning_state_rep
                return self._split_reps_to_output_forward(unconditional_seq_rep, conditional_seq_rep, interpolated_rep, dataset_key)
            case _:
                raise NotImplementedError(f"interpolate_methyl_reps_location={self.interpolate_conditioning_state_location} not implemented.")
        
    def _embeddings_to_unconditional_seq_rep_forward(self, embeddings):
        seq_rep = self.embeddings_to_unconditional_seq_rep(embeddings)
        self.capture_unconditional_seq_rep(seq_rep)
        return seq_rep

    def _embeddings_to_conditional_seq_rep_forward(self, embeddings):
        seq_rep = self.embeddings_to_conditional_seq_rep(embeddings)
        self.capture_conditional_seq_rep(seq_rep)
        return seq_rep

    def _embeddings_to_conditioning_state_rep_forward(self, embeddings, dataset_key):
        conditioning_state_rep = self.embeddings_to_conditioning_state_rep(embeddings)
        assert conditioning_state_rep.shape[1] % self.num_states['all'] == 0, f"Imputed conditioning state representation channels {conditioning_state_rep.shape[1]} not divisible by composite dataset num_states {self.num_states['all']}."
        conditioning_state_rep = conditioning_state_rep.view(conditioning_state_rep.shape[0],self.num_states['all'],-1,conditioning_state_rep.shape[2])
        conditioning_state_rep_sliced = conditioning_state_rep[:,self.cell_type_list_per_dataset[dataset_key],:,:]
        self.capture_imputed_conditioning_state_rep(conditioning_state_rep_sliced)
        return conditioning_state_rep_sliced

    def _conditioning_state_encoder_forward(self, sequence, conditioning_state, embeddings, dataset_key):
        """
        Calculate the conditioning state representation for each conditioning state (cell type) and return a combined
        representation with a cell type dimension. The conditioning_state_encoder processes each conditioning state
        separately (potentially with concatenated pretrained embeddings internally). 

        This should always return a representation of shape (N,num_states,rep_dim,L')
        Note: input_to_conditional_state_rep must encode from (N,3,L) to (N,1,rep_dim,L') per cell type

        Pseudobatching details: here we treat each conditioning state (cell type) as a separate example in a pseudobatch,
        and then reshape back to combine the cell types into a single batch dimension after encoding. This allows the
        conditioning_state_encoder to process each conditioning state separately while still benefiting from batch processing.
        If there is only one conditioning state (cell type), this reduces to a standard batch processing without pseudobatching.
        """
        # TODO: add shape assertions and potentially merge pseudobatch logic with split_reps_to_output_forward
        x_conditioning_pseudobatch_list = []
        for cell_type_idx in range(conditioning_state.shape[1]):
            x_methyl = torch.cat(
                [
                    sequence,
                    conditioning_state[:,cell_type_idx],
                ],
                dim=1,
            )
            if self.crop_off_conditioning_input:
                x_methyl = x_methyl[:,:,self.crop_off_conditioning_input:-self.crop_off_conditioning_input]
            x_conditioning_pseudobatch_list.append(x_methyl)
        pseudobatch_scaleup = len(x_conditioning_pseudobatch_list)
        x_conditioning_pseudobatch = torch.cat(x_conditioning_pseudobatch_list, dim=0)
        for layer_index, layer in enumerate(self.conditioning_state_encoder):
            if layer_index in self.concat_pretrained_embeddings_at:
                rbs = self.concat_pretrained_embeddings_at[layer_index]
                x_conditioning_pseudobatch = self._concat_pretrained_embeddings(
                    embeddings, 
                    rbs, 
                    x_conditioning_pseudobatch, 
                    embeddings_pseudobatch_scaleup=pseudobatch_scaleup)
            x_conditioning_pseudobatch = layer(x_conditioning_pseudobatch)
        batch_size = sequence.size(0)
        
        if conditioning_state.shape[1]>1:
            x_conditioning_allchannels = x_conditioning_pseudobatch.new_zeros(batch_size, self.num_states[dataset_key], *x_conditioning_pseudobatch.shape[1:])
            for cell_type_idx, cell_type in enumerate(self.get_input_to_outputs_dict(dataset_key).keys()):
                start = cell_type_idx*batch_size
                end = (cell_type_idx+1)*batch_size
                x_cell_type = x_conditioning_pseudobatch[start:end]
                x_conditioning_allchannels[:, cell_type, :, :] = x_cell_type
        else:
            x_conditioning_allchannels = x_conditioning_pseudobatch.unsqueeze(1)
        assert x_conditioning_allchannels.shape[1] in (0, 1, self.num_states[dataset_key]), f"Conditioning state representation cell type dimension {x_conditioning_allchannels.shape[1]} is not 1 and does not match expected num_states {self.num_states[dataset_key]} for dataset {dataset_key}."
        x_conditioning_allchannels = self.capture_true_conditioning_state_rep(x_conditioning_allchannels)
        return x_conditioning_allchannels

    def _split_reps_to_output_forward(self, unconditional_seq_rep, conditional_seq_rep, conditioning_state_rep, dataset_key):
        """
        Combine unconditional and conditioned sequence representations after applying conditioning states

        Pseudobatching details: here we treat each conditioning state (cell type) as a separate example in a pseudobatch, and then reshape back to
        combine the cell types into a single batch dimension after applying the conditioning operation and output head. This allows the model to apply
        the conditioning operation separately for each conditioning state while still benefiting from batch processing. If there is only one conditioning
        state (cell type), this reduces to a standard batch processing without pseudobatching.

        Args:
            unconditional_seq_rep: (N, C_indep, L)
            conditional_seq_rep: (N, C_dep, L)
            conditioning_state_rep: (N, num_states, C_methyl, L)
            dataset_key: which dataset is being processed (to select output channels)
        """
        assert unconditional_seq_rep.shape[0] == conditional_seq_rep.shape[0] == conditioning_state_rep.shape[0], f"Batch size mismatch among representations: {unconditional_seq_rep.shape[0]}, {conditional_seq_rep.shape[0]}, {conditioning_state_rep.shape[0]}"
        x_methylseq_pseudobatch_list = []
        for cell_type_idx in range(conditioning_state_rep.shape[1]):
            celltype_conditioning_state_rep = conditioning_state_rep[:, cell_type_idx, :, :]
            conditional_seq_rep_celltype = FEATURE_MODULATION_OPS[self.conditioning_operation](
                features = conditional_seq_rep,
                modulator = celltype_conditioning_state_rep,
                )
            x_methylseq_rep = torch.cat([unconditional_seq_rep, conditional_seq_rep_celltype], dim=1)
            x_methylseq_pseudobatch_list.append(x_methylseq_rep)
        pseudobatch_scaleup = len(x_methylseq_pseudobatch_list)
        x_methylseq_pseudobatch = torch.cat(x_methylseq_pseudobatch_list, dim=0)
        x_methylseq_pseudobatch = self.output_head(x_methylseq_pseudobatch)
        if conditioning_state_rep.shape[1]==1:
            x_output_allchannels = x_methylseq_pseudobatch
        else:
            assert x_methylseq_pseudobatch.shape[1] == self.out_tracks, f"output_head output channels {x_methylseq_pseudobatch.shape[1]} does not match out_tracks {self.out_tracks}"
            x_output_allchannels = x_methylseq_pseudobatch.new_zeros(unconditional_seq_rep.size(0), self.out_tracks, x_methylseq_pseudobatch.size(2))
            batch_size = unconditional_seq_rep.size(0)
            for cell_type_idx, (cell_type, channel_tuples) in enumerate(self.get_input_to_outputs_dict(dataset_key,absolute_and_relative_channels=True).items()):
                start = cell_type_idx*batch_size
                end = (cell_type_idx+1)*batch_size
                x_cell_type = x_methylseq_pseudobatch[start:end]
                for relative_task_index, absolute_task_index in channel_tuples:
                    x_output_allchannels[:, absolute_task_index:absolute_task_index+1, :] = x_cell_type[:, absolute_task_index:absolute_task_index+1, :]
        if self.crop_off_output:
            x_output_allchannels = x_output_allchannels[:,:,self.crop_off_output:-self.crop_off_output]
        assert x_output_allchannels.shape[1] == self.out_tracks, f"Final output channels {x_output_allchannels.shape[1]} does not match out_tracks {self.out_tracks}"
        assert x_output_allchannels.shape[0] == unconditional_seq_rep.shape[0], f"Final output batch size {x_output_allchannels.shape[0]} does not match input batch size {unconditional_seq_rep.shape[0]}"
        return x_output_allchannels
    
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

    def targets_from_batch(self, batch):
        return self.crop_targets(batch['target'])

    def mask_from_batch(self, batch):
        return self.crop_targets(batch['mask'])
    
    def crop_targets(self, targets):
        crop_off_targets = self.crop_off_output + (self.crop_off_conditioning_input // self.total_stride)
        if crop_off_targets > 0:
            targets_cropped = targets[..., crop_off_targets:-crop_off_targets]
        else:
            targets_cropped = targets
        return targets_cropped

    def crop_outputs(self, outputs):
        crop_off_outputs = self.crop_off_output
        if crop_off_outputs > 0:
            outputs_cropped = outputs[..., crop_off_outputs:-crop_off_outputs]
        else:
            outputs_cropped = outputs
        return outputs_cropped

    def crop_sequence_match_output(self, sequence):
        crop_off_sequence = self.crop_off_conditioning_input + self.total_stride*self.crop_off_output
        if crop_off_sequence > 0:
            sequence_cropped = sequence[..., crop_off_sequence:-crop_off_sequence]
        else:
            sequence_cropped = sequence
        return sequence_cropped