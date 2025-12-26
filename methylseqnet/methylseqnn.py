import importlib
from collections import defaultdict
import inspect
import math
import logging
import warnings
logger = logging.getLogger(__name__)

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
gin.register(nn.Sigmoid)
gin.register(nn.Hardtanh)
gin.register(nn.Conv1d)

@gin.configurable
class MethylSeqNN(L.LightningModule):
    def __init__(
        self, 

        # Task details
        out_tracks=None,
        data_types_subset=None,
        regression=True,
        label_threshold_cts=5,
        
        # Pretrained model
        seq_input_head=None,
        seq_output_head=None,  
        pretrained_seq_model_generator=None,
        pretrained_seq_model_weights=None,
        concat_pretrained_embeddings_at={},

        # Residual model
        layers=None,
        model_merge_operation='multiply',
        merged_output_head=None,

        # Factorizer for pretrained model
        embeddings_to_methyl_rep=None,
        embeddings_to_methyl_indep_seq_rep=None,
        embeddings_to_methyl_dep_seq_rep=None,
        methyl_indep_seq_rep_probe=None,
        methyl_dep_seq_rep_probe=None,
        input_to_methyl_rep=None,
        true_methyl_rep_weight=0.5,
        interpolate_methyl_reps_location='rep',
        factorized_reps_to_output=None,
        factorized_reps_to_output_submodel_per_task=True,
        factorized_reps_to_output_submodels_shared=False,

        # Cropping / padding behavior
        pad_all_layers=False,
        crop_off_sequence=None,
        crop_off_final=None, # consider adjusted this name to be more clearly about how much is cropped off. Also, can't be zero??
        
        # Training schedule and stage-specific config
        train_stages={},

        # Optimizer config
        optimizer_class=AdamOptimizer,
        residual_activation_loss_weight=0,
        seq_only_loss_weight=0,
        methyl_rep_loss_weight=0,
        seq_reps_orthogonality_loss_weight=0,
        methyl_indep_seq_rep_probe_upstream_grad_scale=0,
        methyl_dep_seq_rep_probe_upstream_grad_scale=0,
        prediction_criterion=PoissonLoss,
        seq_only_prediction_criterion=PoissonLoss,
        activation_criterion=LogL1Loss,
        methyl_rep_criterion=BCELoss,
        seq_reps_orthogonality_criterion=OrthogonalityLoss,
        seq_reps_to_methyl_criterion=MSELoss,

        # Predict time config
        supplemental_predict_outputs=set(),
    ):
        """
        Args:
            Task details:
                - out_tracks: the number of output tracks (channels) for the model.
                - data_types_subset: a list of data types (str) to subset the loss and metrics to. If None, use all.
                - regression: if True, treat as regression problem; if False, treat as classification problem.
                    In the classification case, dataloader still provides counts but a threshold is applied.
                - label_threshold_cts: threshold in counts above which a label is considered positive (for non-regression tasks)
            Pretrained model:
                - seq_input_head: a list of nn.Modules that run sequentially before the pretrained_seq_model
                - seq_output_head: a list of nn.Modules that run sequentially after the pretrained_seq_model
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
            Residual model:
                - layers: a list of nn.Modules that run sequentially to form the seq+methyl model
                - model_merge_operation: how to combine the pretrained and residual models. Options:
                    - multiply: element-wise multiplication
                    - log_multiply: element-wise multiplication on a log scale
                    - tanh_log_multiply: element-wise multiplication on a log scale with tanh normalization of residuals
                    - add: element-wise addition
                    - mx+b: element-wise softplus(m) * softplus(x+b) where m is first n res channels, b is second n res channels, n is x.shape[1]
                - merged_output_head: a list of nn.Modules that run sequentially after the merging of the pretrained and residual models
            Factorizer for pretrained model
                - embeddings_to_methyl_rep: a list of nn.Modules that run sequentially to convert pretrained embeddings to methyl representation
                - embeddings_to_methyl_indep_seq_rep: a list of nn.Modules that run sequentially to convert pretrained embeddings to methylation-independent sequence representation
                - embeddings_to_methyl_dep_seq_rep: a list of nn.Modules that run sequentially to convert pretrained embeddings to methylation-dependent sequence representation
                - input_to_methyl_rep: a list of nn.Modules that run sequentially to convert methylseq input to methyl representation
                - true_methyl_rep_weight: weight (0-1) for the true methyl representation when interpolating with the predicted methyl representation
                - interpolate_methyl_reps_location: where to do the interpolation of true and predicted methyl representations. Options:
                    - 'rep': interpolate between representations
                    - 'output': interpolate between outputs
                - factorized_reps_to_output: a list of nn.Modules that run sequentially to convert the combined methyl and sequence representations to output
                - factorized_reps_to_output_submodel_per_task: if True, create a separate factorized_reps_to_output submodel for each task
            Cropping / padding behavior:
                - pad_all_layers: if True, all layers that support padding will be padded to keep input length the same as output length.
                - crop_off_sequence: if provided, crop off this many bases from each end of the input sequence before feeding to the methylseq layers model.
                - crop_off_final: if provided, crop off this many bins from each end of the final output of the model.
            Training schedule and stage-specific config:
                - train_stages: a dictionary providing at minimum a model `mode` and `epochs` count for a stage. May also provide
                    `grad_dict` to specify model submodules to train for this stage and `loss_dict` to specify loss function 
                    components.
           Optimizer config:
                - optimizer_class: the optimizer class to use. Must be a subclass of torch.optim.Optimizer. Parameters defined in config.
                - residual_activation_loss_weight: weight for an auxiliary loss on the activations of the final residual layer
                - seq_only_loss_weight: weight for an auxiliary loss on the predictions of the sequence-only model
                - methyl_rep_loss_weight: weight for an auxiliary loss on the predicted methyl representation
                - seq_reps_orthogonality_loss_weight: weight for an auxiliary loss to encourage orthogonality between
                - methyl_indep_seq_rep_probe_upstream_grad_scale: scale for gradients passing back from the methyl_indep_seq_rep_probe to the methyl_indep_seq_rep
                - methyl_dep_seq_rep_probe_upstream_grad_scale: scale for gradients passing back from the methyl_dep_seq_rep_probe to the methyl_dep_seq_rep
                - prediction_criterion: the loss function class to use for main prediction loss. Must be a subclass of nn.Module.
                - seq_only_prediction_criterion: the loss function class to use for sequence-only prediction loss. Must be a subclass of nn.Module.
                - activation_criterion: the loss function class to use for activation loss. Must be a subclass of nn.Module.
                - methyl_rep_criterion: the loss function class to use for methyl representation loss. Must be a subclass of nn.Module.
                - seq_reps_orthogonality_criterion: the loss function class to use for sequence representations orthogonality loss. Must be a subclass of nn.Module.
            Predict-time config:
                - supplemental_predict_outputs: a set of str names of additional outputs to return during predict step.
        """
        super().__init__()
        if not layers and not pretrained_seq_model_generator:
            raise ValueError("MethylSeqNN requires a defined methylseq model (self.layers) or a defined pretrained sequence model.")
        self.use_embeddings_factorization = (
            embeddings_to_methyl_rep
            or embeddings_to_methyl_indep_seq_rep
            or factorized_reps_to_output
        )
        if self.use_embeddings_factorization and not (embeddings_to_methyl_rep and embeddings_to_methyl_indep_seq_rep and input_to_methyl_rep and factorized_reps_to_output):
            raise ValueError("MethylSeqNN embeddings factorization components require all of embeddings_to_methyl_rep, embeddings_to_methyl_indep_seq_rep, input_to_methyl_rep, and factorized_reps_to_output to be defined.")
        if self.use_embeddings_factorization and not pretrained_seq_model_generator:
            raise ValueError("MethylSeqNN pretrained model factorization components require a defined pretrained sequence model.")
        if layers and self.use_embeddings_factorization:
            raise ValueError("MethylSeqNN residual model and embeddings factorization are different, mutually incompatible approaches. Define one or the other, not both.")
        if true_methyl_rep_weight>1 or true_methyl_rep_weight<0:
            raise ValueError("MethylSeqNN true_methyl_rep_weight must be between 0 and 1.")
        
        self.out_tracks = out_tracks
        self.data_types_subset = data_types_subset if data_types_subset is None else set(data_types_subset)
        self.regression = regression
        self.label_threshold_cts = label_threshold_cts
        
        self.train_stages = train_stages
        self.current_stage_name = None
        self.mode = 'full-model' #'residual-w/-pretrained-embeddings'
        self.peak_subset_threshold = 0
        self.pad_all_layers = pad_all_layers
        self.crop_off_sequence = crop_off_sequence
        self.crop_off_final = crop_off_final
        if layers:
            self.concat_pretrained_embeddings_at = { # adjust negative indices to positive
                (i if i >= 0 else len(layers) + i): v
                for i, v in concat_pretrained_embeddings_at.items()
            }
        elif input_to_methyl_rep:
            self.concat_pretrained_embeddings_at = { # adjust negative indices to positive
                (i if i >= 0 else len(input_to_methyl_rep) + i): v
                for i, v in concat_pretrained_embeddings_at.items()
            }
        else:
            if concat_pretrained_embeddings_at:
                raise ValueError("MethylSeqNN concat_pretrained_embeddings_at provided but no layers or input_to_methyl_rep defined; ignoring.")
        self.model_merge_operation = model_merge_operation
        self.operations = { # the different operations that can be used to combine pretrained and residual models. take in (res,x)
            'multiply': torch.mul,  # Element-wise multiplication
            'log_multiply': lambda res, x: torch.exp(torch.log(x + 1e-5) + res), # Element-wise multiplication on a log scale
            'tanh_log_multiply': lambda res, x: x * torch.exp(4.0 * torch.tanh(res/4.0)),
            'add': torch.add,       # Element-wise addition
            # Element-wise softplus(m) * softplus(x+b) where m is first n res channels, b is second n res channels, n is x.shape[1]
            'mx+b': lambda mb, x: (
                nn.functional.softplus(mb[:, :x.shape[1],:]) * 
                nn.functional.softplus(x + mb[:, x.shape[1]:,:])
            ),
            'keep_a': lambda a, b: a,
            'keep_b': lambda a, b: b,
        }
        # TODO: rename layers to something like residual_methylseq_model
        self.layers = nn.ModuleList()
        self.pretrained_seq_model = nn.ModuleList()
        self.seq_input_head = nn.ModuleList()
        self.seq_output_head = nn.ModuleList()
        self.merged_output_head = nn.ModuleList()
        self.capture_true_methyl_rep = ActivationCapture()
        self.capture_imputed_methyl_rep = ActivationCapture()
        self.capture_methyl_indep_seq_rep = ActivationCapture()
        self.capture_methyl_dep_seq_rep = ActivationCapture()
        if pretrained_seq_model_generator is not None:
            self.pretrained_seq_model = pretrained_seq_model_generator(pretrained_seq_model_weights)
            for param in self.pretrained_seq_model.parameters():
                param.requires_grad = False
            if seq_input_head:
                for layer in seq_input_head:
                    self.seq_input_head.append(layer())
            if seq_output_head:
                for layer in seq_output_head:
                    self.seq_output_head.append(layer())
        if layers:
            for layer in layers:
                try:
                    self.layers.append(layer(pad=self.pad_all_layers))
                except:
                    self.layers.append(layer())
        if merged_output_head:
            for layer in merged_output_head:
                try:
                    self.merged_output_head.append(layer(pad=self.pad_all_layers))
                except:
                    self.merged_output_head.append(layer()) 

        self.true_methyl_rep_weight = true_methyl_rep_weight
        self.interpolate_methyl_reps_location = interpolate_methyl_reps_location
        self.factorized_reps_to_output_submodel_per_task = factorized_reps_to_output_submodel_per_task
        self.factorized_reps_to_output_submodels_shared = factorized_reps_to_output_submodels_shared
        self.methyl_indep_seq_rep_probe = nn.ModuleList([])
        self.methyl_dep_seq_rep_probe = nn.ModuleList([])
        self.input_to_methyl_rep = nn.ModuleList([])
        self.methyl_indep_seq_rep_probe_grad_interface = GradientReversalLayer(lambda_=methyl_indep_seq_rep_probe_upstream_grad_scale)
        self.methyl_dep_seq_rep_probe_grad_interface = GradientReversalLayer(lambda_=methyl_dep_seq_rep_probe_upstream_grad_scale)
        if self.use_embeddings_factorization:
            self.embeddings_to_methyl_rep = nn.ModuleList([layer() for layer in embeddings_to_methyl_rep])
            self.embeddings_to_methyl_indep_seq_rep = nn.ModuleList([layer() for layer in embeddings_to_methyl_indep_seq_rep])
            if embeddings_to_methyl_dep_seq_rep:
                self.embeddings_to_methyl_dep_seq_rep = nn.ModuleList([layer() for layer in embeddings_to_methyl_dep_seq_rep])
            else:
                self.embeddings_to_methyl_dep_seq_rep = nn.ModuleList([])
            for layer in input_to_methyl_rep:
                try:
                    self.input_to_methyl_rep.append(layer(pad=self.pad_all_layers))
                except:
                    self.input_to_methyl_rep.append(layer())
            if methyl_indep_seq_rep_probe:
                self.methyl_indep_seq_rep_probe = nn.ModuleList([layer() for layer in methyl_indep_seq_rep_probe])
            if methyl_dep_seq_rep_probe:
                self.methyl_dep_seq_rep_probe = nn.ModuleList([layer() for layer in methyl_dep_seq_rep_probe])
            if self.factorized_reps_to_output_submodel_per_task:
                if self.factorized_reps_to_output_submodels_shared:
                    factorized_reps_to_output_submodel = self._build_modulelist_with_padding(factorized_reps_to_output)
                    self.factorized_reps_to_output = nn.ModuleDict(
                        {
                            f"factorized_reps_to_output_task{task_index}": factorized_reps_to_output_submodel 
                                for task_index in range(self.out_tracks)
                        }
                    )
                else:
                    self.factorized_reps_to_output = nn.ModuleDict(
                        {
                            f"factorized_reps_to_output_task{task_index}": self._build_modulelist_with_padding(factorized_reps_to_output) 
                                for task_index in range(self.out_tracks)
                        }
                    )
            else:
                if self.factorized_reps_to_output_submodels_shared:
                    raise ValueError("MethylSeqNN factorized_reps_to_output_submodels_shared=True requires factorized_reps_to_output_submodel_per_task=True.")
                else:
                    self.factorized_reps_to_output = self._build_modulelist_with_padding(factorized_reps_to_output) 
        else:
            self.input_to_methyl_rep = nn.ModuleList([])

        self.receptive_field,self.total_stride = self.calculate_receptive_field_and_stride()

        self.hooked_activations = {}
        self.residual_activation_loss_weight = residual_activation_loss_weight
        self.seq_only_loss_weight = seq_only_loss_weight
        self.methyl_rep_loss_weight = methyl_rep_loss_weight
        self.seq_reps_orthogonality_loss_weight = seq_reps_orthogonality_loss_weight
        if self.residual_activation_loss_weight!=0:
            self.layers[-1].register_forward_hook(self._capture_activations_hook)
        if self.seq_only_loss_weight!=0:
            self.seq_output_head[-1].register_forward_hook(self._capture_activations_hook)
        if self.methyl_rep_loss_weight!=0:
            self.capture_imputed_methyl_rep.register_forward_hook(self._capture_activations_hook)
        self.capture_true_methyl_rep.register_forward_hook(self._capture_activations_hook)
        self.capture_methyl_indep_seq_rep.register_forward_hook(self._capture_activations_hook)
        self.capture_methyl_dep_seq_rep.register_forward_hook(self._capture_activations_hook)

        self.io_mappings_str = ''
        
        self.prediction_criterion = prediction_criterion()
        self.seq_only_prediction_criterion = seq_only_prediction_criterion()
        self.activation_criterion = activation_criterion()
        self.methyl_rep_criterion = methyl_rep_criterion()
        self.seq_reps_orthogonality_criterion = seq_reps_orthogonality_criterion()
        self.seq_reps_to_methyl_criterion = seq_reps_to_methyl_criterion()
        self.optimizer_class = optimizer_class
        self.start_epoch = 0

        self.supplemental_predict_outputs = supplemental_predict_outputs
        if 'cpg_density' in self.supplemental_predict_outputs:
            warnings.warn("CpG density supplemental output is hardcoded to 128bp bins with no cropping.")
        self.hooked_supplemental_outputs = {}

    def _build_modulelist_with_padding(self, modulelist_layers):
        modulelist = nn.ModuleList()
        for layer in modulelist_layers:
            try:
                modulelist.append(layer(pad=self.pad_all_layers))
            except:
                modulelist.append(layer())
        return modulelist
    
    def forward(self, sequence, methylation, embeddings=None, dataset_key="all"):
        """
        MethylSeqNN forward supports two types of inputs:
            standard input: a single x tensor (sample, channel, position) with 4 sequence channels and 3 methylation channels
            multimethyl input: a tensor with the same 4 sequence channels but 3*n methylation channels for n cell types

        These two input types exist to support either one-to-all mapping for methylation to activity or a more efficient shared-sequence 
        differential-methylation mode for multitask training or inference. In the latter case, the larger sequence-only model needs to run only once,
        while the methylseq residual model runs many times.

        Current valid modes (Oct 28 2025):
            - full-model: run both the pretrained and residual models, including their outputs heads. Full prediction.
            - factorized-from-pretrained: run pretrained model to get embeddings, then run factorization to get outputs
            - factorized-from-pretrained-embeddings: factorization on top of cached embeddings
            - residual-only: run only the residual model; outputs may not reflect true labels.
            - pretrained-only: run only the pretrained model, including its output head. Outputs still predict true labels.
            - pretrained-embeddings-only: run only the pretrained model output head based on cached embeddings dataset.
            - residual-w/-pretrained-embeddings: run full residual model combined with pretrained output head from cached embeddings.
            - residual-only-w/-pretrained-embeddings: run only residual model, with cached embeddings available for concatenation.
        """
        if methylation.shape[1] > 1 and methylation.shape[1]!=self.num_cell_types[dataset_key]:
            raise ValueError(f"MethylSeqNN residual forward received methylation input with {methylation.shape[1]} cell types, but expected either 1 (shared methylation) or {self.num_cell_types[dataset_key] if dataset_key in self.num_cell_types else self.num_cell_types}.")
        if 'cpg_density' in self.supplemental_predict_outputs:
            bin_size = 128
            cpgs = (sequence[:,1,:-1].bool() & sequence[:,2,1:].bool()).float()
            cpgs = torch.nn.functional.pad(cpgs, (0, 1), value=0)
            num_bins = sequence.shape[2] // bin_size
            cpgs_binned = cpgs[:, :num_bins*bin_size].reshape(cpgs.shape[0], num_bins, bin_size)
            cpg_density = cpgs_binned.sum(dim=2) / bin_size
            self.hooked_supplemental_outputs['cpg_density'] = cpg_density
        match self.mode:
            # run both the pretrained and residual models, including their outputs heads. Full prediction.
            case 'full-model':
                embeddings = self._pretrained_embedder_forward(sequence)
                x_seq = self._pretrained_head_forward(embeddings)
                x_res = self._residual_forward(sequence, methylation, embeddings, dataset_key)
                x = self._merge_submodels(x_res, x_seq)
                x = self._merged_output_forward(x)
                return x
            # run pretrained model to get embeddings, then run factorization to get outputs
            case 'factorized-from-pretrained':
                embeddings = self._pretrained_embedder_forward(sequence)
                x = self._embeddings_factorization_forward(sequence, methylation, embeddings, dataset_key)
                return x
            # factorization on top of cached embeddings
            case 'factorized-from-pretrained-embeddings':
                if embeddings is None:
                    raise ValueError("MethylSeqNN forward in 'factorized-from-pretrained-embeddings' mode requires embeddings input.")
                x = self._embeddings_factorization_forward(sequence, methylation, embeddings, dataset_key)
                return x
            # run only the residual model; outputs may not reflect true labels    
            case 'residual-only':
                if self.concat_pretrained_embeddings_at:
                    embeddings = self._pretrained_embedder_forward(sequence)
                    x_res = self._residual_forward(sequence, methylation, embeddings, dataset_key)
                else:
                    x_res = self._residual_forward(sequence, methylation, embeddings, dataset_key)
                return x_res
            # run only the pretrained model, including its output head. Outputs still predict true labels.    
            case 'pretrained-only':
                embeddings = self._pretrained_embedder_forward(sequence)
                x_seq = self._pretrained_head_forward(embeddings)
                x_seq = self._merged_output_forward(x_seq)
                return x_seq
            # run only the pretrained model output head based on cached embeddings dataset.
            case 'pretrained-embeddings-only':
                if embeddings is None:
                    raise ValueError("MethylSeqNN forward in 'pretrained-embeddings-only' mode requires embeddings input.")
                x_seq = self._pretrained_head_forward(embeddings)
                x_seq = self._merged_output_forward(x_seq)
                return x_seq
            # run full residual model combined with pretrained output head from cached embeddings.
            case 'residual-w/-pretrained-embeddings':
                if embeddings is None:
                    raise ValueError("MethylSeqNN forward in 'residual-w/-pretrained-embeddings' mode requires embeddings input.")
                x_seq = self._pretrained_head_forward(embeddings)
                x_res = self._residual_forward(sequence, methylation, embeddings, dataset_key)
                x = self._merge_submodels(x_res, x_seq)
                x = self._merged_output_forward(x)
                return x
            # run only residual model, with cached embeddings available for concatenation
            case 'residual-only-w/-pretrained-embeddings':
                if embeddings is None:
                    raise ValueError("MethylSeqNN forward in 'residual-only-w/-pretrained-embeddings' mode requires embeddings input.")
                x_res = self._residual_forward(sequence, methylation, embeddings, dataset_key)
                return x_res
            case _:
                raise ValueError(f"Invalid MethylSeqNN.mode='{self.mode}'. Check documentation for valid modes.")
    
    def training_step(self,batch,batch_idx):
        return self._shared_step(batch,batch_idx,"train") 

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,"val")  
        
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,None)  

    def predict_step(self, batch, batch_idx):
        sequence_all_variants = batch['sequence']
        methylation_all_variants = batch['methylation']
        embeddings = batch.get('embeddings',None)
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
            methylation = methylation_all_variants[:,variant_idx]
            outputs = self(sequence, methylation, embeddings, dataset_key)
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
        methylation_all_variants = batch['methylation']
        embeddings = batch.get('embeddings',None)
        targets_all_variants = batch['target']
        mask_all_variants = batch.get('mask',torch.ones_like(targets_all_variants,dtype=torch.bool))

        io_mappings_df = self.get_io_mappings_df()
        dataset_key = batch['dataset_key'][0]
        output_tracks_slice = io_mappings_df[io_mappings_df['dataset_key']==dataset_key]['absolute_channel'].tolist()

        all_variants_loss_terms = []
        outputs_list = []

        dataset_log_descriptor = log_descriptor + "/" + dataset_key if log_descriptor else None
        for variant_idx in range(sequence_all_variants.shape[1]):
            sequence = sequence_all_variants[:,variant_idx]
            methylation = methylation_all_variants[:,variant_idx]
            targets = targets_all_variants[:,variant_idx]
            mask = mask_all_variants[:,variant_idx]

            outputs = self(sequence, methylation, embeddings, dataset_key)
            if outputs.shape[1] > len(output_tracks_slice):
                outputs = outputs[:,output_tracks_slice,:]
            outputs_list.append(outputs.unsqueeze(1))

            targets = self.trim_targets(sequence,targets)
            mask = self.trim_targets(sequence,mask) 
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
                
            # option to only train on sites with peaks over threshold in some cell types. If thresh is zero, keep all are active
            active_pos_mask = (targets >= self.peak_subset_threshold).any(dim=1, keepdim=True)
            fraction_true = active_pos_mask.float().mean()
            if log_descriptor and self.peak_subset_threshold>0:
                self.log(f"{dataset_log_descriptor}/sites",fraction_true,sync_dist=True)
            effective_mask = mask & active_pos_mask if mask is not None else active_pos_mask

            all_variants_loss_terms.extend(
                [
                    self._apply_masked_loss(
                        self.prediction_criterion,
                            (outputs,targets),
                            mask=effective_mask,
                            weight=1,
                            log_name=f"{dataset_log_descriptor}/prediction_loss" if dataset_log_descriptor else None,
                        )
                ]
                + self._calculate_auxiliary_losses(dataset_log_descriptor, effective_mask, targets)
                + self._calculate_probe_losses(dataset_log_descriptor, effective_mask, targets)
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

    def _calculate_auxiliary_losses(self, log_descriptor, effective_mask, targets):
        auxiliary_losses = []
        if self.residual_activation_loss_weight>0:
            if id(self.layers[-1]) in self.hooked_activations:
                residual_activations = self.hooked_activations[id(self.layers[-1])]
                auxiliary_losses.append(
                    self._apply_masked_loss(
                        self.activation_criterion,
                        (residual_activations,),
                        mask=None,
                        weight=self.residual_activation_loss_weight,
                        log_name=f"{log_descriptor}/residual_activations_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("residual_activations_loss not calculated; hooked activations not found.")
        if len(self.seq_output_head)>0 and self.seq_only_loss_weight>0:
            if id(self.seq_output_head[-1]) in self.hooked_activations:
                seq_out_activations = self.hooked_activations[id(self.seq_output_head[-1])]
                auxiliary_losses.append(
                    self._apply_masked_loss(
                        self.seq_only_prediction_criterion,
                        (seq_out_activations,targets),
                        mask=effective_mask,
                        weight=self.seq_only_loss_weight,
                        log_name=f"{log_descriptor}/seq_only_prediction_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("seq_only_loss not calculated; hooked activations not found.")
        if self.methyl_rep_loss_weight>0:
            if id(self.capture_imputed_methyl_rep) in self.hooked_activations and id(self.capture_true_methyl_rep) in self.hooked_activations:
                imputed_methyl_rep = self.hooked_activations[id(self.capture_imputed_methyl_rep)]
                true_methyl_rep = self.hooked_activations[id(self.capture_true_methyl_rep)]
                auxiliary_losses.append(
                    self._apply_masked_loss(
                        self.methyl_rep_criterion,
                        (imputed_methyl_rep, true_methyl_rep),
                        mask=None,
                        weight=self.methyl_rep_loss_weight,
                        log_name=f"{log_descriptor}/methyl_rep_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("methyl_rep_loss not calculated; no hooked activations found.")
        if self.seq_reps_orthogonality_loss_weight>0:
            if id(self.capture_methyl_indep_seq_rep) in self.hooked_activations and id(self.capture_methyl_dep_seq_rep) in self.hooked_activations:
                methyl_indep_seq_rep = self.hooked_activations[id(self.capture_methyl_indep_seq_rep)]
                methyl_dep_seq_rep = self.hooked_activations[id(self.capture_methyl_dep_seq_rep)]
                auxiliary_losses.append(
                    self._apply_masked_loss(
                        self.seq_reps_orthogonality_criterion,
                        (methyl_indep_seq_rep, methyl_dep_seq_rep),
                        mask=None,
                        weight=self.seq_reps_orthogonality_loss_weight,
                        log_name=f"{log_descriptor}/seq_reps_orthogonality_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("seq_reps_orthogonality_loss not calculated; no hooked activations found.")
        return auxiliary_losses

    def _calculate_probe_losses(self, log_descriptor, effective_mask, targets):
        probe_losses = []
        if self.methyl_indep_seq_rep_probe and self.embeddings_to_methyl_indep_seq_rep:
            if id(self.capture_methyl_indep_seq_rep) in self.hooked_activations and id(self.capture_true_methyl_rep) in self.hooked_activations:
                # cut off, invert, or otherwise modify gradients passing back into the seq rep from the probe
                x = self.methyl_indep_seq_rep_probe_grad_interface(
                    self.hooked_activations[id(self.capture_methyl_indep_seq_rep)]
                )
                true_methyl_rep = self.hooked_activations[id(self.capture_true_methyl_rep)]
                for layer in self.methyl_indep_seq_rep_probe:
                    x = layer(x)
                x = x.unsqueeze(2)
                probe_losses.append(
                    self._apply_masked_loss(
                        self.seq_reps_to_methyl_criterion,
                        (x, true_methyl_rep),
                        mask=None,
                        weight=1.0,
                        log_name=f"{log_descriptor}/methyl_indep_seq_rep_to_methyl_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("methyl_indep_seq_rep_to_methyl loss not calculated; no hooked activations found.")
        if self.methyl_dep_seq_rep_probe and self.embeddings_to_methyl_dep_seq_rep:
            if id(self.capture_methyl_dep_seq_rep) in self.hooked_activations and id(self.capture_true_methyl_rep) in self.hooked_activations:
                # cut off, invert, or otherwise modify gradients passing back into the seq rep from the probe
                x = self.methyl_dep_seq_rep_probe_grad_interface(
                    self.hooked_activations[id(self.capture_methyl_dep_seq_rep)]
                )
                true_methyl_rep = self.hooked_activations[id(self.capture_true_methyl_rep)]
                for layer in self.methyl_dep_seq_rep_probe:
                    x = layer(x)
                x = x.unsqueeze(2)
                probe_losses.append(
                    self._apply_masked_loss(
                        self.seq_reps_to_methyl_criterion,
                        (x, true_methyl_rep),
                        mask=None,
                        weight=1.0,
                        log_name=f"{log_descriptor}/methyl_dep_seq_rep_to_methyl_loss" if log_descriptor else None,
                    )
                )
            else:
                warnings.warn("methyl_dep_seq_rep_to_methyl loss not calculated; no hooked activations found.")
        return probe_losses
    
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
            self.mode = stage_dict['mode']
            self.set_requires_grad(stage_dict['grad_dict'])
            if 'peak_subset_threshold' in stage_dict:
                self.peak_subset_threshold = stage_dict['peak_subset_threshold']
            else:
                self.peak_subset_threshold = 0
            self.true_methyl_rep_weight = stage_dict.get('true_methyl_rep_weight', self.true_methyl_rep_weight)
            self.interpolate_methyl_reps_location = stage_dict.get('interpolate_methyl_reps_location', self.interpolate_methyl_reps_location)
        
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

    def on_save_checkpoint(self, checkpoint):
        checkpoint["operative_config_str"] = gin.operative_config_str()
        checkpoint["io_mappings_str"] = self.io_mappings_str
        # Store metadata for reloading the external model
        if self.pretrained_seq_model is not None:
            external_model = self.pretrained_seq_model
            checkpoint["pretrained_model_class"] = external_model.__class__.__name__
            checkpoint["pretrained_model_module"] = external_model.__class__.__module__

    def on_load_checkpoint(self, checkpoint):
        self.set_io_mappings(checkpoint.get("io_mappings_str",""))
     
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

    def trim_targets(self,sequence,targets):
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
        inputs_length = sequence.shape[-1] - (2*self.crop_off_sequence if self.crop_off_sequence else 0)
        targets_length = targets.shape[-1]
        
        if self.layers or self.input_to_methyl_rep:
            if not self.pad_all_layers:
                network_outputs_length = (inputs_length - self.receptive_field+self.total_stride)//self.total_stride
            else:
                network_outputs_length = inputs_length//self.total_stride
    
            trim_off_targets = 2*self.crop_off_final + targets_length - network_outputs_length
        else:
            trim_off_targets = 2*self.crop_off_final
        
        if trim_off_targets>1:
            return targets[..., trim_off_targets // 2:-trim_off_targets // 2]
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
    
        for layer in self.layers+self.merged_output_head+self.input_to_methyl_rep:
            kernel_size = getattr(layer, 'kernel_size', 1)
            if isinstance(kernel_size,tuple):
                kernel_size = kernel_size[0]
            pool_size = getattr(layer, 'pool_size', 1)
            if isinstance(pool_size,tuple):
                pool_size = pool_size[0]
            stride = getattr(layer, 'stride', 1)
            if isinstance(stride,tuple):
                stride = stride[0]
            dilation = getattr(layer, 'dilation', 1)
            if isinstance(dilation,tuple):
                dilation = dilation[0]
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

    def set_io_mappings(self,io_mappings_str):
        self.io_mappings_str = io_mappings_str
        subsets = self.get_loss_subsets_from_io_mappings()
        self.apply_subsets_to_losses(subsets)
        io_mappings_df = self.get_io_mappings_df()
        self.num_cell_types = {}
        self.cell_type_list_per_dataset = {}
        cell_type_idx_offset = 0
        for dataset_key in io_mappings_df['dataset_key'].unique():
            input_to_outputs_dict_dataset = self.get_input_to_outputs_dict(dataset_key)
            self.num_cell_types[dataset_key] = len(input_to_outputs_dict_dataset)
            self.cell_type_list_per_dataset[dataset_key] = [cell_type_idx+cell_type_idx_offset for cell_type_idx in input_to_outputs_dict_dataset.keys()]
            cell_type_idx_offset += self.num_cell_types[dataset_key]
        self.num_cell_types['all'] = sum([cell_types for cell_types in self.num_cell_types.values()])
        self.cell_type_list_per_dataset['all'] = [cell_type for cell_types in self.cell_type_list_per_dataset.values() for cell_type in cell_types]
    
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

    def _split_multidataset_input(self, x, split_mode='(methylseq_input,embeddings)'):
        match split_mode:
            case '(methylseq_input,embeddings)':
                try:
                    x, embeddings = x
                    return x, embeddings
                except Exception as e:
                    raise ValueError(f"MethylSeqNN.mode='{self.mode}' mode forward(x) requires x be a tuple containing (methylseq_input, embeddings)") from e 
            case _:
                raise NotImplementedError(f'{split_mode} splitting not implemented.')
    
    # def _check_x_attributes(self, x):
    #     if isinstance(x,tuple):
    #         raise ValueError(f"MethylSeqNN.mode='{self.mode}' does not support MultiDataset tuple inputs; use a forward mode designed for your dataset class.")
    #     if x.dim()!=3:
    #         raise ValueError(f"MethylSeqNN.forward requires a methylseq_input (first or only element of x) with three dimensions: (N,C,L). Found {x.dim()}")
    #     if x.shape[1]<7:
    #         raise ValueError(f"Forward passes for MethylSeqNN require that methylseq_input have (first or only element of x) 7 or more channels; if using only DNA onehot you must pad up to 7 with zeros. Found shape was {x.shape[1]}") 
    
    def _pretrained_embedder_forward(self, sequence):
        x_seq = sequence
        if self.pretrained_seq_model:
            if self.seq_input_head:
                for layer in self.seq_input_head:
                    x_seq = layer(x_seq)
            embeddings = self.pretrained_seq_model(x_seq)     
            if "pretrained_embedder_rep" in self.supplemental_predict_outputs:
                self.hooked_supplemental_outputs['pretrained_embedder_rep'] = embeddings
            return embeddings
        else:
            return None

    def _pretrained_head_forward(self, x):
        if self.seq_output_head:
            for layer in self.seq_output_head:
                x = layer(x) 
        return x
    
    def _residual_forward(self, sequence, methylation, embeddings, dataset_key):
        x_pseudobatch_list = []
        for cell_type_idx in range(methylation.shape[1]):
            x_methylseq = torch.cat(
                [
                    sequence,
                    methylation[:,cell_type_idx],
                ],
                dim=1
            )
            if self.crop_off_sequence:
                x_methylseq = x_methylseq[:,:,self.crop_off_sequence:-self.crop_off_sequence]
            x_pseudobatch_list.append(x_methylseq)

        pseudobatch_scaleup = len(x_pseudobatch_list)
        x_pseudobatch = torch.cat(x_pseudobatch_list, dim=0)

        x_pseudobatch = self._residual_layers_forward(x_pseudobatch,embeddings,pseudobatch_scaleup=pseudobatch_scaleup)
        
        x_methylseq_allchannels = x_pseudobatch.new_zeros(sequence.size(0), *x_pseudobatch.shape[1:])
        
        batch_size = sequence.size(0)
        if pseudobatch_scaleup==1:
            x_methylseq_allchannels = x_pseudobatch
        else:
            for cell_type_idx, (cell_type, channels) in enumerate(self.get_input_to_outputs_dict(dataset_key).items()):
                start = cell_type_idx*batch_size
                end = (cell_type_idx+1)*batch_size
                x_cell_type = x_pseudobatch[start:end]
                x_methylseq_allchannels[:, channels, :] = x_cell_type[:, channels, :]
        return x_methylseq_allchannels

    def _merge_submodels(self, x_res, x_seq):
        if self.layers and self.pretrained_seq_model:
            if self.model_merge_operation in self.operations:
                x = self.operations[self.model_merge_operation](x_res, x_seq)
                return x
            else:
                raise ValueError(f"Unsupported model_merge_operation: {self.model_merge_operation}")    
        elif self.layers:
            return x_res
        elif self.pretrained_seq_model:
            return x_seq
        else:
            raise RuntimeError("No model layers")
    
    def _merged_output_forward(self, x):
        if self.merged_output_head:
            for layer in self.merged_output_head:
                x = layer(x)
        return x
                
    def _residual_layers_forward(self, x, embeddings, pseudobatch_scaleup=1):
        for layer_index,layer in enumerate(self.layers):
            if layer_index in self.concat_pretrained_embeddings_at:
                rbs = self.concat_pretrained_embeddings_at[layer_index]
                x = self._concat_pretrained_embeddings(
                    embeddings, 
                    rbs, 
                    x, 
                    embeddings_pseudobatch_scaleup=pseudobatch_scaleup)
            x = layer(x)
        if self.crop_off_final:
            x = x[:,:,self.crop_off_final:-self.crop_off_final] 

        return x

    def _embeddings_factorization_forward(self, sequence, methylation, embeddings, dataset_key):
        methyl_indep_seq_rep = self._embeddings_to_methyl_indep_seq_rep_forward(embeddings)
        methyl_dep_seq_rep = self._embeddings_to_methyl_dep_seq_rep_forward(embeddings)
        true_methyl_rep = self._input_to_methyl_rep_forward(sequence, methylation, embeddings, dataset_key)
        if math.isclose(self.true_methyl_rep_weight,1.0) and self.methyl_rep_loss_weight==0:
            imputed_methyl_rep = torch.zeros_like(true_methyl_rep)
        else:
            imputed_methyl_rep = self._embeddings_to_methyl_rep_forward(embeddings, dataset_key)
        if "methyl_indep_seq_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['methyl_indep_seq_rep'] = methyl_indep_seq_rep
        if "methyl_dep_seq_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['methyl_dep_seq_rep'] = methyl_dep_seq_rep
        if "true_methyl_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['true_methyl_rep'] = true_methyl_rep
        if "imputed_methyl_rep" in self.supplemental_predict_outputs:
            self.hooked_supplemental_outputs['imputed_methyl_rep'] = imputed_methyl_rep
        match self.interpolate_methyl_reps_location:
            case 'output':
                if math.isclose(self.true_methyl_rep_weight,1.0):
                    return self._factorized_reps_to_output_forward(methyl_indep_seq_rep, methyl_dep_seq_rep, true_methyl_rep, dataset_key)
                elif math.isclose(self.true_methyl_rep_weight,0.0):
                    return self._factorized_reps_to_output_forward(methyl_indep_seq_rep, methyl_dep_seq_rep, imputed_methyl_rep, dataset_key)
                else:
                    x_true_component = self.true_methyl_rep_weight * self._factorized_reps_to_output_forward(methyl_indep_seq_rep, methyl_dep_seq_rep, true_methyl_rep, dataset_key)
                    x_imputed_component = (1 - self.true_methyl_rep_weight) * self._factorized_reps_to_output_forward(methyl_indep_seq_rep, methyl_dep_seq_rep, imputed_methyl_rep, dataset_key)
                    return x_true_component + x_imputed_component
            case 'rep':
                interpolated_rep = self.true_methyl_rep_weight * true_methyl_rep + (1 - self.true_methyl_rep_weight) * imputed_methyl_rep
                return self._factorized_reps_to_output_forward(methyl_indep_seq_rep, methyl_dep_seq_rep, interpolated_rep, dataset_key)
            case _:
                raise NotImplementedError(f"interpolate_methyl_reps_location={self.interpolate_methyl_reps_location} not implemented.")
        
    def _embeddings_to_methyl_indep_seq_rep_forward(self, embeddings):
        seq_rep = embeddings
        for layer in self.embeddings_to_methyl_indep_seq_rep:
            seq_rep = layer(seq_rep)
        self.capture_methyl_indep_seq_rep(seq_rep)
        return seq_rep

    def _embeddings_to_methyl_dep_seq_rep_forward(self, embeddings):
        seq_rep = embeddings
        for layer in self.embeddings_to_methyl_dep_seq_rep:
            seq_rep = layer(seq_rep)
        self.capture_methyl_dep_seq_rep(seq_rep)
        return seq_rep

    def _embeddings_to_methyl_rep_forward(self, embeddings, dataset_key):
        methyl_rep = embeddings
        for layer in self.embeddings_to_methyl_rep:
            methyl_rep = layer(methyl_rep)
        methyl_rep = methyl_rep.view(methyl_rep.shape[0],self.num_cell_types['all'],-1,methyl_rep.shape[2])
        methyl_rep_sliced = methyl_rep[:,self.cell_type_list_per_dataset[dataset_key],:,:]
        self.capture_imputed_methyl_rep(methyl_rep_sliced)
        return methyl_rep_sliced

    def _input_to_methyl_rep_forward(self, sequence, methylation, embeddings, dataset_key):
        """
        Extract methylation channels from x and run through input_to_methyl_rep layers.
        This should always return a representation of shape (N,num_cell_types,rep_dim,L')
        Note: input_to_methyl_rep must encode from (N,3,L) to (N,1,rep_dim,L') per cell type
        """
        x_methyl_pseudobatch_list = []
        for cell_type_idx in range(methylation.shape[1]):
            x_methyl = torch.cat(
                [
                    sequence,
                    methylation[:,cell_type_idx],
                ],
                dim=1,
            )
            if self.crop_off_sequence:
                x_methyl = x_methyl[:,:,self.crop_off_sequence:-self.crop_off_sequence]
            x_methyl_pseudobatch_list.append(x_methyl)
        pseudobatch_scaleup = len(x_methyl_pseudobatch_list)
        x_methyl_pseudobatch = torch.cat(x_methyl_pseudobatch_list, dim=0)
        for layer_index, layer in enumerate(self.input_to_methyl_rep):
            if layer_index in self.concat_pretrained_embeddings_at:
                rbs = self.concat_pretrained_embeddings_at[layer_index]
                x_methyl_pseudobatch = self._concat_pretrained_embeddings(
                    embeddings, 
                    rbs, 
                    x_methyl_pseudobatch, 
                    embeddings_pseudobatch_scaleup=pseudobatch_scaleup)
            x_methyl_pseudobatch = layer(x_methyl_pseudobatch)
        if self.crop_off_final:
            x_methyl_pseudobatch = x_methyl_pseudobatch[:,:,self.crop_off_final:-self.crop_off_final] 
        batch_size = sequence.size(0)
        
        if methylation.shape[1]>1:
            x_methyl_allchannels = x_methyl_pseudobatch.new_zeros(batch_size, self.num_cell_types[dataset_key], *x_methyl_pseudobatch.shape[1:])
            for cell_type_idx, cell_type in enumerate(self.get_input_to_outputs_dict(dataset_key).keys()):
                start = cell_type_idx*batch_size
                end = (cell_type_idx+1)*batch_size
                x_cell_type = x_methyl_pseudobatch[start:end]
                x_methyl_allchannels[:, cell_type, :, :] = x_cell_type
        else:
            x_methyl_allchannels = x_methyl_pseudobatch.unsqueeze(1)
        x_methyl_allchannels = self.capture_true_methyl_rep(x_methyl_allchannels)
        return x_methyl_allchannels

    def _factorized_reps_to_output_forward(self, methyl_indep_seq_rep, methyl_dep_seq_rep, methyl_rep, dataset_key):
        """
        Combine methylation-independent and methylation-dependent sequence representations

        Cases:
        1) factorized_reps_to_output_submodel_per_task = True
            For each task, select the appropriate methylation representation and combine with
            the methylation-dependent sequence representation (if applicable) and methylation-independent
            sequence representation. Pass through the task-specific output submodel and assemble the outputs
            into the full output tensor.
        2) factorized_reps_to_output_submodel_per_task = False
            For each cell type, select the appropriate methylation representation and combine with
            the methylation-dependent sequence representation (if applicable) and methylation-independent
            sequence representation. Create a pseudobatch by stacking all cell types together.
            Pass through the shared output submodel. Then, reassemble the outputs into the full output tensor.

        Args:
            methyl_indep_seq_rep: (N, C_indep, L)
            methyl_dep_seq_rep: (N, C_dep, L)
            methyl_rep: (N, num_cell_types, C_methyl, L)
            dataset_key: which dataset is being processed (to select output channels)
        """
        if self.factorized_reps_to_output_submodel_per_task:
            x_output_allchannels = methyl_indep_seq_rep.new_zeros(methyl_indep_seq_rep.size(0), self.out_tracks, methyl_indep_seq_rep.size(2))
            for cell_type_idx, (cell_type, channel_tuples) in enumerate(self.get_input_to_outputs_dict(dataset_key,absolute_and_relative_channels=True).items()):
                # this is slow! I assume. Something more like the pseudobatching above should be much quicker
                for relative_task_index, absolute_task_index in channel_tuples:
                    if methyl_rep.shape[1]>1:
                        celltype_methyl_rep = methyl_rep[:, cell_type_idx, :, :]
                    else:
                        celltype_methyl_rep = methyl_rep[:, 0, :, :]
                    if self.embeddings_to_methyl_dep_seq_rep:
                        methyl_dep_seq_rep_task = self.operations[self.model_merge_operation](methyl_dep_seq_rep, celltype_methyl_rep)
                        x_methylseq_rep = torch.cat([methyl_indep_seq_rep, methyl_dep_seq_rep_task], dim=1)
                    else:
                        x_methylseq_rep = torch.cat([methyl_indep_seq_rep, celltype_methyl_rep], dim=1)
                    for layer in self.factorized_reps_to_output[f"factorized_reps_to_output_task{absolute_task_index}"]:
                        x_methylseq_rep = layer(x_methylseq_rep)
                    x_output_allchannels[:, absolute_task_index:absolute_task_index+1, :] = x_methylseq_rep
            return x_output_allchannels
        else:
            x_methylseq_pseudobatch_list = []
            for cell_type_idx in range(methyl_rep.shape[1]):
                celltype_methyl_rep = methyl_rep[:, cell_type_idx, :, :]
                if self.embeddings_to_methyl_dep_seq_rep:
                    methyl_dep_seq_rep_celltype = self.operations[self.model_merge_operation](methyl_dep_seq_rep, celltype_methyl_rep)
                    x_methylseq_rep = torch.cat([methyl_indep_seq_rep, methyl_dep_seq_rep_celltype], dim=1)
                else:
                    x_methylseq_rep = torch.cat([methyl_indep_seq_rep, celltype_methyl_rep], dim=1)
                x_methylseq_pseudobatch_list.append(x_methylseq_rep)
            pseudobatch_scaleup = len(x_methylseq_pseudobatch_list)
            x_methylseq_pseudobatch = torch.cat(x_methylseq_pseudobatch_list, dim=0)
            for layer in self.factorized_reps_to_output:
                x_methylseq_pseudobatch = layer(x_methylseq_pseudobatch)
            if methyl_rep.shape[1]==1:
                x_output_allchannels = x_methylseq_pseudobatch
            else:
                x_output_allchannels = x_methylseq_pseudobatch.new_zeros(methyl_indep_seq_rep.size(0), self.out_tracks, methyl_indep_seq_rep.size(2))
                batch_size = methyl_indep_seq_rep.size(0)
                for cell_type_idx, (cell_type, channel_tuples) in enumerate(self.get_input_to_outputs_dict(dataset_key,absolute_and_relative_channels=True).items()):
                    start = cell_type_idx*batch_size
                    end = (cell_type_idx+1)*batch_size
                    x_cell_type = x_methylseq_pseudobatch[start:end]
                    for relative_task_index, absolute_task_index in channel_tuples:
                        x_output_allchannels[:, absolute_task_index:absolute_task_index+1, :] = x_cell_type[:, absolute_task_index:absolute_task_index+1, :]
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

    # def log(self, name, value, *args, **kwargs):
    #     if kwargs.get("sync_dist", False):
    #         if not torch.is_tensor(value) or value.device.type != "cuda":
    #             print(f"⚠️ sync_dist CPU/non-tensor metric → {name}: type={type(value)} "
    #                 f"device={(None if not torch.is_tensor(value) else value.device)}")
    #             # auto-fix so you can keep running:
    #             if not torch.is_tensor(value):
    #                 value = torch.tensor(value, dtype=torch.float32, device=next(self.parameters()).device)
    #             else:
    #                 value = value.to(next(self.parameters()).device)
    #     return super().log(name, value, *args, **kwargs)