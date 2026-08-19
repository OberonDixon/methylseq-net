import h5py
from pathlib import Path
import numpy as np
import gin
import pandas as pd
from tqdm.auto import tqdm
import warnings
import os
from abc import ABC, abstractmethod

import torch
from lightning.pytorch.callbacks import BasePredictionWriter

@gin.register
@gin.configurable
class DatasetWriter:
    """
    Write a dataset wherein each sample is only for one cell type and its associated tasks
    """
    def __init__(
        self,
        seq_length: int,
        cpg_input: bool,
        track_length: int,
        num_tracks: int,
        output_path: str | Path,
        mask: bool = True,
        io_mappings_list: list=[],
        append=False,
    ):
        """
        Args:
            seq_length: the length of the nucleotide sequence
            cpg_input: True if you are providing methylation tracks to write into the dataset
            track_length: the number of predicton track bins
            num_tracks: the number of different prediction tasks
            output_path: the place to which the dataset will be written, including filename
            mask: True if some tasks are masked out for some samples
            io_mappings_list: the task identification for each output task - what cell type, etc
        """
        warnings.warn("DatasetWriter is deprecated; please use MultiMethylWriter instead for new datasets.")

        # The length of the input sequence
        self.seq_length = seq_length
        # True if we are going to provide CpG methylation in the input encoding
        self.cpg_input = cpg_input
        # The length of the label tracks
        self.track_length = track_length
        # The number of label tracks
        self.num_tracks = num_tracks
        # The path for the output hdf5 file
        if Path(output_path).suffix in ['.h5','.hdf5']:
            self.output_path = Path(output_path)
        else:
            raise ValueError(f'{Path(output_path)} is not an .h5 or .hdf5 path')
        # True means we are using a mask for the loss function
        self.mask = mask
        self.io_mappings_list = io_mappings_list
        self.append = append
        
        self.initialize_h5()
    def initialize_h5(self):
        if not self.append:
            with h5py.File(self.output_path,'w') as f:
                if 'specifier' in f:
                    del f['specifier']
                f.create_dataset(
                    'specifier',
                    (0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression='lzf',
                    chunks=(1,),
                )
                if 'sequence' in f:
                    del f['sequence']
                f.create_dataset(
                    'sequence',
                    (0,7,self.seq_length),
                    maxshape=(None,7,self.seq_length),
                    dtype=np.float16,
                    compression='lzf',
                    chunks=(1,7,self.seq_length),
                )
                if 'tracks' in f:
                    del f['tracks']
                f.create_dataset(
                    'tracks',
                    (0,self.num_tracks,self.track_length),
                    maxshape=(None,self.num_tracks,self.track_length),
                    dtype='float',
                    compression='lzf',
                    chunks=(1,self.num_tracks,self.track_length),
                )
                if self.mask:
                    if 'mask' in f:
                        del f['mask']
                    f.create_dataset(
                        'mask',
                        (0,self.num_tracks,self.track_length),
                        maxshape=(None,self.num_tracks,self.track_length),
                        dtype='bool',
                        compression='lzf',
                        chunks=(1,self.num_tracks,self.track_length),
                    )
                # Log the gin config string and io mappings as attributes in the HDF5 file
                gin_config_str = gin.operative_config_str()
                f.attrs['gin_config'] = gin_config_str
                f.attrs['io_mappings'] = pd.DataFrame(self.io_mappings_list).to_csv(sep='\t', index=False)
                
    def __len__(self):
        with h5py.File(self.output_path, 'r') as f:
            return f['sequence'].shape[0]
    
    def write_chunk(
        self,
        indices_list,
        sample_specifier_list,
        onehot_seq_list,
        labels_list=None,
        mask_list=None,
    ):
        if labels_list is not None and len(onehot_seq_list)!=len(labels_list):
            raise ValueError(f'Cannot write chunk, unbalanced lengths:{len(onehot_seq_list)} sequences and {len(labels_list)} labels.')
        if len(onehot_seq_list[0])!=self.seq_length:
            raise ValueError(f'Cannot write chunk, seq length is {len(onehot_seq_list[0])} and should be {self.seq_length}.')
        if labels_list is not None and labels_list[0].shape[0]!=self.track_length:
            raise ValueError(f'Cannot write chunk, track length is {labels_list[0].shape[0]} and should be {self.track_length}.')

        if self.mask:
            if mask_list is None:
                raise ValueError("Datasetwriter initialized with 'mask=True', must provide a mask_list when writing chunk. Got 'None'")
            if mask_list[0].shape[0]!=self.track_length:
                raise ValueError(f'Cannot write chunk, mask length is {mask_list[0].shape[0]} and should be {self.track_length}.')
                
            
        with h5py.File(self.output_path, 'a') as f:
            specifier_dataset = f['specifier']
            seq_dataset = f['sequence']
            track_dataset = f['tracks']
            if self.mask:
                mask_dataset = f['mask']

            current_specifier_size = specifier_dataset.shape[0]
            current_seq_size = seq_dataset.shape[0]
            current_track_size = track_dataset.shape[0]
            if self.mask:
                current_mask_size = mask_dataset.shape[0]
            else:
                current_mask_size = 0

            samples_per_region = int(len(onehot_seq_list)/len(indices_list))
            start_index = np.min(np.array(indices_list))*samples_per_region
            end_index = (np.max(np.array(indices_list)) + 1)*samples_per_region

            if current_specifier_size==current_seq_size and current_seq_size==current_track_size and (not self.mask or current_mask_size==current_track_size):  
                if current_specifier_size<end_index:
                    specifier_dataset.resize(end_index, axis=0)
                    seq_dataset.resize(end_index, axis=0)
                    track_dataset.resize(end_index, axis=0) 
                    if self.mask:
                        mask_dataset.resize(end_index, axis=0)
            else:
                raise ValueError(f"Dataset sizes in {self.output_path} do not match: region={current_regions_size},sequence={current_seq_size},tracks={current_track_size}")

            try:
                specifier_dataset[start_index:end_index] = sample_specifier_list
                seq_dataset[start_index:end_index, :, :] = [np.transpose(onehot_seq,(1,0)) for onehot_seq in onehot_seq_list]
                if labels_list is None:
                    track_dataset[start_index:end_index, :, :] = np.nan
                else:
                    track_dataset[start_index:end_index, :, :] = [np.transpose(label,(1,0)) for label in labels_list]
                if self.mask:
                    mask_dataset[start_index:end_index, :, :] = [np.transpose(mask,(1,0)) for mask in mask_list]
            except IndexError as e:
                raise IndexError(f"Indexing error with indices_list: {indices_list}. Ensure all indices are within bounds.") from e
            except ValueError as e:
                raise ValueError(f"Value assignment error: check dimensions of assigned data. {e}") from e

@gin.register
@gin.configurable
class MultiMethylWriter:
    """
    Writes a dataset containing sequence and methylation information for all cell types/states in a single sample,
    with support for multiple variants per sample (e.g., different haplotypes, epigenetic edits, or environmental contexts).
    """
    def __init__(
        self,
        seq_length: int,
        track_length: int,
        num_tracks: int,
        output_path: str | Path,
        num_variants: int = 1,
        mask: bool = True,
        io_mappings_list: list=[],
        append=False,
    ):
        """
        Args:
            seq_length: the length of the nucleotide sequence
            track_length: the number of predicton track bins
            num_tracks: the number of different prediction tasks
            output_path: the place to which the dataset will be written, including filename
            num_variants: the number of variants per sample (e.g., haplotypes, epigenetic edits)
            mask: True if some tasks are masked out for some samples
            io_mappings_list: the task identification for each output task - what cell type, etc. This will be used to determine dataset shapes.
        """
        # The length of the input sequence
        self.seq_length = seq_length
        # The length of the label tracks
        self.track_length = track_length

        self.num_tracks = num_tracks
        if io_mappings_list:
            if self.num_tracks != max([io_mapping["channel"] for io_mapping in io_mappings_list]) + 1:
                raise ValueError(f"num_tracks unexpected value: calculated {max([io_mapping['channel'] for io_mapping in io_mappings_list]) + 1} from io_mappings_list but {self.num_tracks} was provided instead.")
            self.num_states = max([io_mapping["cell_type"] for io_mapping in io_mappings_list]) + 1
        else:
            warnings.warn("io_mappings_list is empty; setting num_states to 1 by default.")
            self.num_states = 1
        
        
        self.num_variants = num_variants
        
        # The path for the output hdf5 file
        if Path(output_path).suffix in ['.h5','.hdf5']:
            self.output_path = Path(output_path)
        else:
            raise ValueError(f'{Path(output_path)} is not an .h5 or .hdf5 path')
            
        # True means we are using a mask for the loss function
        self.mask = mask
        self.io_mappings_list = io_mappings_list

        self.append = append
        
        self.initialize_h5()
        
    def initialize_h5(self):
        if not self.append:
            with h5py.File(self.output_path,'w') as f:
                if 'specifier' in f:
                    del f['specifier']
                f.create_dataset(
                    'specifier',
                    (0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression='lzf',
                    chunks=(1,),
                )
                if 'sequence' in f:
                    del f['sequence']
                f.create_dataset(
                    'sequence',
                    (0,self.num_variants,4,self.seq_length),
                    maxshape=(None,self.num_variants,4,self.seq_length),
                    dtype=np.float16,
                    compression='lzf',
                    chunks=(1,self.num_variants,4,self.seq_length),
                )
                if 'methylation' in f:
                    del f['methylation']
                f.create_dataset(
                    'methylation',
                    (0,self.num_variants,self.num_states,3,self.seq_length),
                    maxshape=(None,self.num_variants,self.num_states,3,self.seq_length),
                    dtype=np.float16,
                    compression='lzf',
                    chunks=(1,self.num_variants,self.num_states,3,self.seq_length),
                )
                if 'tracks' in f:
                    del f['tracks']
                f.create_dataset(
                    'tracks',
                    (0,self.num_variants,self.num_tracks,self.track_length),
                    maxshape=(None,self.num_variants,self.num_tracks,self.track_length),
                    dtype='float',
                    compression='lzf',
                    chunks=(1,self.num_variants,self.num_tracks,self.track_length),
                )
                if self.mask:
                    if 'mask' in f:
                        del f['mask']
                    f.create_dataset(
                        'mask',
                        (0,self.num_variants,self.num_tracks,self.track_length),
                        maxshape=(None,self.num_variants,self.num_tracks,self.track_length),
                        dtype='bool',
                        compression='lzf',
                        chunks=(1,self.num_variants,self.num_tracks,self.track_length),
                    )
                # Log the gin config string and io mappings as attributes in the HDF5 file
                gin_config_str = gin.operative_config_str()
                f.attrs['gin_config'] = gin_config_str
                f.attrs['io_mappings'] = pd.DataFrame(self.io_mappings_list).to_csv(sep='\t', index=False)

    def __len__(self):
        with h5py.File(self.output_path, 'r') as f:
            return f['sequence'].shape[0]
    
    def write_chunk(
        self,
        indices_list,
        sample_specifier_list,
        onehot_seq_list,
        methylation_info_list=None,
        labels_list=None,
        mask_list=None,
    ):
        """
        Write a chunk of data to the HDF5 file.
        
        Args:
            indices_list: List of sample indices to write
            sample_specifier_list: List of sample identifiers
            onehot_seq_list: List of sequences, each with shape (num_variants, seq_length, 4) 
                            OR (seq_length, 4) if num_variants=1 (will auto-expand)
            methylation_info_list: List of methylation data, each with shape (num_variants, seq_length, 3*num_states) or (num_variants, seq_length, num_states, 3)
                                  OR (seq_length, 3*num_states or (seq_length, num_states, 3) if num_variants=1 (will auto-expand)
            labels_list: List of labels, each with shape (num_variants, track_length, num_tracks)
                        OR (track_length, num_tracks) if num_variants=1 (will auto-expand)
            mask_list: List of masks, each with shape (num_variants, track_length, num_tracks)
                      OR (track_length, num_tracks) if num_variants=1 (will auto-expand)
        """
        if labels_list is not None and len(onehot_seq_list)!=len(labels_list):
            raise ValueError(f'Cannot write chunk, unbalanced lengths:{len(onehot_seq_list)} sequences and {len(labels_list)} labels.')
        # Auto-expand dimensions if num_variants=1 and input lacks variants dimension
        if self.num_variants == 1:
            # Check if sequences need expansion (shape is (seq_length, 4) instead of (1, seq_length, 4))
            if len(onehot_seq_list[0].shape) == 2:
                onehot_seq_list = [seq[np.newaxis, :, :] for seq in onehot_seq_list]
            
            # Check if methylation needs expansion
            if methylation_info_list is not None and len(methylation_info_list[0].shape) == 2:
                methylation_info_list = [methyl[np.newaxis, :, :] for methyl in methylation_info_list]
            
            # Check if labels need expansion
            if labels_list is not None and len(labels_list[0].shape) == 2:
                labels_list = [label[np.newaxis, :, :] for label in labels_list]
            
            # Check if masks need expansion
            if mask_list is not None and len(mask_list[0].shape) == 2:
                mask_list = [mask[np.newaxis, :, :] for mask in mask_list]

        # Auto-reshape methylation from old to new format if needed
        if methylation_info_list is not None:
            reshaped_methyl = []
            for methyl in methylation_info_list:
                # Check if it's the old flattened format: (..., seq_length, 3*num_states)
                if methyl.shape[-1] == 3 * self.num_states and len(methyl.shape) == 3:
                    # Reshape from (num_variants, seq_length, 3*num_states) 
                    # to (num_variants, seq_length, num_states, 3)
                    reshaped = methyl.reshape(methyl.shape[0], methyl.shape[1], self.num_states, 3)
                    reshaped_methyl.append(reshaped)
                elif methyl.shape[-1] == 3 and methyl.shape[-2] == self.num_states:
                    # Already in new format: (num_variants, seq_length, num_states, 3)
                    reshaped_methyl.append(methyl)
                else:
                    raise ValueError(f"Unexpected methylation shape: {methyl.shape}. Expected either "
                                f"(num_variants, seq_length, {3*self.num_states}) or "
                                f"(num_variants, seq_length, {self.num_states}, 3)")
            methylation_info_list = reshaped_methyl
        
        # Check shapes - now expecting (num_variants, seq_length, 4) for sequences
        if len(onehot_seq_list[0].shape) != 3:
            raise ValueError(f'Expected onehot_seq to have 3 dimensions (num_variants, seq_length, 4), got shape {onehot_seq_list[0].shape}')
        if onehot_seq_list[0].shape[0] != self.num_variants:
            raise ValueError(f'Cannot write chunk, first dimension should be {self.num_variants} variants, got {onehot_seq_list[0].shape[0]}.')
        if onehot_seq_list[0].shape[1] != self.seq_length:
            raise ValueError(f'Cannot write chunk, seq length is {onehot_seq_list[0].shape[1]} and should be {self.seq_length}.')
        
        if labels_list is not None:
            if labels_list[0].shape[0] != self.num_variants:
                raise ValueError(f'Cannot write chunk, labels first dimension should be {self.num_variants} variants, got {labels_list[0].shape[0]}.')
            if labels_list[0].shape[1] != self.track_length:
                raise ValueError(f'Cannot write chunk, track length is {labels_list[0].shape[1]} and should be {self.track_length}.')

        if self.mask:
            if mask_list is None:
                raise ValueError("Datasetwriter initialized with 'mask=True', must provide a mask_list when writing chunk. Got 'None'")
            if mask_list[0].shape[0] != self.num_variants:
                raise ValueError(f'Cannot write chunk, mask first dimension should be {self.num_variants} variants, got {mask_list[0].shape[0]}.')
            if mask_list[0].shape[1] != self.track_length:
                raise ValueError(f'Cannot write chunk, mask length is {mask_list[0].shape[1]} and should be {self.track_length}.')
                
            
        with h5py.File(self.output_path, 'a') as f:
            specifier_dataset = f['specifier']
            seq_dataset = f['sequence']
            methyl_dataset = f['methylation']
            track_dataset = f['tracks']
            if self.mask:
                mask_dataset = f['mask']

            current_specifier_size = specifier_dataset.shape[0]
            current_seq_size = seq_dataset.shape[0]
            current_methyl_size = methyl_dataset.shape[0]
            current_track_size = track_dataset.shape[0]
            if self.mask:
                current_mask_size = mask_dataset.shape[0]
            else:
                current_mask_size = 0

            start_index = np.min(np.array(indices_list))
            end_index = np.max(np.array(indices_list)) + 1

            if (
                current_specifier_size==current_seq_size 
                and current_seq_size==current_track_size 
                and current_track_size==current_methyl_size
                and (not self.mask or current_mask_size==current_track_size)):  
                if current_specifier_size<end_index:
                    specifier_dataset.resize(end_index, axis=0)
                    seq_dataset.resize(end_index, axis=0)
                    methyl_dataset.resize(end_index, axis=0)
                    track_dataset.resize(end_index, axis=0) 
                    if self.mask:
                        mask_dataset.resize(end_index, axis=0)
            else:
                raise ValueError(f"Dataset sizes in {self.output_path} do not match: region={current_regions_size},sequence={current_seq_size},tracks={current_track_size}")

            try:
                specifier_dataset[start_index:end_index] = sample_specifier_list
                seq_dataset[start_index:end_index, :, :, :] = [np.transpose(onehot_seq,(0,2,1)) for onehot_seq in onehot_seq_list]
                if labels_list is None:
                    track_dataset[start_index:end_index, :, :, :] = np.nan
                else:
                    track_dataset[start_index:end_index, :, :, :] = [np.transpose(label,(0,2,1)) for label in labels_list]
                if methylation_info_list is None:
                    methyl_dataset[start_index:end_index, :, :, :, :] = 0
                else:
                    methyl_dataset[start_index:end_index, :, :, :, :] = [np.transpose(methyl,(0,2,3,1)) for methyl in methylation_info_list]
                if self.mask:
                    mask_dataset[start_index:end_index, :, :, :] = [np.transpose(mask,(0,2,1)) for mask in mask_list]
            except IndexError as e:
                raise IndexError(f"Indexing error with indices_list: {indices_list}. Ensure all indices are within bounds.") from e
            except ValueError as e:
                raise ValueError(f"Value assignment error: check dimensions of assigned data. {e}") from e

@gin.register
@gin.configurable
class SeqEmbeddingsWriter:
    """
    Writes a dataset containing pretrained model embeddings
    """
    def __init__(
        self, 
        embeddings_shape, 
        output_path,
        append = False,
    ):
        self.embeddings_shape = embeddings_shape
        if Path(output_path).suffix in ['.h5','.hdf5']:
            self.output_path = Path(output_path)
        else:
            raise ValueError(f'{Path(output_path)} is not an .h5 or .hdf5 path')  
        
        self.append = append
        
        self.initialize_h5()

    def initialize_h5(self):
        if not self.append:
            with h5py.File(self.output_path,'w') as f:
                if 'specifier' in f:
                    del f['specifier']
                f.create_dataset(
                    'specifier',
                    (0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression='lzf',
                    chunks=(1,),
                )
                if 'embeddings' in f:
                    del f['embeddings']
                f.create_dataset(
                    'embeddings',
                    (0,) + self.embeddings_shape,
                    maxshape=(None,) + self.embeddings_shape,
                    dtype='float',
                    compression='lzf',
                    chunks=(1,) + self.embeddings_shape,
                )  
                gin_config_str = gin.operative_config_str()
                f.attrs['gin_config'] = gin_config_str
                
    def __len__(self):
        with h5py.File(self.output_path, 'r') as f:
            return f['embeddings'].shape[0]
    
    def write_chunk(
        self,
        indices_list,
        sample_specifier_list,
        embeddings_list,
    ):
        if len(indices_list)!=len(embeddings_list):
            raise ValueError(f'Cannot write chunk, unbalanced lengths: {len(indices_list)} indices and {len(embeddings_list)} embeddings.')
        end_index = np.max(np.array(indices_list))+1
        with h5py.File(self.output_path, 'a') as f:
            specifier_dataset = f['specifier']
            embeddings_dataset = f['embeddings']
            current_specifier_size = specifier_dataset.shape[0]
            current_embeddings_size = embeddings_dataset.shape[0]
            if end_index>current_embeddings_size:
                specifier_dataset.resize(end_index, axis=0) 
                embeddings_dataset.resize(end_index, axis=0) 
            specifier_dataset[indices_list] = sample_specifier_list
            embeddings_dataset[indices_list] = embeddings_list
                
class BigWigWriter:
    def __init__(self, output_file, contigs):
        """
        Initializes the BigWigWriter.

        Args:
            output_file (str): Path to the output BigWig file.
            reference_genome (str): Path to the reference genome file (FASTA).
        """
        self.output_file = output_file
        self.contigs = contigs

    def write_genome(self,genome_data_dict,chunk_size=1000):
        """
        Writes data to the specified coordinate range for a given contig.

        Args:
            contig (str): The name of the contig.
            start (int): Start position (0-based, inclusive).
            end (int): End position (0-based, exclusive).
            data (np.ndarray): Numpy array of data values to write.
        """
        import pyBigWig

        # Open the BigWig file write data
        with pyBigWig.open(self.output_file, "w") as bw:
            bw.addHeader(self.contigs)
            for contig,contig_data in tqdm(genome_data_dict.items(),desc=f"writing contigs for {Path(self.output_file).name}",leave=False):
                entries = contig_data['entries']
                motifs = contig_data['motifs']
                start = contig_data['start']
                end = contig_data['end']

                for chunk_start in range(0,len(entries)-1,chunk_size):
                    chunk_end = min(len(entries),chunk_start+chunk_size)
                    motifs_chunk = motifs[chunk_start:chunk_end]
                    entries_chunk = entries[chunk_start:chunk_end]
                    if len(motifs)>0:
                        # Write only filtered positions to the BigWig file
                        bw.addEntries(
                            ([contig] * len(entries_chunk)),  # Contig names
                            motifs_chunk.tolist(),          # Start positions
                            ends=(motifs_chunk + 1).tolist(),  # End positions
                            values=entries_chunk.tolist()       # Corresponding values
                        )

class BaseHDF5Writer(ABC):
    def __init__(
        self,
        output_dir=None,
        no_targets=False,
    ):
        self.file_handles = {}
        self.pred_counter = 0
        if output_dir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = self._temp_dir_obj.name
        else:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        self.no_targets = no_targets

    def append_batch_to_h5(self, train, pl_module, prediction_dict, batch_indices, batch):
        if "predictions" not in prediction_dict or "specifier" not in prediction_dict:
            raise ValueError("dictionary output from predict_step method must contain 'predictions' and 'specifier' keys.")
        targets = pl_module.targets_from_batch(batch)
        # It appears that all ranks send to rank 0 and write out - but if not, then this logic currently breaks
        rank = train.global_rank
        if rank!=0:
            raise ValueError(f"Unexpected rank {rank}. Code in callbacks.py::HDF5PredictionWriter needs to be rewritten if ranks are not getting merged for writing, otherwise values will be missed.")
        path = os.path.join(self.output_dir, f"predictions.h5")

        if not self.no_targets:
            targets_shape = targets.shape[1:]
            pred_shape = prediction_dict["predictions"].shape[1:]
            assert pred_shape == targets_shape, f"Predictions shape {pred_shape} does not match targets shape {targets_shape}"
        
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "w")
            for key, value in prediction_dict.items():
                if key == "specifier":
                    self.file_handles[path].create_dataset("specifier", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8"), chunks=True)
                else:
                    per_batch_shape = value.shape[1:]
                    self.file_handles[path].create_dataset(key, shape=(0, *per_batch_shape), maxshape=(None, *per_batch_shape), chunks=True)
            if not self.no_targets:
                self.file_handles[path].create_dataset("tracks", shape=(0, *targets_shape), maxshape=(None, *targets_shape), chunks=True)
            self.file_handles[path].create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
            self.file_handles[path].attrs['io_mappings'] = getattr(pl_module, 'io_mappings_str', '')

        f = self.file_handles[path]
        batch_indices = np.array(batch_indices)
        targets_np = targets.detach().cpu().numpy()
        curr_size = f["predictions"].shape[0]

        if batch_indices is None:
            batch_indices = np.arange(self.pred_counter, self.pred_counter + predictions_np.shape[0])
            self.pred_counter += predictions_np.shape[0]

        # Resize datasets
        for key, value in prediction_dict.items():
            f[key].resize(max(curr_size,max(batch_indices)+1), axis=0)
            f[key][batch_indices] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        
        if not self.no_targets:
            f["tracks"].resize(max(curr_size,max(batch_indices)+1), axis=0)
            f["tracks"][batch_indices] = targets_np

        f["indices"].resize(max(curr_size,max(batch_indices)+1), axis=0)
        f["indices"][batch_indices] = batch_indices

    def __del__(self):
        self._close_all()
        if hasattr(self, '_temp_dir_obj') and self._temp_dir_obj:
            try:
                self._temp_dir_obj.cleanup()
            except Exception:
                pass

    def _close_all(self):
        for f in self.file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self.file_handles.clear()


class HDF5PredictionWriter(BasePredictionWriter, BaseHDF5Writer):
    def __init__(
        self, 
        output_dir, 
        write_interval="batch",
        no_targets=False,
    ):
        BasePredictionWriter.__init__(self,write_interval)
        BaseHDF5Writer.__init__(self,output_dir=output_dir,no_targets=no_targets,)

    def write_on_batch_end(self, train, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if train.world_size > 1:
            rank = train.global_rank
            world_size = train.world_size
            
            # these huge input tensors may slow down gathering; excluded from the gather on
            # every rank identically, since dist.gather is a collective that every rank must
            # call the same number of times in the same order
            batch_to_gather = {k: v for k, v in batch.items() if k not in ('sequence', 'methylation')}

            # print(f"batch keys {batch_to_gather.keys()} prediction keys {[prediction.keys()]}")
            # print(f"world_size {world_size} rank {rank} batch indices {batch_indices}")

            if rank == 0:
                # Gather everything
                gathered_prediction = {k: gather_to_rank0(v, world_size, rank) for k, v in prediction.items()}
                gathered_indices = gather_to_rank0(
                    torch.tensor(batch_indices, device=prediction['predictions'].device),
                    world_size, rank
                ).cpu().numpy() if batch_indices is not None else None
                gathered_batch = {k: gather_to_rank0(v, world_size, rank) if isinstance(v, torch.Tensor) else v
                                for k, v in batch_to_gather.items()}

                self.append_batch_to_h5(train, pl_module, gathered_prediction, gathered_indices, gathered_batch)
            else:
                # Non-root ranks just send
                for v in prediction.values():
                    gather_to_rank0(v, world_size, rank)
                if batch_indices is not None:
                    gather_to_rank0(torch.tensor(batch_indices, device=prediction['predictions'].device), world_size, rank)
                for v in batch_to_gather.values():
                    if isinstance(v, torch.Tensor):
                        gather_to_rank0(v, world_size, rank)
        else:
            self.append_batch_to_h5(train, pl_module, prediction, batch_indices, batch)

    def on_predict_end(self, train, pl_module):
        self._close_all()