import h5py
from pathlib import Path
import numpy as np
import gin
import pandas as pd
from tqdm.auto import tqdm

@gin.register
@gin.configurable
class DatasetWriter:
    def __init__(
        self,
        seq_length: int,
        cpg_input: bool,
        track_length: int,
        num_tracks: int,
        output_path: str | Path,
        mask: bool = True,
        io_mappings_list: list=[],
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
        
        self.initialize_h5()
    def initialize_h5(self):
        with h5py.File(self.output_path,'w') as f:
            if 'specifier' in f:
                del f['specifier']
            f.create_dataset(
                'specifier',
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression='gzip',
                compression_opts=2,
            )
            if 'sequence' in f:
                del f['sequence']
            f.create_dataset(
                'sequence',
                (0,7,self.seq_length),
                maxshape=(None,7,self.seq_length),
                dtype=np.float16,
                compression='gzip',
                compression_opts=2
            )
            if 'tracks' in f:
                del f['tracks']
            f.create_dataset(
                'tracks',
                (0,self.num_tracks,self.track_length),
                maxshape=(None,self.num_tracks,self.track_length),
                dtype='float',
                compression='gzip',
                compression_opts=2,
            )
            if self.mask:
                if 'mask' in f:
                    del f['mask']
                f.create_dataset(
                    'mask',
                    (0,self.num_tracks,self.track_length),
                    maxshape=(None,self.num_tracks,self.track_length),
                    dtype='bool',
                    compression='gzip',
                    compression_opts=2,
                )
            # Log the gin config string and io mappings as attributes in the HDF5 file
            gin_config_str = gin.operative_config_str()
            f.attrs['gin_config'] = gin_config_str
            f.attrs['io_mappings'] = pd.DataFrame(self.io_mappings_list).to_csv(sep='\t', index=False)
                
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
    def __init__(
        self,
        seq_length: int,
        track_length: int,
        num_tracks: int,
        output_path: str | Path,
        mask: bool = True,
        io_mappings_list: list=[],
    ):
        """
        Args:
            seq_length: the length of the nucleotide sequence
            track_length: the number of predicton track bins
            output_path: the place to which the dataset will be written, including filename
            mask: True if some tasks are masked out for some samples
            io_mappings_list: the task identification for each output task - what cell type, etc. This will be used to determine dataset shapes.
        """
        # The length of the input sequence
        self.seq_length = seq_length
        # The length of the label tracks
        self.track_length = track_length

        self.num_tracks = num_tracks
        if self.num_tracks != max([io_mapping["channel"] for io_mapping in io_mappings_list]) + 1:
            raise ValueError(f"num_tracks unexpected value: calculated {max([io_mapping['channel'] for io_mapping in io_mappings_list]) + 1} from io_mappings_list but {self.num_tracks} was provided instead.")
        self.num_cell_types = max([io_mapping["cell_type"] for io_mapping in io_mappings_list]) + 1
        
        # The path for the output hdf5 file
        if Path(output_path).suffix in ['.h5','.hdf5']:
            self.output_path = Path(output_path)
        else:
            raise ValueError(f'{Path(output_path)} is not an .h5 or .hdf5 path')
            
        # True means we are using a mask for the loss function
        self.mask = mask
        self.io_mappings_list = io_mappings_list
        
        self.initialize_h5()
    def initialize_h5(self):
        with h5py.File(self.output_path,'w') as f:
            if 'specifier' in f:
                del f['specifier']
            f.create_dataset(
                'specifier',
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression='gzip',
                compression_opts=2,
            )
            if 'sequence' in f:
                del f['sequence']
            f.create_dataset(
                'sequence',
                (0,4,self.seq_length),
                maxshape=(None,4,self.seq_length),
                dtype=np.float16,
                compression='gzip',
                compression_opts=2
            )
            if 'methylation' in f:
                del f['methylation']
            f.create_dataset(
                'methylation',
                (0,3*self.num_cell_types,self.seq_length),
                maxshape=(None,3*self.num_cell_types,self.seq_length),
                dtype=np.float16,
                compression='gzip',
                compression_opts=2
            )
            if 'tracks' in f:
                del f['tracks']
            f.create_dataset(
                'tracks',
                (0,self.num_tracks,self.track_length),
                maxshape=(None,self.num_tracks,self.track_length),
                dtype='float',
                compression='gzip',
                compression_opts=2,
            )
            if self.mask:
                if 'mask' in f:
                    del f['mask']
                f.create_dataset(
                    'mask',
                    (0,self.num_tracks,self.track_length),
                    maxshape=(None,self.num_tracks,self.track_length),
                    dtype='bool',
                    compression='gzip',
                    compression_opts=2,
                )
            # Log the gin config string and io mappings as attributes in the HDF5 file
            gin_config_str = gin.operative_config_str()
            f.attrs['gin_config'] = gin_config_str
            f.attrs['io_mappings'] = pd.DataFrame(self.io_mappings_list).to_csv(sep='\t', index=False)
                
    def write_chunk(
        self,
        indices_list,
        sample_specifier_list,
        onehot_seq_list,
        methylation_info_list=None,
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

            start_index = min(indices_list)
            end_index = max(indices_list) + 1

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
                seq_dataset[start_index:end_index, :, :] = [np.transpose(onehot_seq,(1,0)) for onehot_seq in onehot_seq_list]
                if labels_list is None:
                    track_dataset[start_index:end_index, :, :] = np.nan
                else:
                    track_dataset[start_index:end_index, :, :] = [np.transpose(label,(1,0)) for label in labels_list]
                if methylation_info_list is None:
                    methyl_dataset[start_index:end_index, :, :] = 0
                else:
                    methyl_dataset[start_index:end_index, :, :] = [np.transpose(methyl,(1,0)) for methyl in methylation_info_list]
                if self.mask:
                    mask_dataset[start_index:end_index, :, :] = [np.transpose(mask,(1,0)) for mask in mask_list]
            except IndexError as e:
                raise IndexError(f"Indexing error with indices_list: {indices_list}. Ensure all indices are within bounds.") from e
            except ValueError as e:
                raise ValueError(f"Value assignment error: check dimensions of assigned data. {e}") from e

@gin.register
@gin.configurable
class SeqEmbeddingsWriter:
    def __init__(self, embeddings_shape, output_path):
        self.embeddings_shape = embeddings_shape
        if Path(output_path).suffix in ['.h5','.hdf5']:
            self.output_path = Path(output_path)
        else:
            raise ValueError(f'{Path(output_path)} is not an .h5 or .hdf5 path')      
        self.initialize_h5()

    def initialize_h5(self):
        with h5py.File(self.output_path,'w') as f:
            if 'embeddings' in f:
                del f['embeddings']
            f.create_dataset(
                'embeddings',
                (0,) + self.embeddings_shape,
                maxshape=(None,) + self.embeddings_shape,
                dtype='float',
                compression='gzip',
                compression_opts=2,
            )  
    def write_chunk(
        self,
        indices_list,
        embeddings_list,
    ):
        if len(indices_list)!=len(embeddings_list):
            raise ValueError(f'Cannot write chunk, unbalanced lengths: {len(indices_list)} indices and {len(embeddings_list)} embeddings.')
        end_index = np.max(np.array(indices_list))+1
        with h5py.File(self.output_path, 'a') as f:
            embeddings_dataset = f['embeddings']
            current_embeddings_size = embeddings_dataset.shape[0]
            if end_index>current_embeddings_size:
                embeddings_dataset.resize(end_index, axis=0) 
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