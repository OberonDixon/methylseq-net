import pysam
import h5py
from pathlib import Path
import numpy as np
import gin

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
    ):
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
        self.initialize_h5()
    def initialize_h5(self):
        with h5py.File(self.output_path,'w') as f:
            if 'region' in f:
                del f['region']
            f.create_dataset(
                'region',
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
            # Log the gin config string as an attribute in the HDF5 file
            gin_config_str = gin.operative_config_str()
            f.attrs['gin_config'] = gin_config_str
    def write_chunk(
        self,
        regions_list,
        onehot_seq_list,
        labels_list,
        mask_list=None,
    ):
        if len(onehot_seq_list)!=len(labels_list):
            raise ValueError(f'Cannot write chunk, unbalanced lengths:{len(onehot_seq_list)} sequences and {len(labels_list)} labels.')
        if len(onehot_seq_list[0])!=self.seq_length:
            raise ValueError(f'Cannot write chunk, seq length is {len(onehot_seq_list[0])} and should be {self.seq_length}.')
        if labels_list[0].shape[0]!=self.track_length:
            raise ValueError(f'Cannot write chunk, track length is {labels_list[0].shape[0]} and should be {self.track_length}.')

        if self.mask:
            if mask_list is None:
                raise ValueError("Datasetwriter initialized with 'mask=True', must provide a mask_list when writing chunk. Got 'None'")
            if mask_list[0].shape[0]!=self.track_length:
                raise ValueError(f'Cannot write chunk, mask length is {mask_list[0].shape[0]} and should be {self.track_length}.')
                
            
        with h5py.File(self.output_path, 'a') as f:
            regions_dataset = f['region']
            seq_dataset = f['sequence']
            track_dataset = f['tracks']

            current_regions_size = regions_dataset.shape[0]
            current_seq_size = seq_dataset.shape[0]
            current_track_size = track_dataset.shape[0]          
            
            new_regions_size = current_regions_size + len(regions_list)
            new_seq_size = current_seq_size + len(onehot_seq_list)
            new_track_size = current_track_size + len(labels_list)            

            regions_dataset.resize(new_regions_size, axis=0)
            seq_dataset.resize(new_seq_size, axis=0)
            track_dataset.resize(new_track_size, axis=0)            

            regions_dataset[current_regions_size:new_regions_size] = [f"{region['chrom']}:{region['start']}-{region['end']}" for region in regions_list]
            seq_dataset[current_seq_size:new_seq_size, :, :] = [np.transpose(onehot_seq,(1,0)) for onehot_seq in onehot_seq_list]
            track_dataset[current_track_size:new_track_size, :, :] = [np.transpose(label,(1,0)) for label in labels_list]
            
            if self.mask:
                mask_dataset = f['mask']
                current_mask_size = mask_dataset.shape[0]
                new_mask_size = current_mask_size + len(mask_list)
                mask_dataset.resize(new_mask_size, axis=0)
                mask_dataset[current_mask_size:new_mask_size, :, :] = [np.transpose(mask,(1,0)) for mask in mask_list]