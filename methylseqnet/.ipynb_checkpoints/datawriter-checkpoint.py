import pysam
import h5py
from pathlib import Path

class DatasetWriter:
    def __init__(
        self,
        seq_length: int,
        cpg_input: bool,
        track_length: int,
        num_tracks: int,
        output_path: str | Path,
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
        self.initialize_h5()
    def initialize_h5(self):
        with h5py.File(self.output_path,'w') as f:
            if 'sequence' in f:
                del f['sequence']
            f.create_dataset(
                'sequence',
                (0,self.seq_length,5),
                maxshape=(None,self.seq_length,5),
                dtype='float',
                compression='gzip',
                compression_opts=5
            )
            if 'tracks' in f:
                del f['tracks']
            f.create_dataset(
                'tracks',
                (0,self.num_tracks),
                maxshape=(None,self.num_tracks),
                dtype='bool',
                compression='gzip',
                compression_opts=5,
            )
    def write_chunk(
        self,
        onehot_seq_list,
        labels_list,
    ):
        if len(onehot_seq_list)!=len(labels_list):
            raise ValueError(f'Cannont write chunk, unbalanced lengths:{len(onehot_seq_list)} sequences and {len(labels_list)} labels.')
        if len(onehot_seq_list[0])!=self.seq_length:
            raise ValueError(f'Cannot write chunk, seq length is {len(onehot_seq_list[0])} and should be {seq_length}.')
        if len(labels_list[0])!=self.track_length:
            raise ValueError(f'Cannot write chunk, seq length is {len(labels_list[0])} and should be {track_length}.')
            
        with h5py.File(self.output_path, 'a') as f:
            seq_dataset = f['sequence']
            track_dataset = f['tracks']

            current_seq_size = seq_dataset.shape[0]
            current_track_size = track_dataset.shape[0]

            new_seq_size = current_seq_size + len(onehot_seq_list)
            new_track_size = current_track_size + len(labels_list)

            seq_dataset.resize(new_seq_size, axis=0)
            track_dataset.resize(new_track_size, axis=0)

            seq_dataset[current_seq_size:new_seq_size, :, :] = onehot_seq_list
            track_dataset[current_track_size:new_track_size, :] = labels_list
