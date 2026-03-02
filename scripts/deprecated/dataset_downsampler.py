import h5py
from pathlib import Path
import numpy as np
import argparse

def downsample_h5(input_path, output_path, max_samples=100):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    with h5py.File(input_path, 'r') as src, h5py.File(output_path, 'w') as dst:
        # Copy global attributes
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        
        def copy_group(src_group, dst_group):
            for name, item in src_group.items():
                if isinstance(item, h5py.Dataset):
                    # Determine truncated shape
                    shape = list(item.shape)
                    shape[0] = min(shape[0], max_samples)
                    # Slice the data along axis 0
                    data = item[:shape[0]]
                    # Copy dataset creation properties
                    dset_kwargs = {
                        'dtype': item.dtype,
                        'compression': item.compression,
                        'compression_opts': item.compression_opts,
                        'chunks': item.chunks,
                        'shuffle': item.shuffle,
                        'fletcher32': item.fletcher32,
                    }
                    # Create the new dataset with original metadata
                    dst_dset = dst_group.create_dataset(name, data=data, **{k: v for k, v in dset_kwargs.items() if v is not None})
                    # Copy dataset attributes
                    for attr_key, attr_val in item.attrs.items():
                        dst_dset.attrs[attr_key] = attr_val

                elif isinstance(item, h5py.Group):
                    sub_group = dst_group.create_group(name)
                    copy_group(item, sub_group)

        copy_group(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample an HDF5 dataset along axis 0.")
    parser.add_argument("input", type=str, help="Path to input HDF5 file")
    parser.add_argument("output", type=str, help="Path to output (downsampled) HDF5 file")
    parser.add_argument("max_samples", type=int, nargs="?", default=10, help="Max samples to retain (default: 10)")

    args = parser.parse_args()
    downsample_h5(args.input, args.output, args.max_samples)
