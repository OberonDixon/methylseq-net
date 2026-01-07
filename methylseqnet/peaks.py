import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path

from methylseqnet.dataset import MultiMethylDataset

def selected_peaks_from_target(target, peak_threshold, min_peak_distance_bins):
    peak_indices = (target > peak_threshold).nonzero()[0]
    if len(peak_indices) > 0:
        # Find breakpoints where indices are not consecutive
        breaks = np.where(np.diff(peak_indices) > 1)[0] + 1
        
        # Split into groups of consecutive indices
        peak_groups = np.split(peak_indices, breaks)
        
        # Select max value index from each group
        merged_peak_indices = np.array([
            group[np.argmax(target[group])]
            for group in peak_groups
        ])
    else:
        merged_peak_indices = np.array([])
    selected_peaks = []
    if len(merged_peak_indices) > 0:
        # Enforce minimum peak distance
        last_selected_peak = -min_peak_distance_bins - 1
        for peak_idx in merged_peak_indices:
            if peak_idx - last_selected_peak >= min_peak_distance_bins:
                selected_peaks.append(peak_idx)
                last_selected_peak = peak_idx
    return selected_peaks

def generate_peaks_bed_from_dataset(
    dataset_path,
    output_directory,
    label_substrings,
    data_type='ATAC-seq',
    peak_threshold=10,
    min_peak_distance=50,
    target_bin_size=128,
    num_peaks=1000,
    random_seed=42
):
    dataset = MultiMethylDataset(dataset_path)
    io_mappings_df = dataset.get_io_mappings_df()
    peaks = {}
    for label_substring in label_substrings:
        matching_io_mappings = io_mappings_df[io_mappings_df['label_files'].str.upper().str.contains(label_substring.upper())]
        if len(matching_io_mappings) > 1:
            raise ValueError(f"Multiple io mappings match the label substring '{label_substring}': {matching_io_mappings}")
        elif len(matching_io_mappings) == 0:
            raise ValueError(f"No io mappings match the label substring '{label_substring}'")
        else:
            channel = matching_io_mappings.iloc[0]['channel']
            g = torch.Generator()
            g.manual_seed(random_seed)
            dataloader = DataLoader(dataset, batch_size=None, shuffle=True, generator=g)
            peak_strings = []
            with tqdm(total=num_peaks, desc=f"Generating peaks for {label_substring}") as pbar:
                for sample in dataloader:
                    region_str = sample['specifier'].split('|')[0]
                    chrom = region_str.split(':')[0]
                    start = int(region_str.split(':')[1].split('-')[0])
                    end = int(region_str.split(':')[1].split('-')[1])
                    target = sample['target'][0,channel,:].numpy()
                    if (end - start) // target_bin_size != target.shape[0]:
                        raise ValueError(f"Target length {target.shape[0]} does not match expected length {(end - start) // target_bin_size} for region {region_str}")
                    selected_peak_indices = selected_peaks_from_target(target, peak_threshold, min_peak_distance // target_bin_size)
                    peak_strings.extend([f"{chrom}\t{start + idx * target_bin_size}\t{start + (idx + 1) * target_bin_size}\t{data_type}_peak" for idx in selected_peak_indices])
                    pbar.update(len(selected_peak_indices))
                    if len(peak_strings) >= num_peaks:
                        peak_strings = peak_strings[:num_peaks]
                        pbar.update(num_peaks)
                        break
            if os.path.exists(output_directory) is False:
                os.makedirs(output_directory)
            with open(Path(output_directory) / f'{label_substring}_{data_type}.bed', 'w') as f:
                f.write('\n'.join(peak_strings))


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate peaks bed files from a preprocessed dataset object with io_mappings")
    parser.add_argument("--dataset-paths", type=str, nargs='+', required=True, help="Path to the preprocessed dataset object (HDF5 file)")
    parser.add_argument("--output-directory", type=str, required=True, help="Path to save the output BED file")
    parser.add_argument("--label-substrings", type=str, nargs='+', required=True, help="List of substrings to identify label channel from io mappings")
    parser.add_argument("--data-type", type=str, default='ATAC-seq', help="Type of data to process (e.g., 'ATAC-seq', 'ChIP-seq')")
    parser.add_argument("--peak-threshold", type=float, default=10, help="Threshold to define peaks in preprocessed dataset")
    parser.add_argument("--num-peaks", type=int, default=1000, help="Number of peaks to extract per cell type")
    parser.add_argument("--min-peak-distance", type=int, default=16384, help="Minimum distance between peaks")
    parser.add_argument("--target-bin-size", type=int, default=128, help="Size of target bins in the preprocessed dataset")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    generate_peaks_bed_from_dataset(
        dataset_path=args.dataset_paths,
        output_directory=args.output_directory,
        label_substrings=args.label_substrings,
        data_type=args.data_type,
        peak_threshold=args.peak_threshold,
        num_peaks=args.num_peaks,
        min_peak_distance=args.min_peak_distance,
        target_bin_size=args.target_bin_size,
        random_seed=args.random_seed
    )