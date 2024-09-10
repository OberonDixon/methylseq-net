import numpy as np
import pysam
from pathlib import Path
from dimelo import load_processed

def one_hot_encode_dna(dna_strand, cpg_methylation=None):
    """
    One-hot encodes a DNA strand, encoding invalid characters as zero vectors. Optionally includes methylation probabilities.

    Parameters:
    dna_strand (str): The DNA strand to encode, represented as a string containing characters A, T, C, G, a, t, c, g, and N.
    cpg_methylation (numpy.ndarray, optional): An optional numpy array of floating point numbers representing methylation probabilities. Default is None.

    Returns:
    numpy.ndarray: A 2D numpy array where each row represents the one-hot encoding of the corresponding nucleotide in the input DNA strand.
    """

    # Convert the DNA strand to uppercase
    dna_strand = np.char.upper(np.array(list(dna_strand)))

    # Initialize the output array
    encoded_strand = np.zeros((len(dna_strand), 5), dtype=float)

    # Define the mapping of nucleotides to indices
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Fill in the one-hot encoding based on the nucleotide mapping
    for nucleotide, index in nucleotide_to_index.items():
        encoded_strand[dna_strand == nucleotide, index] = 1

    # Add methylation probabilities if provided
    if cpg_methylation is not None:
        if len(dna_strand) != len(cpg_methylation):
            raise ValueError("The cpg_methylation array must have the same length as the input DNA strand.")
        if not isinstance(cpg_methylation, np.ndarray):
            raise TypeError("The cpg_methylation input must be a numpy array.")
        
        encoded_strand[:, 4] = cpg_methylation

    return encoded_strand

def one_hot_dna_from_file(
    chromosome: str,
    start: int,
    end: int,
    ref_genome: str | Path,
    cpg = False,
    cpg_bedmethyl: str | Path | None = None,
    ablate_list = [],#(start,end)
):
    ref_fasta = pysam.FastaFile(ref_genome)
    dna_strand = ref_fasta.fetch(chromosome,start,end)
    strand_list = list(dna_strand)
    # print(start)
    ablate_list_inframe = [(ablate_range[0]-start,ablate_range[1]-start) for ablate_range in ablate_list if ablate_range[0]>=start and ablate_range[1]<end]
    # print(ablate_list_inframe)
    dna_strand = edit_dna_strand(dna_strand,ablate_list_inframe)
    # print(dna_strand)
        
    if cpg and cpg_bedmethyl is not None:
        cpg_mod,cpg_val = load_processed.pileup_vectors_from_bedmethyl(
            bedmethyl_file = cpg_bedmethyl,
            motif = 'CG,0',
            regions = f'{chromosome}:{start}-{end}'
        )
        non_zero_mask = cpg_val != 0
        cpg_ratio = np.zeros_like(cpg_mod, dtype=float)
        cpg_ratio[non_zero_mask] = cpg_mod[non_zero_mask] / cpg_val[non_zero_mask]
    else:
        cpg_ratio = None
    
    return one_hot_encode_dna(dna_strand=dna_strand,cpg_methylation=cpg_ratio)

def merge_ranges(ranges):
    # Sort ranges by start index
    sorted_ranges = sorted(ranges)
    merged_ranges = []
    
    for current_range in sorted_ranges:
        if not merged_ranges:
            merged_ranges.append(current_range)
        else:
            last_range = merged_ranges[-1]
            if current_range[0] <= last_range[1]:  # Check if ranges overlap
                # Merge the ranges
                merged_ranges[-1] = (last_range[0], max(last_range[1], current_range[1]))
            else:
                merged_ranges.append(current_range)
    
    return merged_ranges

def edit_dna_strand(
    original_string,
    ranges,
    replace_with = 'A',
):
    # Sort ranges to ensure correct order of replacement
    ranges = sorted(ranges)
    ranges = merge_ranges(ranges)
    
    # Initialize a list to store parts of the final string
    parts = []
    last_index = 0
    
    for start, end in ranges:
        # Append the part of the string before the current range
        parts.append(original_string[last_index:start])
        # Append 'N' for the length of the current range
        parts.append(replace_with * (end - start))
        # Update the last index to the end of the current range
        last_index = end
    
    # Append the remaining part of the string after the last range
    parts.append(original_string[last_index:])
    
    # Join all parts together to form the final modified string
    modified_string = ''.join(parts)
    
    return modified_string