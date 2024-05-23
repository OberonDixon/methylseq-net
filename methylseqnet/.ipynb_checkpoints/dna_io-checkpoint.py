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
    list: A list of lists, where each inner list represents the one-hot encoding of the corresponding nucleotide in the input DNA strand.
    """

    if cpg_methylation is not None:
        if len(dna_strand) != len(cpg_methylation):
            raise ValueError("The cpg_methylation array must have the same length as the input DNA strand.")
        if not isinstance(cpg_methylation, np.ndarray):
            raise TypeError("The cpg_methylation input must be a numpy array.")

    # Initialize an empty list to store the one-hot encoded nucleotides
    encoded_strand = []

    # Loop through the input DNA strand
    for idx, nucleotide in enumerate(dna_strand):
        # Convert the nucleotide to uppercase to treat upper and lower case characters the same
        nucleotide = nucleotide.upper()

        # Get the methylation probability, if available
        methylation_prob = cpg_methylation[idx] if cpg_methylation is not None else 0

        # One-hot encode the nucleotide and append it to the encoded_strand list
        if nucleotide == 'A':
            encoded_strand.append([1, 0, 0, 0, methylation_prob])
        elif nucleotide == 'T':
            encoded_strand.append([0, 1, 0, 0, methylation_prob])
        elif nucleotide == 'C':
            encoded_strand.append([0, 0, 1, 0, methylation_prob])
        elif nucleotide == 'G':
            encoded_strand.append([0, 0, 0, 1, methylation_prob])
        else:  # Invalid characters or 'N' will be encoded as zero vectors
            encoded_strand.append([0, 0, 0, 0, methylation_prob])

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
    ablate_list_inframe = [(ablate_range[0]-start,ablate_range[1]-start) for ablate_range in ablate_list if ablate_range[0]>=start and ablate_range[1]<end]
    
    dna_strand = edit_dna_strand(dna_strand,ablate_list_inframe)
        
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

def edit_dna_strand(
    dna_strand,
    ranges,
    replace_with = 'N',
):
    # Sort ranges to ensure correct order of replacement
    ranges = sorted(ranges)
    
    # Initialize a list to store parts of the final string
    parts = []
    last_index = 0
    
    for start, end in ranges:
        # Append the part of the string before the current range
        parts.append(original_string[last_index:start])
        # Append 'N' for the length of the current range
        parts.append('N' * (end - start))
        # Update the last index to the end of the current range
        last_index = end
    
    # Append the remaining part of the string after the last range
    parts.append(original_string[last_index:])
    
    # Join all parts together to form the final modified string
    modified_string = ''.join(parts)
    
    return modified_string