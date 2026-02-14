import numpy as np

def one_hot_encode_dna(dna_strand=None, cpg_methylation=None, valid_cpgs=None):
    """
    One-hot encodes a DNA strand, encoding invalid characters as zero vectors. DNA sequence, CpG methylation, and Valid CpG sides are all optional - their
    respective fields will be left blank if they are not passed.

    Parameters:
    dna_strand (str): The DNA strand to encode, represented as a string containing characters A, T, C, G, a, t, c, g, and N. Default is None.
    cpg_methylation (numpy.ndarray, optional): An optional numpy array of floating point numbers representing methylation probabilities. Default is None.

    Returns:
    numpy.ndarray: A 2D numpy array where each row represents the one-hot encoding of the corresponding nucleotide in the input DNA strand.
    """
    
    encoded_strand = None
    if dna_strand is None and cpg_methylation is None and valid_cpgs is None:
        raise ValueError("At least one of dna_strand, cpg_methylation, or valid_cpgs must be provided.")

    if dna_strand is not None:
        encoded_strand = np.zeros((len(dna_strand), 7), dtype=float)
        dna_strand = np.char.upper(np.array(list(dna_strand),dtype="U1"))
        nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for nucleotide, index in nucleotide_to_index.items():
            encoded_strand[dna_strand == nucleotide, index] = 1       
        if cpg_methylation is not None:
            if len(dna_strand) != len(cpg_methylation):
                raise ValueError("The cpg_methylation array must have the same length as the input DNA strand.")
            if not isinstance(cpg_methylation, np.ndarray):
                raise TypeError("The cpg_methylation input must be a numpy array.")
            encoded_strand[:, 4] = cpg_methylation * encoded_strand[:, 1]  # Track 4 for C methylation
            encoded_strand[:, 5] = cpg_methylation * encoded_strand[:, 2]  # Track 5 for G methylation (rev strand)

    if valid_cpgs is not None:
        if encoded_strand is None:
            encoded_strand = np.zeros((len(valid_cpgs), 7), dtype=float)
        if (
            (dna_strand is not None and len(dna_strand) != len(valid_cpgs)) 
            or (cpg_methylation is not None and len(cpg_methylation) != len(valid_cpgs))
        ):
            raise ValueError("The valid_cpgs array must have the same length as the input DNA strand.")
        if not isinstance(valid_cpgs, np.ndarray):
            raise TypeError("The valid_cpgs input must be a numpy array.")
        encoded_strand[:, 6] = valid_cpgs

    return encoded_strand