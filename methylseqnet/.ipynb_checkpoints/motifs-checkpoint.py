# import pysam
from Bio import motifs
from Bio.Seq import Seq
from Bio.motifs.matrix import PositionWeightMatrix
import numpy as np

def adjust_coordinates(data, motif_len, contig_length):
    return [
        (
            contig_length + coord + motif_len // 2,#coord + motif_len // 2, #else 
            float_value
        )
        for coord, float_value in data if coord <= 0
    ]

def bin_and_max(matches, bin_size, contig_length):
    # Convert the list of tuples to a NumPy array for efficiency
    adjusted_data = np.array(matches)
    coordinates = adjusted_data[:, 0].astype(int)
    float_values = adjusted_data[:, 1].astype(float)
    
    # Calculate bin indices
    bins = coordinates // bin_size
    
    # Initialize an array to hold the max values per bin
    max_values = np.full((contig_length // bin_size + 1,), 0)
    
    # Iterate through the bins and find the max value for each bin
    for bin_index, value in zip(bins, float_values):
        if value > max_values[bin_index]:
            max_values[bin_index] = value
    
    return max_values[0:-1]

def find_motifs(
    genome_fasta,
    motif_pfm,
    chromosome,
    start,
    end,
):
    fastafile = pysam.FastaFile(genome_fasta)
    contig_length = fastafile.get_reference_length(chromosome)
    sequence = Seq(fastafile.fetch(chromosome,start,min(end,contig_length)))
    sequence_rc = sequence.reverse_complement()
    with open(motif_pfm,'r') as f:
        forward_pfm = motifs.read(f,'pfm',strict=False)
    forward_pwm = forward_pfm.counts.normalize(pseudocounts=0.1).log_odds()
    motif_len = len(forward_pwm[0])
    matches = list(forward_pwm.search(sequence))
    adjusted_matches = adjust_coordinates(matches,motif_len,contig_length)
    
    return adjusted_matches, contig_length

def find_ctcf_motifs(
    genome_fasta,
    chromosome,
    start,
    end,
):
    fastafile = pysam.FastaFile(genome_fasta)
    contig_length = fastafile.get_reference_length(chromosome)
    sequence = Seq(fastafile.fetch(chromosome,start,min(end,contig_length)))
    sequence_rc = sequence.reverse_complement()
    # building motif based on https://compbio.mit.edu/encode-motifs/
    pwm_str = """
    >CTCF_known1	CTCF_1	CTCF_jaspar_MA0139.1
    Y 0.095290 0.318729 0.083242 0.502739
    D 0.182913 0.158817 0.453450 0.204819
    R 0.307777 0.053669 0.491785 0.146769
    C 0.061336 0.876233 0.023001 0.039430
    C 0.008762 0.989047 0.000000 0.002191
    A 0.814896 0.014239 0.071194 0.099671
    S 0.043812 0.578313 0.365827 0.012048
    Y 0.117325 0.474781 0.052632 0.355263
    A 0.933114 0.012061 0.035088 0.019737
    G 0.005488 0.000000 0.991219 0.003293
    R 0.365532 0.003293 0.621296 0.009879
    K 0.059276 0.013172 0.553238 0.374314
    G 0.013187 0.000000 0.978022 0.008791
    G 0.061538 0.008791 0.851649 0.078022
    C 0.114411 0.806381 0.005501 0.073707
    R 0.409241 0.014301 0.557756 0.018702
    S 0.090308 0.530837 0.338106 0.040749
    Y 0.128855 0.354626 0.080396 0.436123
    V 0.442731 0.199339 0.292952 0.064978
    """
    # Split the input string into lines and then into components
    lines = pwm_str.strip().split('\n')
    header = lines[0]  # This is the header, can be ignored for now

    # Extracting the data
    matrix_data = []
    for line in lines[1:]:
        parts = line.split()
        # Ignore the IUPAC letter at parts[0]
        probabilities = list(map(float, parts[1:]))
        matrix_data.append(probabilities)

    # Transpose to match the expected format for a PWM (4xN matrix)
    matrix_data = np.array(matrix_data).T.tolist()

    # Create the dictionary expected by PositionWeightMatrix
    nucleotides = ['A', 'C', 'G', 'T']
    pwm_dict = {nucleotide: values for nucleotide, values in zip(nucleotides, matrix_data)}

    # Create the PWM
    forward_pwm = PositionWeightMatrix(nucleotides, pwm_dict).log_odds()
    motif_len = len(forward_pwm[0])
    matches = list(forward_pwm.search(sequence))
    adjusted_matches = adjust_coordinates(matches,motif_len,contig_length)
    
    return adjusted_matches, contig_length