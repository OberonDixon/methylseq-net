import pysam
from Bio import motifs
from Bio.Seq import Seq
import numpy as np

def adjust_coordinates(data, motif_len, contig_length):
    return [
        (
            coord + motif_len // 2 if coord >= 0 else contig_length + coord + motif_len // 2,
            float_value
        )
        for coord, float_value in data
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