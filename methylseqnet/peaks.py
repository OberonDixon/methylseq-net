from dimelo import load_processed
import pysam
import numpy as np

def peaks_from_bedmethyl(
    bedmethyl_file,
    fasta_file,
    chromosome,
    bin_size=128,
    threshold=0.05,
):
    # get contig length
    ref_fasta = pysam.FastaFile(fasta_file)
    contig_length = ref_fasta.get_reference_length(chromosome)
    # run load processed
    whole_chrom_track_mod,whole_chrom_track_val = load_processed.pileup_vectors_from_bedmethyl(
        bedmethyl_file = bedmethyl_file,
        motif = 'A,0',
        regions = f'{chromosome}:{0}-{(contig_length//bin_size)*bin_size}'
    ) 
    track_mod_sums = whole_chrom_track_mod.reshape(contig_length//bin_size, 128).sum(axis=1)
    track_val_sums = whole_chrom_track_val.reshape(contig_length//bin_size, 128).sum(axis=1)
    non_zero_mask = track_val_sums != 0
    track_ratio = np.zeros_like(track_mod_sums, dtype=float)
    track_ratio[non_zero_mask] = track_mod_sums[non_zero_mask] / track_val_sums[non_zero_mask]
    track_peaks = track_ratio > threshold
    
    return track_peaks