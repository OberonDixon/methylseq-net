from methylseqnet import dna_io
from dimelo import parse_bam, load_processed
from methylseqnet.datawriter import DatasetWriter
from pathlib import Path
import pysam
import numpy as np
from tqdm import tqdm
from Bio.Seq import Seq
import time
    
genome_path = '/clusterfs/nilah/oberon/jupyter/chm13.draft_v1.0.fasta'

CpG_pileup = '/clusterfs/nilah/oberon/datasets/deep_ctcf/cpg_bam_whole_genome/pileup.sorted.bed.gz'
mA_pileup = '/clusterfs/nilah/oberon/datasets/deep_ctcf/allcontext_bam_whole_genome/pileup.sorted.bed.gz'

write_chunk_len = 1000
early_stop = 1_000_000_000 # set to less than length of chrom if you want to process less than the whole chrom

bin_size = 128
seq_input_bins = 7
seq_length = bin_size * seq_input_bins
cpg_input = True
track_length = 1
# track is centered on seq_input, each track pred covering bin_size bp
num_tracks = 1
# track index: (source file, motif to extract)
track_file = mA_pileup
track_motif = 'A,0'
# track index: mod fraction threshold, None means floating point track
track_threshold = 0.05

# (source file, motif to extract)
cpg_file = CpG_pileup
cpg_motif = 'CG,0'

datasets_dict = {
    # f'/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/train_th{track_threshold}_fd6d75e.h5':['chr1','chr2','chr3','chr4','chr5','chr6','chr8','chr9','chr10','chr11','chr12','chr13','chr15','chr16','chr17','chr18','chr21'],
    # f'/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/validation_th{track_threshold}_fd6d75e.h5':['chr7','chr20'],
    f'/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/chrX_ALL_th{track_threshold}_25959f3e.h5':['chrX'],
}

ref_fasta = pysam.FastaFile(genome_path)
for dataset_path,chromosomes in datasets_dict.items():
    dataset_writer = DatasetWriter(
        seq_length=seq_length,
        cpg_input=cpg_input,
        track_length=track_length,
        num_tracks=num_tracks,
        output_path=dataset_path,
    )
    for chromosome in chromosomes:
        contig_length = ref_fasta.get_reference_length(chromosome)
        # Load up seq, cpg, mA for whole chromosome
        start_time = time.time()
        print(f'loading {chromosome} sequence from fasta')
        whole_chrom_sequence = ref_fasta.fetch(chromosome,0,contig_length)
        print('took',time.time()-start_time)
        
        if cpg_input:
            start_time = time.time()
            print(f'loading {chromosome} cpg from bedmethyl')
            whole_chrom_cpg_mod,whole_chrom_cpg_val = load_processed.pileup_vectors_from_bedmethyl(
                bedmethyl_file = cpg_file,
                motif = cpg_motif,
                regions = f'{chromosome}:{0}-{contig_length}'
            )
            print('took',time.time()-start_time)
        start_time = time.time()
        print(f'loading {chromosome} track from bedmethyl')
        whole_chrom_track_mod,whole_chrom_track_val = load_processed.pileup_vectors_from_bedmethyl(
            bedmethyl_file = track_file,
            motif = track_motif,
            regions = f'{chromosome}:{0}-{contig_length}'
        )       
        print('took',time.time()-start_time)
        # Loop through chromosome in seq_len size chunks
        onehot_seq_list = []
        track_value_list = []
        # the -2*seq_length is because we want to be guaranteed that our reaching outward for 
        # edge-of-context peaks doesn't go past the end of the contig
        for chunk_start in tqdm(range(128*3,min(contig_length-3*seq_length//2,early_stop),seq_length)):
            chunk_end = chunk_start + seq_length
            # For each chunk, we find the peaks then from there decide whether to send one or more seqs to the dataset
            # track_mod,track_val = load_processed.pileup_vectors_from_bedmethyl(
            #     bedmethyl_file = track_file,
            #     motif = track_motif,
            #     regions = f'{chromosome}:{chunk_start}-{chunk_end}',
            # )
            track_mod_sums = whole_chrom_track_mod[chunk_start:chunk_end].reshape(seq_input_bins, 128).sum(axis=1)
            track_val_sums = whole_chrom_track_val[chunk_start:chunk_end].reshape(seq_input_bins, 128).sum(axis=1)
            # To safely handle division by zero, create a mask for non-zero denominators
            non_zero_mask = track_val_sums != 0

            # Initialize the ratio array with zeros
            track_ratio = np.zeros_like(track_mod_sums, dtype=float)

            # Perform division only where the denominator is non-zero
            track_ratio[non_zero_mask] = track_mod_sums[non_zero_mask] / track_val_sums[non_zero_mask]
            track_peaks = track_ratio > track_threshold
            for track_chunk_index in range(len(track_ratio)):
                # # If we are in a peak OR adjacent to a peak and NOT in a peak, then we want to use this as a data point
                # if (track_peaks[track_chunk_index] 
                #     or (# if we are not currently in a peak
                #         not track_peaks[track_chunk_index] 
                #         and (# if we are just right of a peak
                #             (track_chunk_index-1>=0 and track_peaks[track_chunk_index-1]) or 
                #             # if we are just left of a peak
                #              (track_chunk_index+1<seq_input_bins and track_peaks[track_chunk_index+1])
                #             )
                #        )
                #    ): 
                track_value = track_peaks[track_chunk_index]
                # the center of the peak is determined from the large chunk location plus the track minichunk index
                seq_center = (2*chunk_start + 2*track_chunk_index*bin_size + bin_size) // 2
                # from the center of the peak (or the not-peak-but-adjacent) we define a sequence context
                seq_start = seq_center - (seq_length // 2)
                seq_end = seq_center + (seq_length // 2)
                # print(chromosome,chunk_start,chunk_end,seq_center,seq_start,seq_end)
                # sequence = ref_fasta.fetch(chromosome,seq_start,seq_end)
                sequence = whole_chrom_sequence[seq_start:seq_end]
                rev_comp_sequence = str(Seq(sequence).reverse_complement())
                if cpg_input:
                    # cpg_mod,cpg_val = load_processed.pileup_vectors_from_bedmethyl(
                    #     bedmethyl_file = cpg_file,
                    #     motif = cpg_motif,
                    #     regions = f'{chromosome}:{seq_start}-{seq_end}'
                    # )
                    cpg_mod = whole_chrom_cpg_mod[seq_start:seq_end]
                    cpg_val = whole_chrom_cpg_val[seq_start:seq_end]
                    # To safely handle division by zero, create a mask for non-zero denominators
                    non_zero_mask = cpg_val != 0

                    # Initialize the ratio array with zeros
                    cpg_ratio = np.zeros_like(cpg_mod, dtype=float)

                    # Perform division only where the denominator is non-zero
                    cpg_ratio[non_zero_mask] = cpg_mod[non_zero_mask] / cpg_val[non_zero_mask]
                else:
                    cpg_ratio = None
                onehot_seq_list.append(dna_io.one_hot_encode_dna(sequence,cpg_ratio))
                onehot_seq_list.append(dna_io.one_hot_encode_dna(rev_comp_sequence,cpg_ratio[::-1]))
                track_value_list.append(np.array([track_value]))
                track_value_list.append(np.array([track_value]))
#             # sequence = ref_fasta.fetch(chromosome,chunk_start,chunk_end)
#             # no special seq centered on anything, just the chunk itself
#             sequence = whole_chrom_sequence[chunk_start:chunk_end]
#             rev_comp_sequence = str(Seq(sequence).reverse_complement())
#             if cpg_input:
#                 # cpg_mod,cpg_val = load_processed.pileup_vectors_from_bedmethyl(
#                 #     bedmethyl_file = cpg_file,
#                 #     motif = cpg_motif,
#                 #     regions = f'{chromosome}:{chunk_start}-{chunk_end}'
#                 # )
#                 cpg_mod = whole_chrom_cpg_mod[chunk_start:chunk_end]
#                 cpg_val = whole_chrom_cpg_val[chunk_start:chunk_end]
#                 # To safely handle division by zero, create a mask for non-zero denominators
#                 non_zero_mask = cpg_val != 0

#                 # Initialize the ratio array with zeros
#                 cpg_ratio = np.zeros_like(cpg_mod, dtype=float)

#                 # Perform division only where the denominator is non-zero
#                 cpg_ratio[non_zero_mask] = cpg_mod[non_zero_mask] / cpg_val[non_zero_mask]
#             else:
#                 cpg_ratio = None    
#             onehot_seq_list.append(dna_io.one_hot_encode_dna(sequence,cpg_ratio))
#             onehot_seq_list.append(dna_io.one_hot_encode_dna(rev_comp_sequence,cpg_ratio[::-1]))
#             track_value_list.append(np.array([track_peaks[3]]))
#             track_value_list.append(np.array([track_peaks[3]]))
                    
            if len(onehot_seq_list)>=write_chunk_len:
                dataset_writer.write_chunk(onehot_seq_list,track_value_list)
                onehot_seq_list = []
                track_value_list = []
                
                    
            