from optparse import OptionParser
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pysam
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import methylseqnet
#from methylseqnet.dataset import CustomH5Dataset
from methylseqnet.motif_helpers import insert_center_pos, shuffle_peak, create_subsets

def main():
    usage = 'usage: %prog [options] <PWMS_TOP_DIR> <PEAKS_TOP_DIR> <PEAKS_OUTPUT_DIR>'
    parser = OptionParser(usage)

    parser.add_option('--INPUT_LEN', dest='INPUT_LEN',
        type='int',
        default=524288, # default to Borzoi context
        help='Sequence input length [Default: %default]')
    parser.add_option('--PEAK_LEN', dest='PEAK_LEN',
        type='int',
        default=128, # default bin size
        help='Peak length [Default: %default]')
    parser.add_option('--N', dest='N',
        default=5, # default to 5 trials per motif/peak pair
        help='Number of trials for each motif insertion [Default: %default]')
    parser.add_option('--REFERENCE_GENOME', dest='REFERENCE_GENOME',
        default='/clusterfs/nilah/ayesha/genomes/hg38.ml.fa', # default hg38 path
        help='Reference genome fasta path [Default: %default]')
    (options, args) = parser.parse_args()

    if len(args) == 3:
        PWMS_TOP_DIR = args[0]
        PEAKS_TOP_DIR = args[1]
        PEAKS_OUTPUT_DIR = args[2]
    else:
        parser.error('Must provide parameters PWMS_TOP_DIR, PEAKS_TOP_DIR, and PEAKS_OUTPUT_DIR')

    if not os.path.isdir(PEAKS_OUTPUT_DIR):
        os.mkdir(PEAKS_OUTPUT_DIR)

    # calculate pad length needed to fill input
    INPUT_LEN = options.INPUT_LEN
    PEAK_LEN = options.PEAK_LEN
    N = options.N
    PAD_LEN = (INPUT_LEN-PEAK_LEN)//2
    print(f"Model input sequence length is: {INPUT_LEN}")
    print(f"Peak sequence length is: {PEAK_LEN}")

    # reference genome
    hg38_fasta = pysam.Fastafile(options.REFERENCE_GENOME)
    chrom_lens_dict = dict(zip(hg38_fasta.references, hg38_fasta.lengths))
    print(chrom_lens_dict)

    # PWMs for selection, or otherwise for each human CIS-BP TF (~700)
    # PWMS_TOP_DIR is required so we always have a fallback
    TFS = []
    if os.path.exists('transcription_factors.txt'):
        with open('transcription_factors.txt', 'r') as file:
            TFS = file.read().splitlines()
    else:
        TFS = os.listdir(f"{PWMS_TOP_DIR}/pwms/")
        TFS = [t.split(".csv")[0] for t in TFS]
    print(f"TFs are: {TFS}")
    print(f"{len(TFS)} TFs total")

    # process pwms to be ready for sampling
    pwms = {}
    for tf in TFS:
        pwms[tf] = pd.read_csv(f'{PWMS_TOP_DIR}/pwms/{tf}.csv', index_col=0, skiprows=1, header=None)
        pwms[tf] = pwms[tf]/pwms[tf].sum(axis=0)  # columns sum to 1
    
    # tissue list (from peak files)
    TISSUES = os.listdir(f'{PEAKS_TOP_DIR}/')
    TISSUES = [tis.split('.hg38.bed')[0] for tis in TISSUES if 'peaks' in tis]
    print(f"Tissues are: {TISSUES}")
    # TISSUES = ['Hepatocyte_peaks', 'Adipocyte_peaks'] # dummy for testing, want to run on full list eventually
    TISSUES = ['Hepatocyte_peaks'] # dummy for testing, want to run on full list eventually

    ### Create endogenous sequences of desired length ###
    # Centered at peaks from bed files
    for tis in TISSUES:
        print(f"Writing endogenous peaks for {tis}...")
        peaks = pd.read_csv(f'{PEAKS_TOP_DIR}/{tis}.hg38.bed', sep='\t', names=["Chromosome", "Start", "End"])
        
        # pad provided peaks with endogenous reference sequence
        endogenous_peaks = []
        for i, row in peaks.iterrows():
            try:
                source = row["Chromosome"]
                # Convert 1-based start to 0-based before padding
                # The fetched sequence will now be INPUT_LEN long
                start_0based = row["Start"] - 1 
                start, end = start_0based - PAD_LEN, row["End"] + PAD_LEN
                
                start_pad = 0 - min(start,0)
                chrom_length = chrom_lens_dict[source]
                end_pad = max(end,chrom_length) - chrom_length
                
                # pad peak with Ns if not enough context
                seq = start_pad*"N" + hg38_fasta.fetch(source,
                                                       max(start, 0),
                                                       min(end, chrom_length)) + end_pad*"N"
            except ValueError:
                continue
            endogenous_peaks.append(SeqRecord(Seq(seq), id=f'{i}'))
        
        # save sequence fasta file
        if not os.path.exists(f'{PEAKS_OUTPUT_DIR}/{tis}'):
            os.mkdir(f'{PEAKS_OUTPUT_DIR}/{tis}')
        SeqIO.write(endogenous_peaks, f'{PEAKS_OUTPUT_DIR}/{tis}/endogenous_sequences_{INPUT_LEN}.fasta', "fasta")
    
    print("Done writing endogenous sequences.")

    ### Create endogenous sequences of desired length with SHUFFLED CENTER PEAKS ###
    for tis in TISSUES:
        print(f"Writing endogenous with shuffled peaks for {tis}...")
        peaks = pd.read_csv(f'{PEAKS_TOP_DIR}/{tis}.hg38.bed', sep='\t', names=["Chromosome", "Start", "End"])
        
        # pad provided peaks with endogenous reference sequence
        endogenous_shuffled_peaks = []
        for i, row in peaks.iterrows():
            try:
                source = row["Chromosome"]
                # Convert 1-based start to 0-based before padding
                # The fetched sequence will now be INPUT_LEN long
                start_0based = row["Start"] - 1 
                start, end = start_0based - PAD_LEN, row["End"] + PAD_LEN
                
                start_pad = 0 - min(start,0)
                chrom_length = chrom_lens_dict[source]
                end_pad = max(end,chrom_length) - chrom_length
                
                # pad peak with Ns if not enough context
                seq = start_pad*"N" + hg38_fasta.fetch(source,
                                                       max(start, 0),
                                                       min(end, chrom_length)) + end_pad*"N"
                # shuffle peak portion
                peak_start = (len(seq)-PEAK_LEN)//2
                peak_end = peak_start + PEAK_LEN
                shuffled_peak = shuffle_peak(seq, peak_start, peak_end)
                seq = seq[:peak_start] + shuffled_peak + seq[peak_end:]
            except ValueError:
                continue
            endogenous_shuffled_peaks.append(SeqRecord(Seq(seq), id=f'{i}'))
    
        # save sequence fasta file
        if not os.path.exists(f'{PEAKS_OUTPUT_DIR}/{tis}'):
            os.mkdir(f'{PEAKS_OUTPUT_DIR}/{tis}')
        SeqIO.write(endogenous_shuffled_peaks, f'{PEAKS_OUTPUT_DIR}/{tis}/endogenous_shuffled_peak_sequences_{INPUT_LEN}.fasta', "fasta")
    
    print("Done writing endogenous sequences with shuffled center peaks.")

    ### Create fasta files with motif sequences inserted into tissue-specific peaks ###
    for tis in TISSUES:
        print(f"Writing motif-inserted peaks for {tis}...")
    
        # load endogenous peak sequences
        records = list(SeqIO.parse(f"{PEAKS_OUTPUT_DIR}/{tis}/endogenous_sequences_{INPUT_LEN}.fasta", "fasta"))
        seqs = [str(i.seq) for i in records]
        
        for tf in TFS:
            # IMPORTANT: don't re-write motif insertion file if it's already created for that tf
            tf_motif_insertion_path = f'{PEAKS_OUTPUT_DIR}/{tis}/motif_inserted_sequences_{INPUT_LEN}_{tf}.fasta'
            if os.path.exists(tf_motif_insertion_path):
                print(f"Skipped {tf_motif_insertion_path}")
                continue
            
            tf_motif_seqs = []
            pwm = pwms[tf]
        
            # make motif-inserted sequences
            for i, seq in enumerate(seqs):
                
                trials = insert_center_pos(seq, pwm, PEAK_LEN, N, shuffle=True)
        
                for j, trial in enumerate(trials):
                    tf_motif_seqs.append(SeqRecord(Seq(trial), id=f"{i}_{j}_{tf}"))
            
            # save sequence fasta file with tf-specific insertions
            SeqIO.write(tf_motif_seqs, tf_motif_insertion_path, "fasta")
            print(f"Wrote {tf_motif_insertion_path}")
    
        print(f"Done writing motif-inserted peaks for {tis}.")


if __name__ == '__main__':
    main()