import numpy as np
import pandas as pd
import h5py
import os
from optparse import OptionParser
import random
from multiprocessing import Pool

from Bio import motifs
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.motifs.matrix import PositionWeightMatrix
import numpy as np
import matplotlib.pyplot as plt
import pysam
from tqdm.auto import tqdm

def shuffle_peak(seq, peak_start, peak_end):
    return dinuc_shuffle(seq[peak_start:peak_end], rng=np.random.default_rng())

def insert_random_pos(seq, pwm, peak_len, n, shuffle=False):
    """
    From a PWM, sample and insert a motif at a random sequence position in a peak n times.
    Returns a list of sequences with the motif randomly inserted.
    Don't shuffle peak portion by default.
    
    seq: string
    pwm: dataframe with nucleotide probabilities
    n: number of times to sample and insert
    """
    random_inserts = []
    for j in range(n):
        
        sampled_motif = sample_motif(pwm)
        motif_len = len(sampled_motif)

        # shuffle the peak portion
        if shuffle:
            peak_start = (len(seq)-peak_len)//2
            peak_end = peak_start + peak_len
            shuffled_peak = shuffle_peak(seq, peak_start, peak_end)
            seq = seq[:peak_start] + shuffled_peak + seq[peak_end:]

        # index to start the replacement (random position within the peak)
        left = np.random.randint(0, high=peak_len - motif_len + 1) # +1 for full range
        
        replacement_start = peak_start + left
        # index to end the segment that is being replaced
        replacement_end = replacement_start + motif_len

        # left part + motif + right part (starting AFTER replacement_end)
        motif_seq = seq[:replacement_start] + sampled_motif + seq[replacement_end:]

        random_inserts.append(motif_seq)

    return random_inserts

def insert_center_pos(seq, pwm, peak_len, n, shuffle=True):
    """
    From a PWM, sample and insert a motif at the center sequence position n times.
    Returns a list of sequences with the motif randomly inserted.
    Shuffle peak portion by default.
    """
    shuffled_center_inserts = []
    for j in range(n):

        if shuffle:
            # shuffle only the peak portion
            peak_start = (len(seq)-peak_len)//2
            peak_end = peak_start + peak_len
            shuffled_peak = shuffle_peak(seq, peak_start, peak_end)
            seq = seq[:peak_start] + shuffled_peak + seq[peak_end:]

        sampled_motif = sample_motif(pwm)
        motif_len = len(sampled_motif)
        
        # calculate center position for insertion/replacement
        center_start = (len(seq) - motif_len) // 2
        center_end = center_start + motif_len
        
        # left part + motif + right part (starting AFTER center_end)
        motif_seq = seq[:center_start] + sampled_motif + seq[center_end:]
        
        shuffled_center_inserts.append(motif_seq)
    return shuffled_center_inserts

def sample_motif(pwm):
    sampled = pwm.apply(lambda x: np.random.choice(["A", "C", "G","T"], p=x, size=1, replace=False), axis=0)
    sampled = ''.join(sampled.values[0])
    return sampled

def change_over_control(old_pred, new_pred):
    return np.log(new_pred/old_pred)

'''
def create_subsets(input_file_path, output_file_path_base, tissue_tf_pairs):
    """
    Separate entries of given HDF5 file into independently written HDF5 files for tissue-TF pairs.
    
    input_file_path: input HDF5 file
    output_file_path_base: directory to create output HDF5 files (in tissue folders)
    tissue_tf_pairs: list of (tissue, tf) tuples
    """
    with h5py.File(test_h5path, 'r') as hf:
        #sequences = hf['sequence'].asstr()[:].astype(str)
        sequences = hf['sequence'][:]
        specifiers = hf['specifier'].asstr()[:].astype(str)
        # need both in order to use np operations, asstr() alone doesn't make a np.dtype.str
        #tracks = hf['tracks'].asstr()[:].astype(str)
        tracks = hf['tracks'][:]
        
        #print(specifiers)
        #print(type(specifiers))
        #print(specifiers.dtype)
        #print(specifiers.shape)
        hf.close()
    
    if not os.path.exists(output_file_path_base):
        os.mkdir(output_file_path_base)   
    
    for pair in tissue_tf_pairs:

        tissue, tf = pair[0], pair[1]

        # Substring to search for (string type)
        substring = f'{tissue}/{tf}.fasta'
        #print(type(substring))
    
        # Find indices where the substring is present
        idxs = np.where(np.char.find(dataset, substring) > -1)[0]
    
        # Print the indices
        #print("idxs are ", idxs)
        
        output_file_path = output_file_path_base + f'{tissue}/{tissue}_{tf}.h5'

        if not os.path.exists(output_file_path_base + f'{tissue}'):
            os.mkdir(output_file_path_base + f'{tissue}')
        
        with h5py.File(output_file_path, 'w') as output_file_1:

            sequence_shape = (len(idxs),) + sequences.shape[1:]
            sequence_subset = output_file_1.create_dataset('sequence', sequence_shape, dtype=sequences.dtype)
            sequence_subset[:] = sequences[idxs]
            
            specifier_shape = (len(idxs),) + specifiers.shape[1:]
            specifier_subset = output_file_1.create_dataset('specifier', specifier_shape, dtype=h5py.string_dtype())
            specifier_subset[:] = specifiers[idxs]

            track_shape = (len(idxs),) + tracks.shape[1:]
            track_subset = output_file_1.create_dataset('track', track_shape, dtype=tracks.dtype)
            track_subset[:] = tracks[idxs]

            output_file_1.close()
'''

################
# I/O functions
################

def parse_h5_file(file_path):
    # Usage:
    # preds = parse_h5_file('my_model_outputs.h5')
    with h5py.File(file_path, 'r') as hf:
        # See groups/datasets in root
        print(f"Keys in the file: {list(hf.keys())}")
        
        # Access specific dataset (e.g., 'predictions' or 'labels')
        if 'predictions' in hf:
            data = hf['predictions'][:]
            print(f"Data shape: {data.shape}")
            return data
        else:
            print("Target key not found.")
            return None

def h5_to_dataframe(file_path, dataset_name='specifier'):
    with h5py.File(file_path, 'r') as hf:
        data = hf[dataset_name][:]
        df = pd.DataFrame(data)
        df.columns = [dataset_name]
        return df

def print_h5_structure(name, obj):
    """Function formats to pass to hf.visititems()"""
    print(f"{name} -> {type(obj)}")


def split_h5_by_string(input_path, tfs, output_dir, batch_size=1000):
    # Usage:
    # split_h5_by_string('input_data.h5', my_tfs, target_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Define HDF5 string type as variable length UTF-8 - needed to handle specifiers (?)
    str_type = h5py.string_dtype(encoding='utf-8')
    
    with h5py.File(input_path, 'r') as source:
        preds_ds = source['predictions']
        specs_ds = source['specifier']
        total_rows = preds_ds.shape[0]
        
        # Open output files
        outputs = {}
        for tf in tfs:
            file_path = os.path.join(output_dir, f"predictions_{tf}.h5")
            outputs[tf] = h5py.File(file_path, 'w') # using 'w' will overwrite existing files when run
        
        try:
            for i in range(0, total_rows, batch_size):
                end = min(i + batch_size, total_rows)
                
                # Load chunk, decode specifiers
                batch_preds = preds_ds[i:end]
                batch_specs = [s.decode() if isinstance(s, bytes) else s for s in specs_ds[i:end]]
                
                for tf in tfs:
                    mask = np.array([tf in s for s in batch_specs])
                    filtered_preds = batch_preds[mask]
                    filtered_specs = [batch_specs[j] for j, val in enumerate(mask) if val]
                    
                    if len(filtered_preds) > 0:
                        target_hf = outputs[tf]
                        
                        if 'predictions' not in target_hf:
                            target_hf.create_dataset('predictions', data=filtered_preds, 
                                                     maxshape=(None, *batch_preds.shape[1:]), 
                                                     chunks=True, compression="gzip")
                            
                            target_hf.create_dataset('specifier', data=filtered_specs, 
                                                     maxshape=(None,), dtype=str_type, 
                                                     chunks=True, compression="gzip")
                        else:
                            p_ds = target_hf['predictions']
                            p_ds.resize(p_ds.shape[0] + filtered_preds.shape[0], axis=0)
                            p_ds[-filtered_preds.shape[0]:] = filtered_preds
                            
                            s_ds = target_hf['specifier']
                            s_ds.resize(s_ds.shape[0] + len(filtered_specs), axis=0)
                            s_ds[-len(filtered_specs):] = filtered_specs
                            
                print(f"Processed {end}/{total_rows} rows...", end='\r')

        finally:
            # Close files
            for f in outputs.values():
                f.close()
            print(f"\n Finished! Files located at: {os.path.abspath(output_dir)}")


# Source of all below code: https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

def random_shuffle(seq):
    """Shuffle input sequence."""
    rand_index = np.random.permutation(len(seq))
    return seq[rand_index,:]

def dinuc_shuffle(seq, num_shufs=None, rng=None, seed=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        if seed:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]

##############################################################################
# Helper functions
##############################################################################


def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tobytes().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

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

def write_motif_insertion(args):
    tis, tf, seqs, pwm, PEAKS_OUTPUT_DIR, INPUT_LEN, SHUFFLE_LEN, N, OVERWRITE = args
    tf_motif_insertion_path = f'{PEAKS_OUTPUT_DIR}/{tis}/motif_inserted_sequences_{INPUT_LEN}_{tf}.fasta'
    # IMPORTANT: don't re-write motif insertion file if it's already created for that tf, unless OVERWRITE is set
    if os.path.exists(tf_motif_insertion_path) and not OVERWRITE:
        print(f"Skipped {tf_motif_insertion_path}")
        return
    tf_motif_seqs = []
    for i, seq in enumerate(seqs):
        trials = insert_center_pos(seq, pwm, SHUFFLE_LEN, N, shuffle=True)
        for j, trial in enumerate(trials):
            tf_motif_seqs.append(SeqRecord(Seq(trial), id=f"{i}_{j}_{tf}"))
    SeqIO.write(tf_motif_seqs, tf_motif_insertion_path, "fasta")
    print(f"Wrote {tf_motif_insertion_path}")

def main():
    usage = 'usage: %prog [options] <PWMS_TOP_DIR> <PEAKS_TOP_DIR> <PEAKS_OUTPUT_DIR>'
    parser = OptionParser(usage)

    parser.add_option('--input-len', dest='input_len',
        type='int',
        default=524288, # default to Borzoi context
        help='Sequence input length [Default: %default]')
    parser.add_option('--shuffle-len', dest='shuffle_len',
        type='int',
        default=128, # default bin size
        help='Shuffling length around peak center [Default: %default]')
    parser.add_option('--n', dest='n',
        default=5, # default to 5 trials per motif/peak pair
        help='Number of trials for each motif insertion [Default: %default]')
    parser.add_option('--reference-genome', dest='reference_genome',
        default='', # default hg38 path
        help='Reference genome fasta path [Default: %default]')
    parser.add_option('--tfs-file', dest='tfs_file',
        default='transcription_factors.txt', # default path to text file with TF names (one per line)
        help='Path to text file with TF names (one per line) [Default: %default]')
    parser.add_option('--overwrite', action='store_true', dest='overwrite',
        default=False,
        help='Overwrite existing files if present [Default: %default]')
    (options, args) = parser.parse_args()

    if options.reference_genome == '':
        try:
            from methylseqnet_repro import genomes
            options.reference_genome = genomes / 'hg38.fa'
        except Exception as e:
            print(f"Failed to import genomes path from methylseqnet_repro.paths: {e}. Using hardcoded UC Berkeley HPC hg38 path instead.")
            options.reference_genome = "/clusterfs/nilah/oberon/genomes/hg38.fa"

    if len(args) == 3:
        PWMS_TOP_DIR = args[0]
        PEAKS_TOP_DIR = args[1]
        PEAKS_OUTPUT_DIR = args[2]
    else:
        parser.error('Must provide parameters PWMS_TOP_DIR, PEAKS_TOP_DIR, and PEAKS_OUTPUT_DIR')

    if not os.path.isdir(PEAKS_OUTPUT_DIR):
        os.makedirs(PEAKS_OUTPUT_DIR,exist_ok=True)

    # calculate pad length needed to fill input
    INPUT_LEN = options.input_len
    if INPUT_LEN % 2 != 0:
        raise ValueError("INPUT_LEN must be an even number.")
    SHUFFLE_LEN = options.shuffle_len
    N = options.n
    print(f"Model input sequence length is: {INPUT_LEN}")
    print(f"Shuffled central sequence length is: {SHUFFLE_LEN}")

    # reference genome
    hg38_fasta = pysam.Fastafile(options.reference_genome)
    chrom_lens_dict = dict(zip(hg38_fasta.references, hg38_fasta.lengths))
    print(chrom_lens_dict)

    # PWMs for selection, or otherwise for each human CIS-BP TF (~700)
    # PWMS_TOP_DIR is required so we always have a fallback
    TFS = []
    if os.path.exists(options.tfs_file):
        with open(options.TFS_FILE, 'r') as file:
            TFS = file.read().splitlines()
    else:
        TFS = os.listdir(f"{PWMS_TOP_DIR}/pwms/")
        TFS = [t.split(".csv")[0] for t in TFS]
    print(f"TFs are: {TFS}")
    print(f"{len(TFS)} TFs total")

    # process pwms to be ready for sampling
    pwms = {}
    valid_tfs = []
    for tf in TFS:
        try:
            pwms[tf] = pd.read_csv(f'{PWMS_TOP_DIR}/pwms/{tf}.csv', index_col=0, skiprows=1, header=None)
            pwms[tf] = pwms[tf]/pwms[tf].sum(axis=0)  # columns sum to 1
            valid_tfs.append(tf)
        except FileNotFoundError:
            print(f"PWM file not found for {tf}, removing...")

    TFS = valid_tfs
    
    # tissue list (from peak files)
    TISSUES = os.listdir(f'{PEAKS_TOP_DIR}/')
    TISSUES = [tis.split('.hg38.bed')[0] for tis in TISSUES if 'peaks' in tis]
    print(f"Tissues are: {TISSUES}")
    # TISSUES = ['Hepatocyte_peaks', 'Adipocyte_peaks'] # dummy for testing, want to run on full list eventually
    # TISSUES = ['Hepatocyte_peaks'] # dummy for testing, want to run on full list eventually

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
                # The fetched sequence is INPUT_LEN long
                center_loc = (row["Start"] + row["End"]) // 2
                start, end = center_loc - INPUT_LEN//2, center_loc + INPUT_LEN//2
                
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
            os.makedirs(f'{PEAKS_OUTPUT_DIR}/{tis}',exist_ok=True)
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
                # The fetched sequence must be INPUT_LEN long
                center_loc = (row["Start"] + row["End"]) // 2
                start, end = center_loc - INPUT_LEN//2, center_loc + INPUT_LEN//2
                
                start_pad = 0 - min(start,0)
                chrom_length = chrom_lens_dict[source]
                end_pad = max(end,chrom_length) - chrom_length
                
                # pad peak with Ns if not enough context
                seq = start_pad*"N" + hg38_fasta.fetch(source,
                                                       max(start, 0),
                                                       min(end, chrom_length)) + end_pad*"N"
                # shuffle peak portion
                peak_start = (len(seq)-SHUFFLE_LEN)//2
                peak_end = peak_start + SHUFFLE_LEN
                shuffled_peak = shuffle_peak(seq, peak_start, peak_end)
                seq = seq[:peak_start] + shuffled_peak + seq[peak_end:]
            except ValueError:
                continue
            endogenous_shuffled_peaks.append(SeqRecord(Seq(seq), id=f'{i}'))
    
        # save sequence fasta file
        if not os.path.exists(f'{PEAKS_OUTPUT_DIR}/{tis}'):
            os.makedirs(f'{PEAKS_OUTPUT_DIR}/{tis}',exist_ok=True)
        SeqIO.write(endogenous_shuffled_peaks, f'{PEAKS_OUTPUT_DIR}/{tis}/endogenous_shuffled_peak_sequences_{INPUT_LEN}.fasta', "fasta")
    
    print("Done writing endogenous sequences with shuffled center peaks.")

    ### Create fasta files with motif sequences inserted into tissue-specific peaks ###
    for tis in TISSUES:
        print(f"Writing motif-inserted peaks for {tis}...")
    
        # load endogenous peak sequences
        records = list(SeqIO.parse(f"{PEAKS_OUTPUT_DIR}/{tis}/endogenous_sequences_{INPUT_LEN}.fasta", "fasta"))
        seqs = [str(i.seq) for i in records]

        task_args = [(tis, tf, seqs, pwms[tf], PEAKS_OUTPUT_DIR, INPUT_LEN, SHUFFLE_LEN, N, options.overwrite) for tf in TFS]
        with Pool() as pool:
            pool.map(write_motif_insertion, task_args)
    
        print(f"Done writing motif-inserted peaks for {tis}.")


if __name__ == '__main__':
    main()