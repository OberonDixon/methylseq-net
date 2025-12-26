import numpy as np
import pandas as pd
import h5py
import os

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
    return arr.tostring().decode("ascii")


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