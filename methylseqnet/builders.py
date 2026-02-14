import gin, pyBigWig, pysam
from Bio import SeqIO
import os
import numpy as np
from methylseqnet.encoding import one_hot_encode_dna
from methylseqnet.readers import load_sequence, load_track, load_masked_track
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import defaultdict
import warnings

################################################################################################################
####                                         Abstract Base Classes                                          ####
################################################################################################################


class SampleGenerator:
    """
    A SampleGenerator class provides an interface to build a dict of lists of samples by data split (e.g.train).
    These samples must be possible to independently grab using a MultitaskIOHandler, i.e. the sample lists must 
    be possible to process in a fully parallelized fashion when building a dataset. 

    The format for a sample is {'source':str,'start':int,'end':int}, where source delineates where to grab
    information from (e.g. a contig in some reference genome or the path to a file from which every entry is
    to be loaded) and the start and end delineating subsequences to load from e.g. the reference contig. The format
    for the returned dict is {'train':[sample1,...,sampleN],...,'test':[sample1,...,sampleM]}.
    """
    def __init__(self, **kwargs):
        pass
    def create_samples(self):
        raise NotImplementedError("Subclass must implement this method.")

class MultitaskIOHandler:
    """
    A MultitaskIOHandler class provides an interface to load data and build input, label, and mask tensors for a 
    specified set of genomic region samples. It must process the input sequence/cpg and the labels into
    appropriate shapes and write them using a DatasetWriter instance. process_batch handles loading
    through to writing to the dataset file, and must be compatible with parallelization.
    """
    def __init__(self, num_tracks=1, label_bin_size=128, **kwargs):
        self.num_tracks = num_tracks
        self.label_bin_size = label_bin_size
        self.io_mappings_list = []
    def process_batch(self,sample_list):
        raise NotImplementedError("Subclass must implement this method.")
    def seq_to_gc_content(
        self,
        sequence,
    ):
        """
        Returns an array of gc content fractions using the binning of the labels
        """
        sequence_array = np.array(list(sequence.lower()))
        gc_mask = (sequence_array == 'g') | (sequence_array == 'c')
        gc_counts = gc_mask.astype(int)
        num_bins = len(sequence) // self.label_bin_size
        trimmed_gc_counts = gc_counts[:num_bins * self.label_bin_size]
        gc_matrix = trimmed_gc_counts.reshape((num_bins, self.label_bin_size))
        gc_fractions = gc_matrix.sum(axis=1) / self.label_bin_size
        
        return gc_fractions  

class MultimethylMultitaskIOHandler:
    """
    A MultimethylMultitaskIOHandler class provides an interface to load data and build sequence, methyl, label,
    and mask tensors for a specified set of genomic region samples. The main difference from a MultitaskIOHandler
    is that MultimethylMultitaskIOHandler saves a single sample per genomic region, with both many methyl tracks
    and many label tracks corresponding to the io_mappings_list. Unlike MultitaskIOHandler, the mask will typically
    not mask out most tasks for a given sample, but can do so if appropriate. Instead, the forward pass of the 
    methylseq model will run once for each methylation track, and the tasks will be concatenated into an output tensor.
    """
    def __init__(self, num_tracks=1, label_bin_size=128, **kwargs):
        self.num_tracks = num_tracks
        self.label_bin_size = label_bin_size
        self.io_mappings_list = []
    def process_batch(self,sample_list):
        raise NotImplementedError("Subclass must implement this method.")
    def seq_to_gc_content(
        self,
        sequence,
    ):
        """
        Returns an array of gc content fractions using the binning of the labels
        """
        sequence_array = np.array(list(sequence.lower()))
        gc_mask = (sequence_array == 'g') | (sequence_array == 'c')
        gc_counts = gc_mask.astype(int)
        num_bins = len(sequence) // self.label_bin_size
        trimmed_gc_counts = gc_counts[:num_bins * self.label_bin_size]
        gc_matrix = trimmed_gc_counts.reshape((num_bins, self.label_bin_size))
        gc_fractions = gc_matrix.sum(axis=1) / self.label_bin_size
        
        return gc_fractions     

class SequenceHandler:
    """
    A SequenceHandler class provides an interface to load string sequences for a genomic region, or
    a batch of them all at once for efficiency. A MultitaskIOHandler instance must reference one or more
    SequenceHandler instances to pull in sequences for the model input. The load_sequence and load_sequence_batch
    functions must be compatible with parallelization. load_sequence_batch is called from outside and uses
    load_sequence as a helper.
    """
    def __init__(self,**kwargs):
        raise NotImplementedError("Subclass must implement this method.")
    def load_sequence(self,source,start,end,**kwargs):
        raise NotImplementedError("Subclass must implement this method")
    def load_sequence_batch(self,sample_list,**kwargs):
        raise NotImplementedError("Subclass must implement this method")
    
class CpGHandler:
    """
    A CpGHandler class provides an interface to load CpG methylation fractions or binary maps for a genomic
    region, or a batch of them all at once for efficiency. A MultitaskIOHandler instance must reference one
    or more CpGHandler instances to pull in methylation for the model input. The load_cpg and load_cpg_batch functions
    must be compatible with parallelization. load_cpg_batch is called from outside and uses load_cpg as a helper.
    """
    def __init__(self,**kwargs):
        raise NotImplementedError("Subclass must implement this method.")
    def load_cpg(self,source,start,end,**kwargs):
        raise NotImplementedError("Subclass must implement this method")
    def load_cpg_batch(self,sample_list,**kwargs):
        raise NotImplementedError("Subclass must implement this method")
    
class LabelHandler:
    """
    A LabelHandler class provides an interface to load binned label data for a genomic region, or a batch of them
    all at once for efficiency. A MultitaskIOHandler instance must reference one or more LabelHandler classes,
    each of which will return a single vector of label data for each region, i.e. a single task. The load_labels and
    load_labels_batch functions must be compatible with parallelization. load_labels_batch is called from outside
    and uses load_labels as a helper.
    """
    def __init__(self,**kwargs):
        raise NotImplementedError("Subclass must implement this method.")
    def load_labels(self,source,start,end,**kwargs):
        raise NotImplementedError("Subclass must implement this method")
    def load_labels_batch(self,sample_list,**kwargs):
        raise NotImplementedError("Subclass must implement this method") 

################################################################################################################
####                                 SampleGenerator implementations                                        ####
################################################################################################################

@gin.register
@gin.configurable
class RegionBedParser(SampleGenerator):
    """
    This subclass handles creating the task dict from a bed file defining regions by split.
    """
    def __init__(
        self,
        regions_bed: str | Path,
        seq_length: str | None = None
    ):
        """
        if seq_length is not provided, the regions bed lengths are used unmodified. If provided, they are padded.
        """
        self.regions_bed = regions_bed
        self.seq_length = seq_length
    def create_samples(self):
        """
        This function exists to parse out the regions_bed file into lists for region
        batch creation.
        
        The region_list_by_split dict will contain a list of region_dict for each
        data split, e.g. test / train / validation, as each of these categories will ultimately be
        written to a separate dataset file.
        """
        region_list_by_split = defaultdict(list)
        with open(self.regions_bed) as f:
            for line in f:
                fields = line.split('\t')
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                # handle the case where the bed file isn't actually the entire sequence length we want
                if self.seq_length is not None and (end - start) < self.seq_length:
                    diff = self.seq_length - (end - start)
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    start -= left_pad
                    end += right_pad
                split = fields[3].strip()
                region_list_by_split[split].append({'source':chrom,'start':start,'end':end,}) 
        return region_list_by_split

@gin.register
@gin.configurable
class ChromosomeChunker(SampleGenerator):
    """
    This subclass handles creating the task dict from a dictionary of chromosome lists by split.
    """
    def __init__(
        self,
        chromosomes_dict: dict,
        fasta_file: str,
        seq_length: int,
        seq_overlap: int = 0
    ):
        """
        at least one chromosome in the chromosome dict, a valid fasta file with the chromosomes all in it, and a chunk 
        seq_length are required. seq_overlap is optional, and will overlap each chromosome subset by the specified number of bp.
        """
        if not isinstance(seq_length,int) or seq_length<1:
            raise ValueError(f"seq_length must be an integer >=1. seq_length={seq_length} is invalid.")
        if not isinstance(seq_overlap,int) or seq_overlap<0:
            raise ValueError(f"seq_overlap must be an integer and >=0. seq_overlap={seq_overlap} is invalid.")
        if seq_overlap > seq_length:
            raise ValueError(f"seq_overlap {seq_overlap} is greater than seq_length {seq_length}.")
        self.chromosomes_dict = chromosomes_dict
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.seq_overlap = seq_overlap
    def create_samples(self):
        """
        This function exists to parse out the chromosomes_dict into lists for region
        batch creation.
        
        The region_list_by_split dict will contain a list of region_dict for each
        data split, i.e. the keys of the chromosomes_dict, as each of these categories will ultimately be
        written to a separate dataset file.
        """
        region_list_by_split = defaultdict(list)
        with pysam.FastaFile(self.fasta_file) as fasta:
            stride = self.seq_length - self.seq_overlap
            for split,chromosomes in self.chromosomes_dict.items():
                if isinstance(chromosomes,list):
                    for chromosome in chromosomes:
                        if chromosome not in fasta.references:
                            raise ValueError(f"Chromosome '{chromosome}' not found in FASTA file {self.fasta_file}.")
                        chrom_length = fasta.get_reference_length(chromosome)
                        for chunk_start in range(0,chrom_length - self.seq_length + 1, stride):
                            region_list_by_split[split].append({
                                'source':chromosome,
                                'start':chunk_start,
                                'end':chunk_start + self.seq_length,
                            })
                else:
                    raise ValueError(f"chromosomes_dict values must be of type list rather than {type(chromosomes)}")
        return region_list_by_split

@gin.register
@gin.configurable
class DirectoryIndexer(SampleGenerator):
    """
    This subclass handles creating the task dict for all files within a directory that match
    a given suffix. All will be put into the same split, which can be provided on initialization.

    The intended use case as of this writing is to list fasta files in a directory or a nested
    set of directories, so one can create a dataset that contains all the sequences therein, annotated
    with the paths to the files. If recursive=True, all folders within the directory will also
    be searched.
    """
    def __init__(
        self,
        directory,
        suffix='fasta',
        subsequence_start=None,
        subsequence_end=None,
        split='pred',
        recursive=False,
        pad_with_Ns=False,
        center_subsequence=False,
    ):
        self.directory=Path(directory)
        self.suffix=suffix
        self.pad_with_Ns = pad_with_Ns
        self.center_subsequence = center_subsequence
        self.start=subsequence_start
        self.end=subsequence_end
        self.split=split
        self.recursive=recursive
        if self.center_subsequence and self.start is not None and self.start!=0:
            warnings.warn(f"When center_subsequence=True, sequence length (subsequence_end - subsequence_start) is centered on each fasta entry. subsequence_end>0 therefore has no function: you set start={self.start} but instead every entry will simply be {self.end-self.start}bp centered on the fasta contig.")
    def create_samples(self):
        """
        This function will generate the samples dict, with all samples listed under the split key provided
        at initialization
        """
        samples_by_split = defaultdict(list)
        
        if self.recursive:
            files = list(self.directory.rglob(f'*.{self.suffix}'))
        else:
            files = list(self.directory.glob(f'*.{self.suffix}'))
        
        for file_path in tqdm(files,desc=f"Indexing *.{self.suffix} files in {self.directory}",leave=False):
            if self.suffix in ["fasta","fa"]:
                if ">" in str(file_path):
                    raise ValueError(f"Disallowed character `:` in file path {file_path}")
                index_path = file_path.with_suffix(".fasta.fai") if file_path.suffix == ".fasta" else file_path.with_suffix(".fa.fai")
                needs_reindex = not index_path.exists() or file_path.stat().st_mtime > index_path.stat().st_mtime
                if needs_reindex:
                    pysam.faidx(str(file_path))
                fasta = pysam.FastaFile(str(file_path))
                for record, length in tqdm(zip(fasta.references,fasta.lengths),"Indexing records in {file_path.name}",leave=False,total=len(fasta.references)):
                    if self.end is not None and self.end > length and not self.pad_with_Ns:
                        warnings.warn(f"Specified end {self.end} is greater than length {length} of record {record} in file {file_path}. Omitting. Set pad_with_Ns to True to include with N padding.")
                    else:
                        if self.center_subsequence and self.start is not None and self.end is not None:
                            seq_length = self.end - self.start
                            center = length // 2
                            half_length = seq_length // 2
                            start_for_contig = center - half_length
                            end_for_contig = center + half_length + (seq_length % 2)
                        else:
                            start_for_contig = self.start
                            end_for_contig = self.end
                        if (
                            (start_for_contig is not None and end_for_contig is not None)
                            and (start_for_contig < 0 or end_for_contig > length)
                            and not self.pad_with_Ns
                            ):
                            warnings.warn(f"Centered subsequence {start_for_contig}-{end_for_contig} goes out of bounds for record {record} in file {file_path}. Omitting. Set pad_with_Ns to True to include with N padding.")
                        else:
                            file_dict = {
                                'source': f"{str(file_path)}>{record}",
                                'start': start_for_contig,
                                'end': end_for_contig,
                            }
                            samples_by_split[self.split].append(file_dict)
        
        return samples_by_split

################################################################################################################
####                                 SequenceHandler implementations                                        ####
################################################################################################################

@gin.register
@gin.configurable
class SingleFastaHandler(SequenceHandler):
    """
    This subclass handles simple fasta sequence loading from a single fasta file where the samples
    are loaded from specified, named sequences, e.g. chromosomes or named entries where the name has meaning.
    """
    def __init__(self,ref_genome: str):
        if os.path.isfile(ref_genome):
            # check that pysam can open the fasta file
            _ = pysam.FastaFile(ref_genome)
            self.ref_genome = ref_genome
            fastafile = pysam.FastaFile(self.ref_genome)
            self.chrom_lengths_dict = {ref: fastafile.lengths[i] for i, ref in enumerate(fastafile.references)}
        else:
            raise OSError(f"{ref_genome} does not exist.")
    def load_sequences(self,source,start,end,fastafile):
        start_pad = 0 - min(start,0)
        chrom_length = self.chrom_lengths_dict[source]
        end_pad = max(end,chrom_length) - chrom_length
        return start_pad*"N" + fastafile.fetch(source,max(start,0),min(end,chrom_length)) + end_pad*"N"
    def load_sequence_batch(self,sample_list):
        fastafile = pysam.FastaFile(self.ref_genome)
        return [self.load_sequences(**sample,fastafile=fastafile) for sample in sample_list]

@gin.register
@gin.configurable
class MultiFastaHandler(SequenceHandler):
    """
    This subclass handles loading sequences from a set of fasta files where the samples specifier
    tell which file to load from and all sequences are loaded up as part of the one entry, with
    no unique identifiers passed back. This makes sense if e.g. each fasta file refers to a specific
    motif insertion or other type of perturbation, and the sequences within are different versions of
    the same thing, randomly generated or otherwise not delineated from one another.
    """
    def __init__(self):
        pass
    
    def load_sequences(self,source,start,end):
        """
        Load specified record from FASTA. 
        This method only supports indexed FASTA files.

        TODO: Implement checks for start and end against contig extent
        """
        file_path = source.split(">")[0]
        contig = source.split(">")[1]
        sequence = load_sequence(file_path,contig,start,end)
        return sequence
        
    
    def load_sequence_batch(self, sample_list):
        """
        Load a batch of samples, where each sample refers to a record in a FASTA file.
        """
        return [self.load_sequences(**sample) for sample in sample_list]

################################################################################################################
####                                    CpGHandler implementations                                          ####
################################################################################################################

@gin.register
@gin.configurable
class MultiBigWigCpGHandler(CpGHandler):
    """
    This subclass handles CpG methylation data from one or more bigwig files, combining the files by the 
    specified operation and binarizing by a threshold if binarize=True
    """
    def __init__(
            self,
            bigwig_files: list,
            combine_operation='mean',
            binarize=False,
            threshold=0.5):
        if not isinstance(bigwig_files,list):
            raise ValueError("bigwig_files input is not a list.")
        for bigwig_file in bigwig_files:   
            if os.path.isfile(bigwig_file):
                try:
                    bw = pyBigWig.open(str(bigwig_file))
                    self.chroms = bw.chroms()
                    bw.close()
                except:
                    raise ValueError(f"{bigwig_file} cannot be opened by pyBigWig.")
            else:
                raise OSError(f"{bigwig_file} does not exist.")
        self.bigwig_files = bigwig_files
        self.combine_operation = combine_operation
        self.binarize = binarize
        self.threshold = threshold
    def load_cpg(self,source,start,end,bws):
        cpg_fractions_list = []
        num_bws = len(bws)
        aggregated_valid_cpgs = np.zeros(end - start)
        for bw in bws:
            start_pad = max(-start,0)
            end_pad = max(end,self.chroms[source]) - self.chroms[source]
            try:
                raw_values = np.array(start_pad*[0] + bw.values(
                    source,
                    max(start,0),
                    min(end,self.chroms[source])
                    ) + end_pad*[0])
            except Exception as e:
                raise RuntimeError(f"Error in bigwig loading for {source}:{start}-{end}. contig length {self.chroms[source]}") from e
            # interpolate -1 values
            raw_values[raw_values < 0] = 1
            valid_mask = ~np.isnan(raw_values)
            normalized_mask = valid_mask.astype(int) / num_bws       
            # Add the normalized mask to the aggregated_valid_cpgs
            aggregated_valid_cpgs += normalized_mask
            # set nan (not a CpG) to zero
            cpg_fractions_list.append(np.nan_to_num(raw_values,nan=0.0))
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(cpg_fractions_list, axis=0)
            aggregated_fractions = np.mean(stacked_values, axis=0)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.binarize:
            return aggregated_fractions>self.threshold,aggregated_valid_cpgs
        else:
            return (aggregated_fractions,aggregated_valid_cpgs)      
   
    def load_cpg_batch(self,sample_list):
        bws = [pyBigWig.open(str(bigwig_file)) for bigwig_file in self.bigwig_files]
        cpgs = [self.load_cpg(**sample,bws=bws) for sample in sample_list]
        for bw in bws:
            bw.close()
        return tuple(map(list,zip(*cpgs))) # this converts the list of many tuples into a tuple of two lists

@gin.register
@gin.configurable
class MultiBedMethylModHandler(CpGHandler):
    """
    This subclass handles methylation data from one or more bedmethyl files, combining the files by the 
    specified operation and binarizing by a threshold if binarize=True
    """
    def __init__(
            self,
            bedmethyl_files: list,
            motif: str='CG,0',
            combine_operation='mean',
            binarize=False,
            threshold=0.5):    
        if not isinstance(bedmethyl_files,list):
            raise ValueError("bedmethyl_files input is not a list.")
        for bedmethyl_file in bedmethyl_files:   
            if not os.path.isfile(bedmethyl_file):
                raise OSError(f"{bedmethyl_file} does not exist.")
        self.bedmethyl_files = bedmethyl_files
        self.motif = motif
        self.combine_operation = combine_operation
        self.binarize = binarize
        self.threshold = threshold    
        
    def load_cpg(self,source,start,end):
        """
        This cpg loader is actually for any mod type specifier by motif
        """
        from dimelo import load_processed
        cpg_fractions_list = []
        aggregated_valid_cpgs = np.zeros(end - start)
        for bedmethyl_file in self.bedmethyl_files:
            start_pad = 0 - min(start,0)
            try:
                modified_base_counts,valid_base_counts = load_processed.pileup_vectors_from_bedmethyl(
                    bedmethyl_file=bedmethyl_file,
                    motif=self.motif,
                    regions=f'{source}:{max(start,0)}-{end}',
                    cores=1,
                    quiet=True,
                )
                nans_everywhere = np.full_like(
                    modified_base_counts, np.nan, dtype=float
                )
                raw_values = np.array(
                    start_pad*[np.nan] + list(
                        np.divide(
                            modified_base_counts,
                            valid_base_counts,
                            out=nans_everywhere,
                            where=valid_base_counts != 0,
                        )
                    )
                )
            except:
                raise RuntimeError(f"Error in CpG bedmethyl loading for {source}:{start}-{end}")
            valid_mask = ~np.isnan(raw_values)
            normalized_mask = valid_mask.astype(int) / len(self.bedmethyl_files)       
            # Add the normalized mask to the aggregated_valid_cpgs
            aggregated_valid_cpgs += normalized_mask
            # set nan (not a CpG) to zero
            cpg_fractions_list.append(np.nan_to_num(raw_values,nan=0.0))
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(cpg_fractions_list, axis=0)
            aggregated_fractions = np.mean(stacked_values, axis=0)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.binarize:
            return aggregated_fractions>self.threshold,aggregated_valid_cpgs
        else:
            return (aggregated_fractions,aggregated_valid_cpgs)      
   
    def load_cpg_batch(self,sample_list):
        """
        This cpg loader is actually for any mod type specifier by motif
        """
        cpgs = [self.load_cpg(**sample) for sample in sample_list]
        return tuple(map(list,zip(*cpgs))) # this converts the list of many tuples into a tuple of two lists

class MultiFileCpGHandler(CpGHandler):
    """
    TODO: this class should likely replace all current CpGHandler implementations: filetype-specific implementation now lives in
    readers.py and outside that the rest of the logic is actually all the same.
    """
    def __init__(
            self,
            cpg_files: list,
            combine_operation='mean',
            binarize=False,
            threshold=0.5,
            extend_cpg_sites=False,
            cpg_values_rescale=1.0,
        ):  
        """
        This subclass handles CpG methylation data from one or more files, combining the files by the
        specified operation and binarizing by a threshold if binarize=True

        Parameters
        ----------
        cpg_files : list
            List of file paths containing CpG methylation data.
        combine_operation : str, optional
            Operation to combine data from multiple files. Default is 'mean'.
        binarize : bool, optional
            Whether to binarize the methylation data. Default is False.
        threshold : float, optional
            Threshold for binarization if binarize is True. Default is 0.5.
        extend_cpg_sites : bool, optional
            Whether to extend CpG sites by one position downstream. Default is False.
        cpg_values_rescale : float, optional
            Factor to rescale CpG values. Default is 1.0.
        """  
        if not isinstance(cpg_files,list):
            raise ValueError("cpg_files input is not a list.")
        for cpg_file in cpg_files:   
            if not os.path.isfile(cpg_file):
                raise OSError(f"{cpg_file} does not exist.")
        self.cpg_files = cpg_files
        self.combine_operation = combine_operation
        self.binarize = binarize
        self.threshold = threshold
        self.extend_cpg_sites = extend_cpg_sites
        self.cpg_values_rescale = cpg_values_rescale

    def load_cpg(self,source,start,end):
        cpg_fractions_list = []
        valid_sites_list = []
        for cpg_file in self.cpg_files:
            raw_values, valid_mask = load_masked_track(
                file_path=cpg_file,
                contig=source,
                start=start,
                end=end,
                negative_to_value=0,
                nan_to_zero=True,
                file_type='bedmethyl' if Path(cpg_file).name.endswith(".bed.gz") else None,
            )
            raw_values *= self.cpg_values_rescale
            if self.extend_cpg_sites:
                indices = np.where(valid_mask > 0)[0]
                indices = indices[indices < len(raw_values) - 1]  # Remove last index if present
                raw_values[indices + 1] = raw_values[indices]
                valid_mask[indices + 1] = 1
            cpg_fractions_list.append(raw_values)
            valid_sites_list.append(valid_mask)
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(cpg_fractions_list, axis=0)
            aggregated_fractions = np.mean(stacked_values, axis=0)
            stacked_valids = np.stack(valid_sites_list, axis=0)
            aggregated_valids = np.mean(stacked_valids, axis=0)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.binarize:
            return aggregated_fractions>self.threshold,aggregated_valids
        else:
            return (aggregated_fractions,aggregated_valids)
    def load_cpg_batch(self,sample_list):
        cpgs = [self.load_cpg(**sample) for sample in sample_list]
        return tuple(map(list,zip(*cpgs)))

@gin.register
@gin.configurable
class SyntheticCpGHandler(CpGHandler):
    """
    This subclass generates synthetic CpG methylation patterns based on provided DNA sequence information.
    """
    def __init__(self,landscape_specification_params=None):
        self.landscape_specification_params = landscape_specification_params
    def load_cpg(self,source,start,end,sequence):
        # find CGs
        # identify which to methylate
        return np.zeros(len(sequence)),np.zeros(len(sequence))
    def load_cpg_batch(self,sample_list,sequence_list):
        cpgs = [self.load_cpg(**sample,sequence=sequence) for sample,sequence in zip(
                    sample_list*(len(sequence_list)//len(sample_list)),
                    sequence_list
                    )
               ]
        return tuple(map(list,zip(*cpgs))) # this converts the list of many tuples into a tuple of two lists
    
################################################################################################################
####                                  LabelHandler implementations                                          ####
################################################################################################################

@gin.register
@gin.configurable
class MultiBigWigLabelHandler(LabelHandler):
    def __init__(
            self,
            bigwig_files: list,
            label_bin_size: int,
            trim_off_ends: int = 0,
            combine_operation='mean',
            normalize_counts_per=None,
            normalize_gc=False,
            binarize=False,
            threshold=5,
            scale=1,
            clip=1024,
            soft_clip=True,
            ):
        if not isinstance(bigwig_files,list):
            raise ValueError("bigwig_files input is not a list.")
        self.bigwig_files = bigwig_files
        self.combine_operation = combine_operation
        self.normalize_gc = normalize_gc
        self.normalize_counts_per = normalize_counts_per
        self.binarize = binarize
        self.threshold = threshold     
        self.trim_off_ends = trim_off_ends
        self.label_bin_size = label_bin_size 
        self.counts_normalization = 0
        self.scale=scale
        self.clip=clip
        self.soft_clip=soft_clip
        if self.normalize_counts_per is not None:
            total_counts = 0
            for bigwig_file in bigwig_files:
                bw = pyBigWig.open(str(bigwig_file))
                # Sum across all chromosomes
                for chrom, length in bw.chroms().items():
                    stats = bw.stats(chrom, 0, length, type="sum", nBins=1,exact=True)
                    if stats[0] is not None:
                        total_counts += stats[0]
                bw.close()
            
            if self.combine_operation == 'mean':
                self.counts_normalization = (total_counts / len(bigwig_files)) / self.normalize_counts_per
            else:
                raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        else:
            self.counts_normalization = 1
 
    def load_labels(self,source,start,end,bws,gc_content):
        values_list = []
        if (end - start)%self.label_bin_size != 0: # - 2*self.trim_off_ends
            raise ValueError(f"Genomic region {source}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size}.") 
        for bw in bws:
            self.chroms = bw.chroms()
            start_pad = max(-(start+self.trim_off_ends),0)
            end_pad = max(end-self.trim_off_ends,self.chroms[source]) - self.chroms[source]
            try:
                raw_values = start_pad*[0] + bw.values(
                    source,
                    max(start+self.trim_off_ends,0),
                    min(end-self.trim_off_ends,self.chroms[source])
                    ) + end_pad*[0]
                # set nan to zero
                values_list.append(np.nan_to_num(raw_values,nan=0.0))
            except Exception as e:
                raise RuntimeError(f"Error in bigwig loading for {source}:{start}-{end}. contig length {self.chroms[source]}") from e
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(values_list, axis=0)
            aggregated_values = np.mean(stacked_values, axis=0).reshape(-1,self.label_bin_size).mean(axis=1)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.normalize_gc:
            # This may in future be replaced with a more sophisticated calculation
            aggregated_values = aggregated_values/(gc_content + 0.1)
        aggregated_values = aggregated_values/self.counts_normalization
        if self.binarize:
            return aggregated_values>self.threshold
        else:
            if self.soft_clip:
                # sqrt over clip threshold
                aggregated_values = np.where(aggregated_values>self.clip,
                                             self.clip + np.sqrt(np.maximum(aggregated_values - self.clip, 0)),
                                             aggregated_values)
                return aggregated_values
            else:
                return np.clip(a=self.scale*aggregated_values, a_min=0, a_max=self.clip)    
            
    def load_labels_batch(self,sample_list,gc_content_list=None):
        bws = [pyBigWig.open(str(bigwig_file)) for bigwig_file in self.bigwig_files]
        if self.normalize_gc:
            if gc_content_list is None:
                raise ValueError("Must provide a gc_content_list if MultiBigWigLabelHandler.normalize_gc = True")
            labels = [self.load_labels(**sample,bws=bws,gc_content=gc_content) for sample,gc_content in zip(sample_list,gc_content_list)]  
        else:
            labels = [self.load_labels(**sample,bws=bws,gc_content=None) for sample in sample_list]
        for bw in bws:
            bw.close()
        # for region,label in zip(regions_list,labels):
        #     print(f'summary for {region}: min=',np.min(label),'max=',np.max(label),'mean',np.mean(label))
        return labels

@gin.register
@gin.configurable
class MultiBedGzLabelHandler(LabelHandler):
    def __init__(
            self,
            bedgz_files: list,
            label_bin_size: int,
            combine_operation='mean',
            normalize_counts_per=None,
            normalize_gc=False,
            binarize=False,
            threshold=5,
            scale=1,
            clip=1024,
            soft_clip=True,
            ):    
        self.bedgz_files = [str(bedgz_file) for bedgz_file in bedgz_files]
        self.label_bin_size = label_bin_size
        self.combine_operation = combine_operation
        self.normalize_counts_per = normalize_counts_per
        self.normalize_gc = normalize_gc
        self.binarize = binarize
        self.threshold = threshold
        self.scale=scale
        self.clip=clip
        self.soft_clip=soft_clip
        if self.normalize_counts_per is not None:
            total_counts = 0
            for bedgz_file in self.bedgz_files:
                tbx = pysam.TabixFile(bedgz_file)
                for chrom in tbx.contigs:
                    for row in tbx.fetch(chrom):
                        tabix_fields = row.split("\t")
                        counts = int(tabix_fields[4])
                        total_counts += counts
            
            if self.combine_operation == 'mean':
                self.counts_normalization = (total_counts / len(self.bedgz_files)) / self.normalize_counts_per
            else:
                raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        else:
            self.counts_normalization = 1
                
    def counts_vector_from_bedgz(self,bedgz_file,chrom,start,end):
        counts_vector = np.zeros(end-start)
        start_pad = 0 - min(start,0)
        for row in pysam.TabixFile(str(bedgz_file)).fetch(chrom,max(start,0),end):
            tabix_fields = row.split("\t")
            genomic_coord = int(tabix_fields[1])
            counts = int(tabix_fields[4])
            counts_vector[genomic_coord-start]+=counts
        return counts_vector
        
    def load_labels(self,source,start,end,bedgz_files,gc_content):
        values_list = []
        if (end - start)%self.label_bin_size != 0: # - 2*self.trim_off_ends
            raise ValueError(f"Genomic region {source}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size}.") # after trimming {self.trim_off_ends} from the ends of the input
        for bedgz_file in self.bedgz_files:
            values_list.append(self.counts_vector_from_bedgz(bedgz_file,source,start,end))
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(values_list, axis=0)
            aggregated_values = np.mean(stacked_values, axis=0).reshape(-1,self.label_bin_size).mean(axis=1)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.normalize_gc:
            # This may in future be replaced with a more sophisticated calculation
            aggregated_values = aggregated_values/(gc_content + 0.1)

        aggregated_values = aggregated_values/self.counts_normalization

        if self.binarize:
            return aggregated_values>self.threshold
        else:
            if self.soft_clip:
                # sqrt over clip threshold
                aggregated_values = np.where(aggregated_values>self.clip,
                                             self.clip + np.sqrt(np.maximum(aggregated_values - self.clip, 0)),
                                             aggregated_values)
                return aggregated_values
            else:
                return np.clip(a=self.scale*aggregated_values, a_min=0, a_max=self.clip)       
    def load_labels_batch(self,sample_list,gc_content_list=None):
        if self.normalize_gc:
            if gc_content_list is None:
                raise ValueError("Must provide a gc_content_list if MultiBigWigLabelHandler.normalize_gc = True")
            labels = [self.load_labels(**sample,bedgz_files=self.bedgz_files,gc_content=gc_content) for sample,gc_content in zip(sample_list,gc_content_list)]  
        else:
            labels = [self.load_labels(**sample,bedgz_files=self.bedgz_files,gc_content=None) for sample in sample_list]

        return labels

@gin.register
@gin.configurable
class MultiBigWigCpGLabelHandler(LabelHandler):
    """
    This LabelHandler uses the MultiBigWigCpGHandler to load appropriate CpG track information, then repurposes it
    to create label vectors for a sequence-to-methylation model. 
    """
    def __init__(
        self,
        bigwig_files: list,
        label_bin_size: int,
        combine_operation='mean',
        binarize=False,
        threshold=0.5,
        no_cpgs_value=0.7,
    ):
        self.label_bin_size = label_bin_size
        self.no_cpgs_value = no_cpgs_value
        self.binarize = binarize
        self.threshold = threshold
        self.cpg_handler = MultiBigWigCpGHandler(
            bigwig_files=bigwig_files,
            combine_operation=combine_operation,
            binarize=binarize,
            threshold=threshold,
        )
    def load_labels(self,source,start,end):
        if (end - start)%self.label_bin_size != 0:
            raise ValueError(f"Genomic region {source}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size}.")
        bws = [pyBigWig.open(str(bigwig_file)) for bigwig_file in self.cpg_handler.bigwig_files]
        aggregated_fractions,aggregated_valid_cpgs = self.cpg_handler.load_cpg(source,start,end,bws=bws)
        for bw in bws:
            bw.close()

        # Reshape into bins and calculate mean for valid CpG sites
        fractions_reshaped = aggregated_fractions.reshape(-1, self.label_bin_size)
        valid_cpgs_reshaped = aggregated_valid_cpgs.reshape(-1, self.label_bin_size)

        valid_rows = valid_cpgs_reshaped.sum(axis=1) > 0  # Identify rows with valid CpG values
    
        bin_means = np.full(valid_cpgs_reshaped.shape[0], self.no_cpgs_value)  # Initialize with default value
        valid_sums = valid_cpgs_reshaped.sum(axis=1)
        bin_means = np.divide(
            fractions_reshaped.sum(axis=1),
            valid_sums,
            out=bin_means,
            where=valid_sums > 0
        )
        
        if self.binarize:
            return bin_means>self.threshold
        else:
            return bin_means

    def load_labels_batch(self,sample_list):
        return [self.load_labels(**sample) for sample in sample_list]
        
@gin.register
@gin.configurable
class MultiBedMethylLabelHandler(LabelHandler):
    """
    This LabelHandler uses the MultiBedMethylModHandler to load appropriate modification track information, then repurposes it
    to create label vectors for a sequence-to-mod-fraction model. 
    """
    def __init__(
        self,
        bedmethyl_files: list,
        label_bin_size: int,
        motif: str = 'A,0',
        combine_operation='mean',
        binarize=False,
        threshold=0.5,
        no_mods_value=0.0,
    ):
        self.label_bin_size = label_bin_size
        self.no_mods_value = no_mods_value
        self.binarize = binarize
        self.threshold = threshold
        self.methyl_handler = MultiBedMethylModHandler(
            bedmethyl_files=bedmethyl_files,
            motif=motif,
            combine_operation=combine_operation,
            binarize=binarize,
            threshold=threshold,
        )
    def load_labels(self,source,start,end):
        if (end - start)%self.label_bin_size != 0:
            raise ValueError(f"Genomic region {source}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size}.")
        aggregated_fractions,aggregated_valid_mods = self.methyl_handler.load_cpg(source,start,end)

        # Reshape into bins and calculate mean for valid CpG sites
        fractions_reshaped = aggregated_fractions.reshape(-1, self.label_bin_size)
        valid_mods_reshaped = aggregated_valid_mods.reshape(-1, self.label_bin_size)
    
        bin_means = np.full(valid_mods_reshaped.shape[0], self.no_mods_value)  # Initialize with default value
        bin_means = np.divide(
            fractions_reshaped.sum(axis=1),
            valid_mods_reshaped.sum(axis=1),
            out=bin_means,
            where=valid_mods_reshaped.sum(axis=1) > 0,
        )
        
        if self.binarize:
            return bin_means>self.threshold
        else:
            return bin_means

    def load_labels_batch(self,sample_list):
        return [self.load_labels(**sample) for sample in sample_list]  

@gin.register
@gin.configurable
class BamCovLabelHandler(LabelHandler):
    """
    This LabelHandler pysam to load read coverage information, then repurposes it
    to create label vectors for a sequence-to-coverage model. 
    """
    def __init__(
            self,
            bam_files: list,
            label_bin_size: int,
            combine_operation='mean',
            normalize_counts_per=None,
            scale=1,
            clip=1024,
            soft_clip=True,
            ):
        self.bam_files = [str(bam_file) for bam_file in bam_files]
        self.label_bin_size = label_bin_size
        self.combine_operation = combine_operation
        self.normalize_counts_per = normalize_counts_per
        self.scale=scale
        self.clip=clip
        self.soft_clip=soft_clip
        total_reads = 0
        for bam_file in self.bam_files:   
            if os.path.isfile(bam_file):
                try:
                    bam = pysam.AlignmentFile(bam_file)
                    self.chroms = dict(zip(bam.references, bam.lengths))
                    total_reads += bam.mapped
                    bam.close()
                except:
                    raise ValueError(f"{bam_file} cannot be opened by pysam AlignmentFile.")
            else:
                raise OSError(f"{bam_file} does not exist.")
        if self.normalize_counts_per is not None and self.bam_files:
            self.read_depth_scaling = self.normalize_counts_per / (total_reads / len(self.bam_files))
        else:
            self.read_depth_scaling = 1
    def load_labels(self,source,start,end):
        if (end - start)%self.label_bin_size != 0:
            raise ValueError(f"Genomic region {source}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size}.")
        values_list = []
        for bam_file in self.bam_files:
            bam = pysam.AlignmentFile(bam_file)
            start_pad = max(-start,0)
            end_pad = max(end,self.chroms[source]) - self.chroms[source]
            try:
                values = np.array(
                    start_pad*[0] + 
                    [
                        sum(x) for x in zip(
                            *(
                                bam.count_coverage(
                                    source,
                                    max(start,0),
                                    min(end,self.chroms[source])
                                )
                            )
                        )
                    ] + end_pad*[0])
                values_list.append(values)
            except Exception as e:
                raise RuntimeError(f"Error in coverage loading for {source}:{start}-{end} from {bam_file}. contig length {self.chroms[source]}") from e
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(values_list, axis=0)
            aggregated_values = np.mean(stacked_values, axis=0).reshape(-1,self.label_bin_size).mean(axis=1)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        
        aggregated_values = self.read_depth_scaling * aggregated_values
        
        if self.soft_clip:
            # sqrt over clip threshold
            aggregated_values = np.where(aggregated_values>self.clip,
                                         self.clip + np.sqrt(np.maximum(aggregated_values - self.clip, 0)),
                                         aggregated_values)
            return aggregated_values
        else:
            return np.clip(a=self.scale*aggregated_values, a_min=0, a_max=self.clip)
    def load_labels_batch(self,sample_list):
        return [self.load_labels(**sample) for sample in sample_list]
    
        

################################################################################################################
####                                MultitaskIOHandler implementations                                      ####
################################################################################################################
        
@gin.register
@gin.configurable
class MethylAtacAtlases(MultitaskIOHandler):
    """
    This IOHandler can map from one set of inputs Seq+Methyl sources to one set of corresponding outputs, e.g. ATAC-seq

    It relies on a match_file tab-separated values table. Currently it is a bit hardcoded to look at .hg38.bigwig methylation
    files and Adult .bw ATAC-seq files, but the intent is to increase its flexibility in the future.

    TODO: consider an option to preserve a fixed task structure rather than rebuilding each time
    """
    def __init__(
            self,
            ref_genome,
            methylation_directory,
            targets_directory,
            match_file,       
            label_bin_size,
            label_num_bins,
            max_chunks_in_mem: int=1000,
            normalize_label_counts: bool=False,
            normalize_label_gc: bool=False,
            binarize_cpg: bool=False,
            binarize_labels: bool=False,
            threshold_cpg: float | None=None,
            threshold_labels: float | None=None,
    ):
        self.max_chunks_in_mem = max_chunks_in_mem
        self.label_num_bins = label_num_bins
        self.label_bin_size = label_bin_size
        self.binarize_labels = binarize_labels

        self.labels_specifier_list = [
            # List of dicts defined as follows:
            #  'index':integer index for label in labels array
            #  'input_handler':InputHandler instance
            #  'label_handler':LabelHandler instance
        ]

        methylation_files = [f for f in os.listdir(methylation_directory) if f.endswith('.hg38.bigwig')]
        atac_files = [f for f in os.listdir(targets_directory) if f.endswith('.bw') and 'Fetal' not in f]
        
        label_index = 0

        self.io_mappings_list = []

        with open(match_file) as f:
            for index,line in tqdm(enumerate(f),desc='Identifying and setting scaling for input files'):
                if index>0: #first line is the headers
                    fields = line.split('\t')
                    methylation_names = fields[0].split(',')
                    atac_names = fields[1].strip('"\'').split(',')
                    try:
                        atac_scale = float(fields[3])
                        atac_clip = float(fields[4])
                    except:
                        atac_scale = 1
                        atac_clip = 1024
                    methylation_celltype_files = []
                    atac_celltype_files = []
                    for methylation_name in methylation_names:
                        if methylation_name!='':
                            methylation_celltype_files+=[(Path(methylation_directory) / f) for f in methylation_files if methylation_name in f]  
                    for atac_name in atac_names:
                        if atac_name!='':
                            atac_name_20 = atac_name.replace(' ','%20')
                            atac_celltype_files += [(Path(targets_directory) / f) for f in atac_files if atac_name_20 in f]
                        
                    if len(methylation_celltype_files)>0 and len(atac_celltype_files)>0:
                        # We have a cell type match; add to the label specifier list
                        self.labels_specifier_list.append(
                            {
                                'index':label_index,
                                'sequence_handler':SingleFastaHandler(
                                    ref_genome=ref_genome,
                                ),
                                'cpg_handler':MultiBigWigCpGHandler(
                                    bigwig_files = methylation_celltype_files,
                                    binarize = binarize_cpg,
                                    threshold = threshold_cpg,
                                ),
                                'label_handler':MultiBigWigLabelHandler(
                                    bigwig_files = atac_celltype_files,
                                    # trim_off_ends=trim_off_ends,
                                    label_bin_size=label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=atac_scale,
                                    clip=atac_clip,
                                )
                            }
                        )     
                        self.io_mappings_list.append({
                            'channel':label_index,
                            'cell_type':label_index,
                            'data_type':'ATAC-seq',
                            'genome':ref_genome,
                            'methylation_files':methylation_names,
                            'label_files':atac_names,
                        })
                        label_index+=1
        self.num_tracks = label_index     
    
    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):
        sample_specifier_list = []
        onehot_dna_list = []
        label_list = []
        mask_list = []
        for label_specifier_dict in self.labels_specifier_list:
            # Load and parse out sequences (returned as dict with sequences per sample)
            sequence_list = label_specifier_dict['sequence_handler'].load_sequence_batch(sample_list)
            gc_content_list = [self.seq_to_gc_content(sequence) for sequence in sequence_list]
            
            # Load and parse out cpgs (returned as dict with meth fraction,valid CG sites per sample)
            methylation_fractions_list,valid_cpgs_list = label_specifier_dict['cpg_handler'].load_cpg_batch(sample_list)

            # Load and parse out labels (returned as dict with labels per sample)
            label_columns_list = label_specifier_dict['label_handler'].load_labels_batch(sample_list,gc_content_list)

            # Process label arrays to account for the masked multitask structure
            label_arrays = [np.zeros((self.label_num_bins,self.num_tracks),dtype=bool if self.binarize_labels else float) for _ in label_columns_list]
            for label_array,label_column in zip(label_arrays,label_columns_list):
                label_array[:,label_specifier_dict['index']]=label_column 
                
            # Generate corresponding masks
            mask_arrays = [np.full((self.label_num_bins,self.num_tracks),False) for _ in label_columns_list]
            for mask_array in mask_arrays:
                mask_array[:,label_specifier_dict['index']] = True
                
            # Add to the relevant building-up lists
            sample_specifier_list+=[f"{sample['source']}:{sample['start']}-{sample['end']}|task{label_specifier_dict['index']}" 
                                         for sample in sample_list 
                                         for _ in range(len(sequence_list)//len(sample_list))]
            label_list+=label_arrays
            mask_list += mask_arrays
            
            onehot_dna_list+=[one_hot_encode_dna(
                dna_strand=sequence,
                cpg_methylation=cpg,
                valid_cpgs=valid_cpgs) for 
                              sequence,cpg,valid_cpgs in zip(
                                  sequence_list,
                                  methylation_fractions_list,
                                  valid_cpgs_list
                              )
                             ]

            
            if len(onehot_dna_list)>=self.max_chunks_in_mem:
                with lock: # we need the lock so allow parallel threads to all write to the same output file
                    dataset_writer.write_chunk(
                        indices_list,
                        sample_specifier_list,
                        onehot_dna_list,
                        label_list,
                        mask_list,
                    )
                sample_specifier_list = []
                onehot_dna_list = []
                label_list = []
                mask_list = []
        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
                label_list,
                mask_list,
            )
    
@gin.register
@gin.configurable
class MultiFastaSequenceOnly(MultitaskIOHandler):
    def __init__(self,num_tracks=1):
        self.num_tracks = num_tracks
        self.multi_fasta_handler = MultiFastaHandler()
        self.io_mappings_list = []
    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):    
        sequence_list = self.multi_fasta_handler.load_sequence_batch(sample_list)

        onehot_dna_list = [one_hot_encode_dna(dna_strand=sequence)[:,0:4] for sequence in sequence_list]

        sample_specifier_list = [
            f"{sample['source']}:{sample['start']}-{sample['end']}" 
            for sample in sample_list 
        ]

        with lock:
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
            )

@gin.register
@gin.configurable
class MethylAtacCageAtlases(MultitaskIOHandler):
    """
    TODO: consider an option to preserve a fixed task structure rather than rebuilding each time
    """
    def __init__(
            self,
            ref_genome,
            methylation_directory,
            atac_directory,
            cage_directory,
            match_file,       
            label_bin_size,
            label_num_bins,
            atac_scaling: float=2,
            cage_scaling: float=128,
            atac_clip: float=32,
            cage_clip: float=384,
            merge_labels: bool=True,
            max_chunks_in_mem: int=100,
            normalize_label_counts: bool=False,
            normalize_label_gc: bool=False,
            binarize_cpg: bool=False,
            binarize_labels: bool=False,
            threshold_cpg: float | None=None,
            threshold_labels: float | None=None,
    ):
        self.merge_labels = merge_labels
        self.max_chunks_in_mem = max_chunks_in_mem
        self.label_num_bins = label_num_bins
        self.label_bin_size = label_bin_size
        self.binarize_labels = binarize_labels

        self.labels_specifier_list = [
            # List of dicts defined as follows:
            #  'index':integer index for label in labels array
            #  'input_handler':InputHandler instance
            #  'label_handler':LabelHandler instance
        ]

        methylation_files = [f for f in os.listdir(methylation_directory) if f.endswith('.hg38.bigwig')]
        atac_files = [f for f in os.listdir(atac_directory) if f.endswith('.bw') and 'Fetal' not in f]
        cage_files = [f for f in os.listdir(cage_directory) if f.endswith('.sorted.bed.gz')]
        
        celltype_index = 0
        label_channel_index = 0

        self.io_mappings_list = []

        with open(match_file) as f:
            for index,line in tqdm(enumerate(f),desc='Identifying and setting scaling for input files'):
                if index>0: #first line is the headers
                    fields = line.split('\t')
                    methylation_names = fields[0].strip('"\'').split(',')
                    atac_names = fields[1].strip('"\'').split(',')
                    try:
                        atac_scale = float(fields[3])
                        atac_clip = float(fields[4])
                    except:
                        atac_scale = 1
                        atac_clip = 1024
                    cage_names = fields[6].strip('"\'').split(',')
                    try:
                        cage_scale = float(fields[8])
                        cage_clip = float(fields[9])
                    except:
                        cage_scale = 1
                        cage_clip = 1024
                    
                    methylation_celltype_files = []
                    atac_celltype_files = []
                    cage_celltype_files = []
                    
                    for methylation_name in methylation_names:
                        if methylation_name!='':
                            methylation_celltype_files+=[(Path(methylation_directory) / f) for f in methylation_files if methylation_name in f]  
                    
                    for atac_name in atac_names:
                        if atac_name!='':
                            atac_name_20 = atac_name.replace(' ','%20')
                            atac_celltype_files += [(Path(atac_directory) / f) for f in atac_files if atac_name_20 in f]

                    for cage_name in cage_names:
                        if cage_name!='':
                            cage_celltype_files += [(Path(cage_directory) / f) for f in cage_files if cage_name in f]
                        
                    if len(methylation_celltype_files)>0 and (len(atac_celltype_files)>0 or len(cage_celltype_files)>0):
                        # We have a cell type match; add to the label specifier list
                        sequence_handler = SingleFastaHandler(
                            ref_genome=ref_genome,
                        )
                        cpg_handler = MultiBigWigCpGHandler(
                            bigwig_files = methylation_celltype_files,
                            binarize = binarize_cpg,
                            threshold = threshold_cpg,
                        )
                        label_channel_indices = []
                        label_handlers = []
                        if self.merge_labels:
                            if len(atac_celltype_files)>0:
                                label_handlers.append(MultiBigWigLabelHandler(
                                    bigwig_files = atac_celltype_files,
                                    label_bin_size=label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=atac_scale,
                                    clip=atac_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'ATAC-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':atac_names,
                                })
                                label_channel_index+=1
                            if len(cage_celltype_files)>0:
                                label_handlers.append(MultiBedGzLabelHandler(
                                    bedgz_files = cage_celltype_files,
                                    label_bin_size = label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=cage_scale,
                                    clip=cage_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'CAGE-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':cage_names,
                                })
                                label_channel_index+=1
                        else:
                            for atac_celltype_file in atac_celltype_files:
                                label_handlers.append(MultiBigWigLabelHandler(
                                    bigwig_files = [atac_celltype_file],
                                    label_bin_size=label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=atac_scale,
                                    clip=atac_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'ATAC-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':Path(atac_celltype_file).stem,
                                })
                                label_channel_index+=1       
                            for cage_celltype_file in cage_celltype_files:
                                label_handlers.append(MultiBedGzLabelHandler(
                                    bedgz_files = [cage_celltype_file],
                                    label_bin_size = label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=cage_scale,
                                    clip=cage_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'ATAC-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':Path(cage_celltype_file).stem,
                                })                                
                                label_channel_index+=1
                                
                        self.labels_specifier_list.append(
                            {
                                'celltype_index':celltype_index,
                                'indices':label_channel_indices,
                                'sequence_handler':sequence_handler,
                                'cpg_handler':cpg_handler,
                                'label_handlers':label_handlers,
                            }
                        ) 
                        celltype_index += 1 
        self.num_tracks = label_channel_index     

    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):
        sample_specifier_list = []
        onehot_dna_list = []
        label_list = []
        mask_list = []
        for label_specifier_dict in self.labels_specifier_list:
            # Load and parse out sequences (returned as dict with sequences per sample)
            sequence_list = label_specifier_dict['sequence_handler'].load_sequence_batch(sample_list)
            gc_content_list = [self.seq_to_gc_content(sequence) for sequence in sequence_list]
            
            # Load and parse out cpgs (returned as dict with meth fraction,valid CG sites per sample)
            methylation_fractions_list,valid_cpgs_list = label_specifier_dict['cpg_handler'].load_cpg_batch(sample_list)

            # the below loop creates a set of arrays for labels/masks and then populates with one or more channels of information
            label_arrays = None
            mask_arrays = None
            for channel,label_handler in zip(label_specifier_dict['indices'],label_specifier_dict['label_handlers']):
                # load up the label columns for the specified channel
                label_columns_list = label_handler.load_labels_batch(
                    sample_list,
                    gc_content_list
                )
                
                # we only initialize the arrays once per label specifier entry, as all label handlers corresponding to that entry share inputs
                if label_arrays is None:
                    label_arrays = [np.zeros((self.label_num_bins,self.num_tracks),dtype=bool if self.binarize_labels else float) for _ in label_columns_list]
                if mask_arrays is None:
                    mask_arrays = [np.full((self.label_num_bins,self.num_tracks),False) for _ in label_columns_list]
                    
                # Process label arrays to account for the masked multitask structure
                for label_array,label_column in zip(label_arrays,label_columns_list):
                    label_array[:,channel]=label_column    
                # Generate corresponding masks
                for mask_array in mask_arrays:
                    mask_array[:,channel] = True
                    
            # Add to the relevant building-up lists
            sample_specifier_list+=[f"{sample['source']}:{sample['start']}-{sample['end']}|tasks{label_specifier_dict['indices']}" 
                                         for sample in sample_list 
                                         for _ in range(len(sequence_list)//len(sample_list))]
            label_list+=label_arrays
            mask_list += mask_arrays
            
            onehot_dna_list+=[one_hot_encode_dna(
                dna_strand=sequence,
                cpg_methylation=cpg,
                valid_cpgs=valid_cpgs) for 
                              sequence,cpg,valid_cpgs in zip(
                                  sequence_list,
                                  methylation_fractions_list,
                                  valid_cpgs_list
                              )
                             ]

            
            if len(onehot_dna_list)>=self.max_chunks_in_mem:
                with lock: # we need the lock so allow parallel threads to all write to the same output file
                    dataset_writer.write_chunk(
                        indices_list,
                        sample_specifier_list,
                        onehot_dna_list,
                        label_list,
                        mask_list,
                    )
                sample_specifier_list = []
                onehot_dna_list = []
                label_list = []
                mask_list = []
        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
                label_list,
                mask_list,
            )

@gin.register
@gin.configurable
class SeqToMethylAtlas(MultitaskIOHandler):
    """
    This IOHandler can map from reference sequence to methylation output in bins, set to match the task structure of a MethylAtacCageAtlases dataset.
    The purpose is largely to test a sequence-to-methylation-to-activity heirarchical model design using the same basic abstractions as the seq+methyl
    to activity model.

    It relies on a match_file tab-separated values table. Currently it is a bit hardcoded to look at .hg38.bigwig methylation
    files and Adult .bw ATAC-seq files, but the intent is to increase its flexibility in the future.
    """
    def __init__(
            self,
            ref_genome,
            methylation_directory,
            atac_directory,
            cage_directory,
            match_file,       
            label_bin_size,
            label_num_bins,
            max_chunks_in_mem: int=100,
            binarize_cpg: bool=False,
            threshold_cpg: float | None=None,
    ):
        
        self.max_chunks_in_mem = max_chunks_in_mem
        self.label_num_bins = label_num_bins
        self.label_bin_size = label_bin_size
        self.binarize_cpg = binarize_cpg

        self.labels_specifier_list = [
            # List of dicts defined as follows:
            #  'index':integer index for label in labels array
            #  'input_handler':InputHandler instance
            #  'label_handler':LabelHandler instance
        ]

        methylation_files = [f for f in os.listdir(methylation_directory) if f.endswith('.hg38.bigwig')]
        atac_files = [f for f in os.listdir(atac_directory) if f.endswith('.bw') and 'Fetal' not in f]
        cage_files = [f for f in os.listdir(cage_directory) if f.endswith('.sorted.bed.gz')]
        
        celltype_index = 0
        label_channel_index = 0

        self.io_mappings_list = []

        with open(match_file) as f:
            for index,line in tqdm(enumerate(f),desc='Identifying and setting scaling for input files'):
                if index>0: #first line is the headers
                    fields = line.split('\t')
                    methylation_names = fields[0].strip('"\'').split(',')
                    atac_names = fields[1].strip('"\'').split(',')
                    cage_names = fields[6].strip('"\'').split(',')
                    
                    methylation_celltype_files = []
                    atac_celltype_files = []
                    cage_celltype_files = []
                    
                    for methylation_name in methylation_names:
                        if methylation_name!='':
                            methylation_celltype_files+=[
                                (Path(methylation_directory) / f) 
                                for f in methylation_files if methylation_name in f
                            ]  
                    
                    for atac_name in atac_names:
                        if atac_name!='':
                            atac_name_20 = atac_name.replace(' ','%20')
                            atac_celltype_files += [
                                (Path(atac_directory) / f) 
                                for f in atac_files if atac_name_20 in f
                            ]

                    for cage_name in cage_names:
                        if cage_name!='':
                            cage_celltype_files += [
                                (Path(cage_directory) / f) 
                                for f in cage_files if cage_name in f
                            ]
                        
                    if len(methylation_celltype_files)>0 and (len(atac_celltype_files)>0 or len(cage_celltype_files)>0):
                        # We have a cell type match; add to the label specifier list
                        sequence_handler = SingleFastaHandler(
                            ref_genome=ref_genome,
                        )
                        cpg_handler = MultiBigWigCpGHandler(
                            bigwig_files = methylation_celltype_files,
                            binarize = binarize_cpg,
                            threshold = threshold_cpg,
                        )
                        label_handler = MultiBigWigCpGLabelHandler(
                            bigwig_files = methylation_celltype_files,
                            label_bin_size = label_bin_size,
                            binarize = binarize_cpg,
                            threshold = threshold_cpg,   
                        )
                        self.io_mappings_list.append(
                            {
                            'channel':label_channel_index,
                            'cell_type':celltype_index,
                            'data_type':'bisulfite',
                            'genome':sequence_handler.ref_genome,
                            'methylation_files':methylation_celltype_files,
                            'label_files':methylation_celltype_files,
                            }
                        )     
                        self.labels_specifier_list.append(
                            {
                                'celltype_index':celltype_index,
                                'indices':[label_channel_index],
                                'sequence_handler':sequence_handler,
                                'cpg_handler':cpg_handler,
                                'label_handlers':[label_handler],
                            }
                        ) 
                        
                        label_channel_index+=1
                        celltype_index += 1 
        self.num_tracks = label_channel_index     

    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):
        sample_specifier_list = []
        onehot_dna_list = []
        label_list = []
        mask_list = []
        for label_specifier_dict in self.labels_specifier_list:
            # Load and parse out sequences (returned as dict with sequences per sample)
            sequence_list = label_specifier_dict['sequence_handler'].load_sequence_batch(sample_list)
            
            # Load and parse out cpgs (returned as dict with meth fraction,valid CG sites per sample)
            methylation_fractions_list,valid_cpgs_list = label_specifier_dict['cpg_handler'].load_cpg_batch(sample_list)

            # the below loop creates a set of arrays for labels/masks and then populates with information
            label_arrays = None
            mask_arrays = None
            for channel,label_handler in zip(label_specifier_dict['indices'],label_specifier_dict['label_handlers']):
                # load up the label columns for the specified channel
                label_columns_list = label_handler.load_labels_batch(
                    sample_list,
                )
                
                # we only initialize the arrays once per label specifier entry, as all label handlers corresponding to that entry share inputs
                if label_arrays is None:
                    label_arrays = [np.zeros((self.label_num_bins,self.num_tracks),dtype=bool if self.binarize_cpg else float) for _ in label_columns_list]
                if mask_arrays is None:
                    mask_arrays = [np.full((self.label_num_bins,self.num_tracks),False) for _ in label_columns_list]
                    
                # Process label arrays to account for the masked multitask structure
                for label_array,label_column in zip(label_arrays,label_columns_list):
                    label_array[:,channel]=label_column    
                # Generate corresponding masks
                for mask_array in mask_arrays:
                    mask_array[:,channel] = True
                    
            # Add to the relevant building-up lists
            sample_specifier_list+=[f"{sample['source']}:{sample['start']}-{sample['end']}|tasks{label_specifier_dict['indices']}" 
                                         for sample in sample_list 
                                         for _ in range(len(sequence_list)//len(sample_list))]
            label_list+=label_arrays
            mask_list += mask_arrays
            
            onehot_dna_list+=[one_hot_encode_dna(
                dna_strand=sequence,
                cpg_methylation=cpg,
                valid_cpgs=valid_cpgs) for 
                              sequence,cpg,valid_cpgs in zip(
                                  sequence_list,
                                  methylation_fractions_list,
                                  valid_cpgs_list
                              )
                             ]

            
            if len(onehot_dna_list)>=self.max_chunks_in_mem:
                with lock: # we need the lock so allow parallel threads to all write to the same output file
                    dataset_writer.write_chunk(
                        indices_list,
                        sample_specifier_list,
                        onehot_dna_list,
                        label_list,
                        mask_list,
                    )
                sample_specifier_list = []
                onehot_dna_list = []
                label_list = []
                mask_list = []
        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
                label_list,
                mask_list,
            )

@gin.register
@gin.configurable
class BedMethylIO(MultitaskIOHandler):
    """
    This IOHandler provides a single input CpG track and a specified number of output label tracks
    based on bedmethyl files with mod counts and read counts provided.

    It assumed you know how many tasks you have, and provides the values from the one bedmethyl label
    handler to all those tasks.
    """
    def __init__(
            self,
            ref_genome,
            methylation_files,
            target_files,
            target_motif,
            label_bin_size,
            label_num_bins,
            num_tracks,
    ):
        self.label_num_bins = label_num_bins
        self.label_bin_size = label_bin_size
        self.num_tracks = num_tracks

        self.sequence_handler = SingleFastaHandler(
            ref_genome=ref_genome,
        )
        self.cpg_handler = MultiBedMethylModHandler(
            bedmethyl_files = methylation_files,
            motif = 'CG,0',
        )
        self.label_handler = MultiBedMethylLabelHandler(
            bedmethyl_files = target_files,
            motif = target_motif,
            label_bin_size = label_bin_size,
        )
        
        self.io_mappings_list = [] 
    
    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):
        sequence_list = self.sequence_handler.load_sequence_batch(sample_list)

        methylation_fractions_list,valid_cpgs_list = self.cpg_handler.load_cpg_batch(sample_list)

        label_columns = self.label_handler.load_labels_batch(sample_list)
        label_list = [np.tile(labels[:,None], (1,self.num_tracks)) for labels in label_columns]
        mask_list = [np.full((self.label_num_bins,self.num_tracks),True) for _ in label_columns]
            
        # Add to the relevant building-up lists
        sample_specifier_list=[
            f"{sample['source']}:{sample['start']}-{sample['end']}|all_tasks" 
            for sample in sample_list 
            for _ in range(len(sequence_list)//len(sample_list))
        ]
        
        onehot_dna_list=[one_hot_encode_dna(
            dna_strand=sequence,
            cpg_methylation=cpg,
            valid_cpgs=valid_cpgs) for 
                          sequence,cpg,valid_cpgs in zip(
                              sequence_list,
                              methylation_fractions_list,
                              valid_cpgs_list
                          )
                         ]
        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
                label_list,
                mask_list,
            )

@gin.register
@gin.configurable
class PhasedFiberRNA(MultitaskIOHandler):
    """
    This IOHandler can map from phased CpG methylation to phased Fiber-seq and RNA-seq

    Phasing is handled along the variants axis of the dataset, which is of shape
    (num_samples, num_variants, num_channels, length) for sequence and label tracks
    and (num_samples, num_variants, num_cell_types, 3, length) for methylation
    """
    def __init__(
            self,
            ref_genome,
            methylation_files_by_phase,
            fiberseq_bigwigs_by_phase,
            rna_files_by_phase,     
            label_bin_size,
            label_num_bins,
            rna_handler_cls = BamCovLabelHandler,
            unphased_rna_file: list[str] = [],
            kwargs_by_data_type: dict = {
                'methylation': {'binarize':False,'threshold':None, 'extend_cpg_sites':False},
                'fiberseq': {'normalize_counts_per':None,'scale':2, 'clip':32},
                'rna': {'normalize_counts_per':1e9,'scale':1, 'clip':384},
            },
            max_chunks_in_mem: int=1000,
            normalize_phased_to_unphased_counts: bool=False,
            phase_normalization_pseudocount: float=0.1,
            normalize_label_counts: bool=False,
    ):
        assert len(methylation_files_by_phase)==len(fiberseq_bigwigs_by_phase)==len(rna_files_by_phase), \
            "The number of phases must be the same for methylation, fiber-seq, and RNA-seq data."
        self.max_chunks_in_mem = max_chunks_in_mem
        self.label_num_bins = label_num_bins
        self.label_bin_size = label_bin_size
        self.normalize_phased_to_unphased_counts = normalize_phased_to_unphased_counts
        self.phase_normalization_pseudocount = phase_normalization_pseudocount

        if self.normalize_phased_to_unphased_counts:
            assert unphased_rna_file!=[], "An unphased RNA BAM file must be provided for normalization."
            phased_rna_kwargs = kwargs_by_data_type['rna'].copy()
            phased_rna_kwargs['normalize_counts_per'] = False
        else:
            phased_rna_kwargs = kwargs_by_data_type['rna']
        self.sequence_handler = SingleFastaHandler(ref_genome=ref_genome)
        self.cpg_handlers = [
            MultiFileCpGHandler(cpg_files = [cpg_file], **kwargs_by_data_type['methylation'])
            for cpg_file in methylation_files_by_phase
            ]
        self.fiber_label_handlers = [
            MultiBigWigLabelHandler(
                bigwig_files = [bigwig_file],
                label_bin_size=label_bin_size,
                normalize_gc=False,
                binarize=False,
                threshold=None,
                **kwargs_by_data_type['fiberseq'],
            )
            for bigwig_file in fiberseq_bigwigs_by_phase
        ]
        self.rna_label_handlers = [
            rna_handler_cls(
                [rna_file],
                label_bin_size = label_bin_size,
                **phased_rna_kwargs,
            )
            for rna_file in rna_files_by_phase
        ]
        self.unphased_rna_loader = rna_handler_cls(
            [unphased_rna_file],
            label_bin_size = label_bin_size,
            **kwargs_by_data_type['rna'],
        )
        self.num_phases = len(methylation_files_by_phase)
        self.num_tracks = 2
        self.io_mappings_list = [
            {
                'channel':0,
                'cell_type':0,
                'data_type':'Fiber-seq',
                'genome':ref_genome,
                'methylation_files':methylation_files_by_phase,
                'label_files':fiberseq_bigwigs_by_phase,
            },
            {
                'channel':1,
                'cell_type':0,
                'data_type':'RNA-seq',
                'genome':ref_genome,
                'methylation_files':methylation_files_by_phase,
                'label_files':rna_files_by_phase,
            },
        ]

    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):
        sequence_list = self.sequence_handler.load_sequence_batch(sample_list)

        sample_specifier_list = [
            f"{sample['source']}:{sample['start']}-{sample['end']}|all_tasks" 
            for sample in sample_list 
            for _ in range(len(sequence_list)//len(sample_list))
        ]

        onehot_dna_phases_list = []
        fiber_phases_list = []   
        rna_phases_list = []   

        for phase in range(self.num_phases):
            methylation_fractions_list,valid_cpgs_list = self.cpg_handlers[phase].load_cpg_batch(sample_list)
            onehot_dna_phases_list.append(
                [
                    one_hot_encode_dna(
                        dna_strand=sequence,
                        cpg_methylation=cpg,
                        valid_cpgs=valid_cpgs) for 
                                    sequence,cpg,valid_cpgs in zip(
                                        sequence_list,
                                        methylation_fractions_list,
                                        valid_cpgs_list
                    )
                ]
            )
            fiber_phases_list.append(self.fiber_label_handlers[phase].load_labels_batch(sample_list))
            rna_phases_list.append(self.rna_label_handlers[phase].load_labels_batch(sample_list))

        onehot_dna_list = [
            np.stack([onehot_dna_phases_list[phase][sample_idx][:,0:4] for phase in range(self.num_phases)])
            for sample_idx in range(len(sample_list))
        ]
        methylation_info_list = [
            np.stack([onehot_dna_phases_list[phase][sample_idx][:,4:] for phase in range(self.num_phases)])
            for sample_idx in range(len(sample_list))
        ]
        fiber_list = [
            np.stack([fiber_phases_list[phase][sample_idx] for phase in range(self.num_phases)])
            for sample_idx in range(len(sample_list))
        ]
        if self.normalize_phased_to_unphased_counts:
            unphased_rna_counts = np.array(self.unphased_rna_loader.load_labels_batch(sample_list))
            # print("unphased cts",unphased_rna_counts.shape,np.sum(unphased_rna_counts,axis=1))
            phased_counts = np.array([
                np.stack([rna_phases_list[phase][sample_idx] for phase in range(self.num_phases)])
                for sample_idx in range(len(sample_list))
            ])
            # print("phased cts",phased_counts.shape,np.sum(phased_counts,axis=2))
            phased_sum = phased_counts.sum(axis=1)
            # print("phased sum",np.mean(phased_sum[phased_sum>0]))
            scaling_factor = unphased_rna_counts / (phased_sum + self.phase_normalization_pseudocount)
            # print("scaling factor",scaling_factor.shape,scaling_factor[scaling_factor>0])
            # print("phase 0 before scaling",phased_counts[:,0][scaling_factor>0])
            # print("phase 1 before scaling",phased_counts[:,1][scaling_factor>0])
            # print("phased sum before scaling",phased_sum[scaling_factor>0])
            # print("unphased before scaling",unphased_rna_counts[scaling_factor>0])
            normalized_phased_counts = phased_counts * scaling_factor[:, np.newaxis, :]
            # print("cts scaling on avg",np.sum(normalized_phased_counts,axis=2)/np.sum(phased_counts,axis=2))
            # print("scaled to",np.sum(unphased_rna_counts,axis=1),"equal now to",np.sum(normalized_phased_counts,axis=1).sum(axis=1))
            # print("nonzero sites unphased|hp1|hp2:",np.sum(unphased_rna_counts>0),np.sum(normalized_phased_counts[:,0]>0),np.sum(normalized_phased_counts[:,1]>0))
            rna_list = [normalized_phased_counts[i] for i in range(len(sample_list))]
        else:
            rna_list = [
                np.stack([rna_phases_list[phase][sample_idx] for phase in range(self.num_phases)])
                for sample_idx in range(len(sample_list))
            ]
        label_list = [
            np.stack([fiber_list[sample_idx], rna_list[sample_idx]], axis=-1)
            for sample_idx in range(len(sample_list))
        ]
        mask_list = [np.ones_like(labels,dtype=bool) for labels in label_list]

        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
                methylation_info_list,
                label_list,
                mask_list,
            )        
            




################################################################################################################
####                           MultimethylMultitaskIOHandler implementations                                ####
################################################################################################################

@gin.register
@gin.configurable
class MultiMethylAtacCageAtlases(MultimethylMultitaskIOHandler):
    """
    TODO: consider an option to preserve a fixed task structure rather than rebuilding each time
    """
    def __init__(
            self,
            ref_genome,
            methylation_directory,
            atac_directory,
            cage_directory,
            match_file,       
            label_bin_size,
            label_num_bins,
            atac_scaling: float=2,
            cage_scaling: float=128,
            atac_clip: float=32,
            cage_clip: float=384,
            merge_labels: bool=True,
            max_chunks_in_mem: int=100,
            normalize_label_gc: bool=False,
            binarize_cpg: bool=False,
            binarize_labels: bool=False,
            threshold_cpg: float | None=None,
            threshold_labels: float | None=None,
    ):
        self.merge_labels = merge_labels
        self.max_chunks_in_mem = max_chunks_in_mem
        self.label_num_bins = label_num_bins
        self.label_bin_size = label_bin_size
        self.binarize_labels = binarize_labels

        self.labels_specifier_list = [
            # List of dicts defined as follows:
            #  'index':integer index for label in labels array
            #  'input_handler':InputHandler instance
            #  'label_handler':LabelHandler instance
        ]

        methylation_files = [f for f in os.listdir(methylation_directory) if f.endswith('.hg38.bigwig')]
        atac_files = [f for f in os.listdir(atac_directory) if f.endswith('.bw') and 'Fetal' not in f]
        cage_files = [f for f in os.listdir(cage_directory) if f.endswith('.sorted.bed.gz')]
        
        celltype_index = 0
        label_channel_index = 0

        self.io_mappings_list = []

        with open(match_file) as f:
            lines = f.readlines()
            for index,line in tqdm(enumerate(lines),desc='Identifying and setting scaling for input files',total=len(lines)):
                if index>0: #first line is the headers
                    fields = line.split('\t')
                    methylation_names = fields[0].strip('"\'').split(',')
                    atac_names = fields[1].strip('"\'').split(',')
                    try:
                        atac_scale = float(fields[3])
                        atac_clip = float(fields[4])
                    except:
                        atac_scale = 1
                        atac_clip = 1024
                    cage_names = fields[6].strip('"\'').split(',')
                    try:
                        cage_scale = float(fields[8])
                        cage_clip = float(fields[9])
                    except:
                        cage_scale = 1
                        cage_clip = 1024
                    
                    methylation_celltype_files = []
                    atac_celltype_files = []
                    cage_celltype_files = []
                    
                    for methylation_name in methylation_names:
                        if methylation_name!='':
                            methylation_celltype_files+=[(Path(methylation_directory) / f) for f in methylation_files if methylation_name in f]  
                    
                    for atac_name in atac_names:
                        if atac_name!='':
                            atac_name_20 = atac_name.replace(' ','%20')
                            atac_celltype_files += [(Path(atac_directory) / f) for f in atac_files if atac_name_20 in f]

                    for cage_name in cage_names:
                        if cage_name!='':
                            cage_celltype_files += [(Path(cage_directory) / f) for f in cage_files if cage_name in f]
                        
                    if len(methylation_celltype_files)>0 and (len(atac_celltype_files)>0 or len(cage_celltype_files)>0):
                        # We have a cell type match; add to the label specifier list
                        sequence_handler = SingleFastaHandler(
                            ref_genome=ref_genome,
                        )
                        cpg_handler = MultiBigWigCpGHandler(
                            bigwig_files = methylation_celltype_files,
                            binarize = binarize_cpg,
                            threshold = threshold_cpg,
                        )
                        label_channel_indices = []
                        label_handlers = []
                        if self.merge_labels:
                            if len(atac_celltype_files)>0:
                                label_handlers.append(MultiBigWigLabelHandler(
                                    bigwig_files = atac_celltype_files,
                                    label_bin_size=label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=atac_scale,
                                    clip=atac_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'ATAC-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':atac_names,
                                })
                                label_channel_index+=1
                            if len(cage_celltype_files)>0:
                                label_handlers.append(MultiBedGzLabelHandler(
                                    bedgz_files = cage_celltype_files,
                                    label_bin_size = label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=cage_scale,
                                    clip=cage_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'CAGE-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':cage_names,
                                })
                                label_channel_index+=1
                        else:
                            for atac_celltype_file in atac_celltype_files:
                                label_handlers.append(MultiBigWigLabelHandler(
                                    bigwig_files = [atac_celltype_file],
                                    label_bin_size=label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=atac_scale,
                                    clip=atac_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'ATAC-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':Path(atac_celltype_file).stem,
                                })
                                label_channel_index+=1       
                            for cage_celltype_file in cage_celltype_files:
                                label_handlers.append(MultiBedGzLabelHandler(
                                    bedgz_files = [cage_celltype_file],
                                    label_bin_size = label_bin_size,
                                    normalize_gc=normalize_label_gc,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                    scale=cage_scale,
                                    clip=cage_clip,
                                    )
                                )
                                label_channel_indices.append(label_channel_index)
                                self.io_mappings_list.append({
                                    'channel':label_channel_index,
                                    'cell_type':celltype_index,
                                    'data_type':'ATAC-seq',
                                    'genome':sequence_handler.ref_genome,
                                    'methylation_files':methylation_names,
                                    'label_files':Path(cage_celltype_file).stem,
                                })                                
                                label_channel_index+=1
                                
                        self.labels_specifier_list.append(
                            {
                                'celltype_index':celltype_index,
                                'indices':label_channel_indices,
                                'sequence_handler':sequence_handler,
                                'cpg_handler':cpg_handler,
                                'label_handlers':label_handlers,
                            }
                        ) 
                        celltype_index += 1 
        self.num_tracks = label_channel_index    
        self.cell_types = celltype_index

    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):
        sample_specifier_list = [f"{sample['source']}:{sample['start']}-{sample['end']}|tasks{list(range(self.num_tracks))}" 
                                         for sample in sample_list]
        onehot_dna_list = [np.zeros((sample['end']-sample['start'],4)) for sample in sample_list]
        methylation_info_list = [np.zeros((sample['end']-sample['start'],3*self.cell_types)) for sample in sample_list]
        label_list = [np.zeros((self.label_num_bins,self.num_tracks),dtype=bool if self.binarize_labels else float) for _ in sample_list]
        mask_list = [np.full((self.label_num_bins,self.num_tracks),False) for _ in sample_list]
        
        for label_idx,label_specifier_dict in enumerate(self.labels_specifier_list):
            # Load and parse out sequences (returned as dict with sequences per sample)
            sequence_list = label_specifier_dict['sequence_handler'].load_sequence_batch(sample_list)
            gc_content_list = [self.seq_to_gc_content(sequence) for sequence in sequence_list]

            # We assume that all samples actually have the same DNA sequence for this IO handler
            if label_idx==0:
                dna_encodings_list = [one_hot_encode_dna(dna_strand=sequence) for sequence in sequence_list]
                for idx,sample_dna_encoding in enumerate(dna_encodings_list):
                    onehot_dna_list[idx] = sample_dna_encoding[:,0:4]
                    
            # Load and parse out cpgs (returned as dict with meth fraction,valid CG sites per sample)
            methylation_fractions_list,valid_cpgs_list = label_specifier_dict['cpg_handler'].load_cpg_batch(sample_list)
            for idx,sample_valid_cpg_encoding in enumerate(valid_cpgs_list):
                methylation_info_list[idx][:,3*label_specifier_dict['celltype_index']+2] = sample_valid_cpg_encoding

            cpg_encodings_list = [
                one_hot_encode_dna(dna_strand=sequence,cpg_methylation=cpg) 
                for sequence,cpg in zip(
                    sequence_list,
                    methylation_fractions_list
                )
            ]
            for idx,sample_cpg_encodings in enumerate(cpg_encodings_list):
                methylation_info_list[idx][:,3*label_specifier_dict['celltype_index']:3*(label_specifier_dict['celltype_index']+1)-1] = sample_cpg_encodings[:,4:6]

            for channel,label_handler in zip(
                label_specifier_dict['indices'],
                label_specifier_dict['label_handlers'],
            ):
                # load up the label columns for the specified channel
                label_columns_list = label_handler.load_labels_batch(
                    sample_list,
                    gc_content_list
                )
                # Process label arrays to account for the masked multitask structure
                for label_array,label_column in zip(label_list,label_columns_list):
                    label_array[:,channel]=label_column    
                # Generate corresponding masks
                for mask_array in mask_list:
                    mask_array[:,channel] = True

        # print([np.sum(onehot,axis=0) for onehot in onehot_dna_list])
        # print([np.sum(info,axis=0) for info in methylation_info_list])
        # print([np.sum(label,axis=00) for label in label_list])
        # print([np.sum(mask,axis=0) for mask in mask_list])
        
        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                indices_list,
                sample_specifier_list,
                onehot_dna_list,
                methylation_info_list,
                label_list,
                mask_list,
            )