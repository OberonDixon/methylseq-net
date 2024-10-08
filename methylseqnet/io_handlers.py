import gin, pyBigWig, pysam
from Bio import SeqIO
# import dimelo
import os
import numpy as np
from methylseqnet.dna_io import one_hot_encode_dna
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import defaultdict

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
    A MultitaskIOHandler class provides an interface to load build input, label, and mask tensors for a 
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
    def __init__(self,regions_bed: str | Path):
        self.regions_bed = regions_bed
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
                split = fields[3].strip()
                region_list_by_split[split].append({'source':chrom,'start':start,'end':end,}) 
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
    ):
        self.directory=Path(directory)
        self.suffix=suffix
        self.start=subsequence_start
        self.end=subsequence_end
        self.split=split
        self.recursive=recursive
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
        
        for file_path in files:
            file_dict = {
                'source': str(file_path),
                'start': self.start,
                'end': self.end,
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
        else:
            raise OSError(f"{ref_genome} does not exist.")
    def load_sequences(self,source,start,end,fastafile):
        return fastafile.fetch(source,start,end)
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
        Load all sequences from the specified FASTA file in the order they appear. 
        This method supports non-indexed FASTA files as well as indexed ones.
        """
        if os.path.isfile(source):  
            if start and end:
                sequences = [str(record.seq)[start:end] for record in SeqIO.parse(source, "fasta")]
            else:
                sequences = [str(record.seq) for record in SeqIO.parse(source, "fasta")]
        else:
            raise OSError(f"{source} does not exist.")
        return sequences
        
    
    def load_sequence_batch(self, sample_list):
        """
        Load a batch of samples, where each sample refers to a FASTA file.
        Each file's sequences are loaded in full and returned as a list.
        """
        return [sequence for sample in sample_list for sequence in self.load_sequences(**sample)]

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
            raw_values = np.array(bw.values(source,start,end))
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
            # trim_off_ends: int,
            label_bin_size: int,
            combine_operation='mean',
            normalize_counts=False,
            normalize_gc=True,
            binarize=False,
            threshold=5,
            scale=1,
            clip=1024,
            ):
        if not isinstance(bigwig_files,list):
            raise ValueError("bigwig_files input is not a list.")
        self.bigwig_files = bigwig_files
        self.combine_operation = combine_operation
        self.normalize_gc = normalize_gc
        self.normalize_counts = normalize_counts
        self.binarize = binarize
        self.threshold = threshold     
        # self.trim_off_ends = trim_off_ends
        self.label_bin_size = label_bin_size 
        self.counts_normalization = 0
        self.scale=scale
        self.clip=clip
        for bigwig_file in bigwig_files:   
            if os.path.isfile(bigwig_file):
                try:
                    bw = pyBigWig.open(str(bigwig_file))
                    if self.normalize_counts:
                        sample_values = bw.values('chr1',0,200000000)
                        sample_values_sum = np.sum(np.nan_to_num(sample_values,nan=0))
                        if combine_operation=='mean':
                            self.counts_normalization+=sample_values_sum/(200000000+len(bigwig_files))
                        else:
                            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
                    bw.close()
                except:
                    raise ValueError(f"{bigwig_file} cannot be opened by pyBigWig.")
            else:
                raise OSError(f"{bigwig_file} does not exist.")
 
    def load_labels(self,source,start,end,bws,gc_content):
        values_list = []
        if (end - start)%self.label_bin_size != 0: # - 2*self.trim_off_ends
            raise ValueError(f"Genomic region {source}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size}.") 
        for bw in bws:
            raw_values = bw.values(source,start,end)
            # raw_values = bw.values(source,start+self.trim_off_ends,end-self.trim_off_ends)
            # set nan to zero
            values_list.append(np.nan_to_num(raw_values,nan=0.0))
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(values_list, axis=0)
            aggregated_values = np.mean(stacked_values, axis=0).reshape(-1,self.label_bin_size).mean(axis=1)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.normalize_gc:
            # This may in future be replaced with a more sophisticated calculation
            aggregated_values = aggregated_values/(gc_content + 0.1)
        if self.normalize_counts:
            aggregated_values = 1000*aggregated_values/self.counts_normalization
        if self.binarize:
            return aggregated_values>self.threshold
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
            normalize_counts=False,
            normalize_gc=True,
            binarize=False,
            threshold=5,
            scale=1,
            clip=1024,
            ):    
        self.bedgz_files = [str(bedgz_file) for bedgz_file in bedgz_files]
        self.label_bin_size = label_bin_size
        self.combine_operation = combine_operation
        self.normalize_counts = normalize_counts
        self.normalize_gc = normalize_gc
        self.binarize = binarize
        self.threshold = threshold
        self.scale=scale
        self.clip=clip
        for bedgz_file in self.bedgz_files:   
            if os.path.isfile(bedgz_file):
                try:
                    if self.normalize_counts:
                        sample_values = self.counts_vector_from_bedgz(bedgz_file,'chr1',0,200000000)
                        sample_values_sum = np.sum(np.nan_to_num(sample_values,nan=0))
                        if self.combine_operation=='mean':
                            self.counts_normalization+=sample_values_sum/(200000000+len(self.bedgz_files))
                        else:
                            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
                    else:
                        pysam.TabixFile(bedgz_file) # just check that we can open the file
                except Exception as e:
                    raise ValueError(f"{e}. {bedgz_file} cannot be opened by pysam tabix - is it bgzipped and indexed?")
            else:
                raise OSError(f"{bedgz_file} does not exist.") 
                
    def counts_vector_from_bedgz(self,bedgz_file,chrom,start,end):
        counts_vector = np.zeros(end-start)
        for row in pysam.TabixFile(str(bedgz_file)).fetch(chrom,start,end):
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
        if self.normalize_counts:
            aggregated_values = 1000*aggregated_values/self.counts_normalization
        if self.binarize:
            return aggregated_values>self.threshold
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
                                    normalize_counts=normalize_label_counts,
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
        self.synthetic_cpg_handler = SyntheticCpGHandler()
        self.io_mappings_list = []
    def process_batch(
        self,
        indices_list,
        sample_list,
        dataset_writer,
        lock,
    ):    
        sequence_list = self.multi_fasta_handler.load_sequence_batch(sample_list)
        # the synthetic methylation class currently just returns zeros; we need to implement some pattern generation
        methylation_fractions_list,valid_cpgs_list = self.synthetic_cpg_handler.load_cpg_batch(sample_list,sequence_list)

        onehot_dna_list = [one_hot_encode_dna(
                dna_strand=sequence,
                cpg_methylation=cpg,
                valid_cpgs=valid_cpgs) for 
                              sequence,cpg,valid_cpgs in zip(
                                  sequence_list,
                                  methylation_fractions_list,
                                  valid_cpgs_list,
                              )
                             ]

        sample_specifier_list = [f"{sample['source']}:{sample['start']}-{sample['end']}|{sequence_idx}" 
                                 for sample in sample_list 
                                 for sequence_idx in range(len(sequence_list)//len(sample_list))
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
                                    normalize_counts=normalize_label_counts,
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
                                    normalize_counts=normalize_label_counts,
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
                                    normalize_counts=normalize_label_counts,
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
                                    normalize_counts=normalize_label_counts,
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