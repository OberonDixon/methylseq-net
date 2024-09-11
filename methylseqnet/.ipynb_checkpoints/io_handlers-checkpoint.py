import gin, dimelo, pyBigWig, pysam
import os
import numpy as np
from methylseqnet.dna_io import one_hot_encode_dna
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class GenomeRegionGenerator:
    def __init__(self, **kwargs):
        pass
    def create_regions(self):
        raise NotImplementedError("Subclass must implement this method.")

class MultitaskIOHandler:
    """
    A MultitaskIOHandler class provides an interface to load build input, label, and mask tensors for a 
    specified set of genomic region samples. It must process the input sequence/cpg and the labels into
    appropriate shapes and write them using a DatasetWriter instance. process_batch handles loading
    through to writing to the dataset file, and must be compatible with parallelization.
    """
    def __init__(self, **kwargs):
        self.num_tracks = 1
    def process_batch(self,region_list):
        raise NotImplementedError("Subclass must implement this method.")

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
    def load_sequence(self,chrom,start,end,**kwargs):
        raise NotImplementedError("Subclass must implement this method")
    def load_sequence_batch(self,region_list):
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
    def load_cpg(self,chrom,start,end):
        raise NotImplementedError("Subclass must implement this method")
    def load_cpg_batch(self,region_list):
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
    def load_labels(self,chrom,start,end):
        raise NotImplementedError("Subclass must implement this method")
    def load_labels_batch(self,region_list):
        raise NotImplementedError("Subclass must implement this method")       

@gin.register
@gin.configurable
class FastaHandler(SequenceHandler):
    """
    This subclass handles simple fasta sequence loading
    """
    def __init__(self,ref_genome: str):
        if os.path.isfile(ref_genome):
            # check that pysam can open the fasta file
            _ = pysam.FastaFile(ref_genome)
            self.ref_genome = ref_genome
        else:
            raise OSError(f"{ref_genome} does not exist.")
    def load_sequence(self,chrom,start,end,fastafile):
        return fastafile.fetch(chrom,start,end)
    def load_sequence_batch(self,regions_list):
        fastafile = pysam.FastaFile(self.ref_genome)
        return [self.load_sequence(**region,fastafile=fastafile) for region in regions_list]

@gin.register
@gin.configurable
class MultiBigWigCpGHandler(CpGHandler):
    """
    This subclass handles CpG methylation data from one or more bigwig files, combining the files by the specified operation and binarizing by a threshold if binarize=True
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
    def load_cpg(self,chrom,start,end,bws):
        values_list = []
        for bw in bws:
            raw_values = np.array(bw.values(chrom,start,end))
            # interpolate -1 values
            raw_values[raw_values < 0] = 1
            # set nan (not a CpG) to zero
            values_list.append(np.nan_to_num(raw_values,nan=0.0))
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(values_list, axis=0)
            aggregated_values = np.mean(stacked_values, axis=0)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.binarize:
            return aggregated_values>self.threshold
        else:
            return aggregated_values       
   
    def load_cpg_batch(self,regions_list):
        bws = [pyBigWig.open(str(bigwig_file)) for bigwig_file in self.bigwig_files]
        cpgs = [self.load_cpg(**region,bws=bws) for region in regions_list]
        for bw in bws:
            bw.close()
        return cpgs
    
@gin.register
@gin.configurable
class MultiBigWigLabelHandler(LabelHandler):
    def __init__(
            self,
            bigwig_files: list,
            trim_off_ends: int,
            label_bin_size: int,
            combine_operation='mean',
            binarize=False,
            threshold=5,
            ):
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
        self.trim_off_ends = trim_off_ends
        self.label_bin_size = label_bin_size  
    def load_labels(self,chrom,start,end,bws):
        values_list = []
        if (end - start - 2*self.trim_off_ends)%self.label_bin_size != 0:
            raise ValueError(f"Genomic region {chrom}:{start}-{end} cannot be evenly binned into bins of size {self.label_bin_size} after trimming {self.trim_off_ends} from the ends of the input.")
        for bw in bws:
            raw_values = bw.values(chrom,start+self.trim_off_ends,end-self.trim_off_ends)
            # set nan to zero
            values_list.append(np.nan_to_num(raw_values,nan=0.0))
        if self.combine_operation=='mean':
            # Stack the arrays along a new axis (0) and compute the mean along this axis
            stacked_values = np.stack(values_list, axis=0)
            aggregated_values = np.mean(stacked_values, axis=0).reshape(-1,self.label_bin_size).mean(axis=1)
        else:
            raise NotImplementedError(f"No implementation for {self.combine_operation}.")
        if self.binarize:
            return aggregated_values>self.threshold
        else:
            return aggregated_values       
            
    def load_labels_batch(self,regions_list):
        bws = [pyBigWig.open(str(bigwig_file)) for bigwig_file in self.bigwig_files]
        labels = [self.load_labels(**region,bws=bws) for region in regions_list]    
        for bw in bws:
            bw.close()
        return labels
    
@gin.register
@gin.configurable
class BigWigCellAtlas(MultitaskIOHandler):
    def __init__(
            self,
            ref_genome,
            methylation_directory,
            targets_directory,
            match_file,       
            max_chunks_in_mem,
            trim_off_ends,
            label_bin_size,
            label_num_bins,
            binarize_cpg: bool=False,
            binarize_labels: bool=False,
            threshold_cpg: float | None=None,
            threshold_labels: float | None=None,
    ):
        self.max_chunks_in_mem = max_chunks_in_mem
        self.label_num_bins = label_num_bins
        self.binarize_labels=binarize_labels

        self.labels_specifier_list = [
            # List of dicts defined as follows:
            #  'index':integer index for label in labels array
            #  'input_handler':InputHandler instance
            #  'label_handler':LabelHandler instance
        ]

        methylation_files = [f for f in os.listdir(methylation_directory) if f.endswith('.hg38.bigwig')]
        atac_files = [f for f in os.listdir(targets_directory) if f.endswith('.bw') and 'Fetal' not in f]
        
        label_index = 0

        self.io_mappings_dict = {}

        with open(match_file) as f:
            for index,line in enumerate(f):
                if index>0: #first line is the headers
                    fields = line.split('\t')
                    methylation_names = fields[0].split(',')
                    atac_names = fields[1].split(',')
                    # split = fields[2] if len(fields)>2 else "train"
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
                                'sequence_handler':FastaHandler(
                                    ref_genome=ref_genome,
                                ),
                                'cpg_handler':MultiBigWigCpGHandler(
                                    bigwig_files = methylation_celltype_files,
                                    binarize = binarize_cpg,
                                    threshold = threshold_cpg,
                                ),
                                'label_handler':MultiBigWigLabelHandler(
                                    bigwig_files = atac_celltype_files,
                                    trim_off_ends=trim_off_ends,
                                    label_bin_size=label_bin_size,
                                    binarize=binarize_labels,
                                    threshold=threshold_labels,
                                )
                            }
                        )           
                        self.io_mappings_dict[label_index] = (
                            methylation_names,
                            atac_names,
                        )
                        label_index+=1
        self.num_tracks = label_index

    def process_batch(
        self,
        indices_list,
        regions_list,
        dataset_writer,
        lock,
    ):
        onehot_dna_list = []
        label_list = []
        mask_list = []
        for label_specifier_dict in self.labels_specifier_list:#tqdm(self.labels_specifier_list,desc=f'processing batch',leave=False):
            sequence_list = label_specifier_dict['sequence_handler'].load_sequence_batch(regions_list)
            cpg_list = label_specifier_dict['cpg_handler'].load_cpg_batch(regions_list)
            label_columns_list = label_specifier_dict['label_handler'].load_labels_batch(regions_list)
            label_arrays = [np.zeros((self.label_num_bins,self.num_tracks),dtype=bool if self.binarize_labels else float) for _ in label_columns_list]
            for label_array,label_column in zip(label_arrays,label_columns_list):
                label_array[:,label_specifier_dict['index']]=label_column 

            label_list+=label_arrays

            mask_arrays = [np.full((self.label_num_bins,self.num_tracks),False) for _ in label_columns_list]
            for mask_array in mask_arrays:
                mask_array[:,label_specifier_dict['index']] = True

            mask_list += mask_arrays
            
            onehot_dna_list+=[one_hot_encode_dna(sequence,cpg) for sequence,cpg in zip(sequence_list,cpg_list)]

            
            if len(onehot_dna_list)>=self.max_chunks_in_mem:
                with lock: # we need the lock so allow parallel threads to all write to the same output file
                    dataset_writer.write_chunk(
                        regions_list,
                        onehot_dna_list,
                        label_list,
                        mask_list,
                    )
                onehot_dna_list = []
                label_list = []
                mask_list = []
        with lock: # we need the lock so allow parallel threads to all write to the same output file
            dataset_writer.write_chunk(
                regions_list,
                onehot_dna_list,
                label_list,
                mask_list,
            )
        # plt.figure(figsize=(10,5))
        # img=plt.imshow(np.sum(np.array(label_list),axis=0),aspect='auto',interpolation='none')
        # plt.title('sum of all labels in batch')
        # plt.colorbar(img)
        # plt.show()
        # plt.figure(figsize=(10,5))
        # img=plt.imshow(label_list[-1],aspect='auto',interpolation='none')
        # plt.title('last labels array in batch')
        # plt.colorbar(img)
        # plt.show()
        # plt.figure(figsize=(10,5))
        # img=plt.imshow(onehot_dna_list[-1][0:100,:],aspect='auto',interpolation='none')
        # plt.title('first 100bp of sequence encoding, last in batch')
        # plt.colorbar(img)
        # plt.show()
        # plt.figure(figsize=(10,5))
        # img=plt.imshow(mask_list[-1],aspect='auto',interpolation='none')
        # plt.title('mask for last labels in batch')
        # plt.colorbar(img)
        # plt.show()
        # plt.figure(figsize=(10,5))
        # img=plt.imshow(np.sum(np.array(mask_list),axis=0),aspect='auto',interpolation='none')
        # plt.title('sum of all masks in batch')
        # plt.colorbar(img)
        # plt.show()
    


