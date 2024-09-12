import argparse
from methylseqnet.io_handlers import *
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Manager
from collections import defaultdict
import gin
import os

@gin.configurable
class PreprocessingPipeline:
    def __init__(
            self,
            genome_region_generator,
            multitask_io_handler,
            dataset_writer_class,
            regions_per_batch,
            output_directory=None,
            ):
        # run sampler
        # create dataset_writer
        # create IO handler

        self.genome_region_generator = genome_region_generator
        self.multitask_io_handler = multitask_io_handler
        self.dataset_writer_class = dataset_writer_class
        self.regions_per_batch = regions_per_batch
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def set_sample_regions(self,bed_file):
        self.regions_bed = bed_file
    def create_sample_regions(self):
        self.regions_bed = self.genome_region_generator.create_regions()
    def create_region_lists(self):
        """
        This function exists to parse out the regions_bed file into lists for region
        batch creation.
        
        The region_list_by_split dict will contain a list of region_dict for each
        data split, e.g. test / train / validation, as each of these categories will ultimately be
        written to a separate dataset file.
        """
        self.region_list_by_split = defaultdict(list)
        with open(self.regions_bed) as f:
            for line in f:
                fields = line.split('\t')
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                split = fields[3].strip()
                self.region_list_by_split[split].append({'chrom':chrom,'start':start,'end':end,})   

    def create_region_batches(self):
        """
        This function itself calls create_region_lists, which assumes that either set_sample_regions or
        create_sample_regions has been used to set a value for self.regions_bed.
        
        The region_batches_by_split dict will contain a list of region batches, each of which is itself
        a tuple of batch_indices,batch_region_dicts. The batches are split out
        by e.g. test / train / validation, as each of these categories will utlimately be written to a 
        separate dataset file.

        The region_index can be used downstream to enable parallel dataset writing while maintaining a
        direct mapping from order-in-bed-file to order-in-dataset even if the order of processing 
        has stochasticity. The region_dict gives the chrom,start,end for the region in a way that can
        be directly passed to loading function in io_handlers.
        """
        self.create_region_lists()
        self.region_batches_by_split = {split:[] for split in self.region_list_by_split.keys()}
        for split,region_list in self.region_list_by_split.items():
            batch_indices = []
            batch_regions = []
            for region_index,region_dict in enumerate(region_list):
                batch_indices.append(region_index)
                batch_regions.append(region_dict)
                if len(batch_regions)>=self.regions_per_batch or region_index>=len(region_list)-1:
                    self.region_batches_by_split[split].append((batch_indices,batch_regions))
                    batch_indices = []
                    batch_regions = []                    
                    
    def initialize_dataset_writer(self,split_name):
        num_tracks = self.multitask_io_handler.num_tracks
        output_path = Path(self.output_directory) / (split_name.strip() + ".h5")
        dataset_writer = self.dataset_writer_class(output_path=output_path,num_tracks=num_tracks)
        return dataset_writer

    def initialize_region_process(
        self,
        subset,
    ):
        self.create_region_batches()
        if subset=='all' and subset not in self.region_batches_by_split.keys():
            self.splits = sorted(self.region_batches_by_split.keys(), 
                key=lambda k: len(self.region_batches_by_split[k]))
        else:
            self.splits = [subset]
        print(f'splits to process:{self.splits}')
        manager = Manager()  # Create a manager
        self.lock = manager.Lock()  # Create a lock via the manager
    
    def sequential_region_process(
        self,
        subset = 'all',
    ):
        self.initialize_region_process(subset)
        for split in self.splits:
            batches = self.region_batches_by_split[split]
            # Each key here is a data split that will want its own dataset
            dataset_writer = self.initialize_dataset_writer(split)
            for indices_list,regions_list in tqdm(batches,
                                                  desc=f'processing and writing {split} to {dataset_writer.output_path}'):
                self.multitask_io_handler.process_batch(
                    indices_list,
                    regions_list,
                    dataset_writer,
                    self.lock,
                )
                
    def parallel_region_process(
        self,
        subset = 'all',
        max_workers=4,
    ):
        self.initialize_region_process(subset)
        for split in self.splits:  
            batches = self.region_batches_by_split[split]
            # Each key here is a data split that will want its own dataset
            dataset_writer = self.initialize_dataset_writer(split)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(
                    self.multitask_io_handler.process_batch, 
                    indices_list, 
                    regions_list, 
                    dataset_writer, 
                    self.lock,
                ) for indices_list,regions_list in batches]
                for future in tqdm(as_completed(futures), total=len(futures), 
                                   desc=f"processing and writing {split} to {dataset_writer.output_path}"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Batch processing failed with exception: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run PreprocessingPipeline")
    parser.add_argument("--config", required=True, help="Path to the gin config file")
    parser.add_argument("--bed_file", required=True, help="Path to the BED file")
    parser.add_argument("--subset", required=True, help="Subset to process (e.g., train, test, validation, or all)")
    parser.add_argument("--mode", required=False, default="sequential", help="Processing mode, sequential or parallel")
    parser.add_argument("--workers", required=False, default="all", help="max_workers across which to parallelize")

    args = parser.parse_args()

    # Initialize gin-config with the provided config file
    gin.parse_config_file(args.config)

    # Create an instance of the PreprocessingPipeline class
    pipeline = PreprocessingPipeline()

    # Set the BED file for the pipeline
    pipeline.set_sample_regions(args.bed_file)
    
    # Run the pipeline with the specified subset
    if args.mode=='sequential':
        pipeline.sequential_region_process(subset=args.subset)
    elif args.mode=='parallel':
        cores_avail = multiprocessing.cpu_count()
        if args.workers=='all':
            cores = cores_avail
        else:
            if int(args.workers)<cores_avail:
                cores = int(args.workers)
            else:
                cores = cores_avail
        print(f"parallelizing across {cores} processes")
        pipeline.parallel_region_process(subset=args.subset,max_workers=cores)
    else:
        raise ValueError(f"Unexpected --mode {args.mode}")

if __name__ == "__main__":
    main()