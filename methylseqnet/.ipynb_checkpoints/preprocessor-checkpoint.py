from methylseqnet.io_handlers import *
from tqdm.auto import tqdm
import concurrent.futures
from collections import defaultdict
import gin

class PreprocessingPipeline:
    def __init__(
            self,
            genome_region_generator,
            multitask_io_handler,
            dataset_writer_class,
            regions_per_batch,
            config_path='config.gin',
            output_directory=None,
            ):
        gin.parse_config_file(config_path)
        # run sampler
        # create dataset_writer
        # create IO handler

        self.genome_region_generator = genome_region_generator
        self.multitask_io_handler = multitask_io_handler
        self.dataset_writer_class = dataset_writer_class
        self.regions_per_batch = regions_per_batch
        self.output_directory = output_directory

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
                split = fields[3]
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
    def sequential_region_process(self):
        self.create_region_batches()
        for split,batch in self.region_batches_by_split.items():
            # Each key here is a data split that will want its own dataset
            dataset_writer = self.initialize_dataset_writer(split)
            for indices_list,regions_list in tqdm(batch):
                self.multitask_io_handler.process_batch(
                    indices_list,
                    regions_list,
                    dataset_writer,
                )
    # def parallel_region_process(self,max_workers=4):
    #     self.create_region_batches()
        
    #     def process_batch(indices_list,regions_list):
    #         self.multitask_io_handler.process_batch(
    #             indices_list,
    #             regions_list,
    #             self.dataset_writer,
    #         )
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(process_batch, indices_list, regions_list) for indices_list,regions_list in zip(self.region_indices,self.region_batches)]
    #         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Batches"):
    #             try:
    #                 future.result()
    #             except Exception as e:
    #                 print(f"Batch processing failed with exception: {e}")
    