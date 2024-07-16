from methylseqnet.io_handlers import *
from tqdm.auto import tqdm
import concurrent.futures

class PreprocessingPipeline:
    def __init__(
            self,
            genome_region_generator,
            multitask_io_handler,
            dataset_writer,
            regions_per_batch,
            ):
        # run sampler
        # create dataset_writer
        # create IO handler

        self.genome_region_generator = genome_region_generator
        self.multitask_io_handler = multitask_io_handler
        self.dataset_writer = dataset_writer
        self.regions_per_batch = regions_per_batch

    def set_sample_regions(self,bed_file):
        self.regions_bed = bed_file
    def create_sample_regions(self):
        self.regions_bed = self.genome_region_generator.create_regions()
    def create_region_batches(self):
        self.region_batches = []
        self.region_indices = []
        indices_list = []
        regions_list = []
        with open(self.regions_bed) as f:
            for line_index,line in enumerate(f):
                fields = line.split('\t')
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                indices_list.append(line_index)
                regions_list.append({'chrom':chrom,'start':start,'end':end,})     
                if len(regions_list)>=self.regions_per_batch:
                    self.region_indices.append(indices_list)
                    self.region_batches.append(regions_list)
                    indices_list = []
                    regions_list = []
    def sequential_region_process(self):
        self.create_region_batches()
        for indices_list,regions_list in tqdm(zip(self.region_indices,self.region_batches),total=len(self.region_batches)):
            self.multitask_io_handler.process_batch(
                indices_list,
                regions_list,
                self.dataset_writer,
            )
    def parallel_region_process(self,max_workers=4):
        self.create_region_batches()
        
        def process_batch(indices_list,regions_list):
            self.multitask_io_handler.process_batch(
                indices_list,
                regions_list,
                self.dataset_writer,
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, indices_list, regions_list) for indices_list,regions_list in zip(self.region_indices,self.region_batches)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Batches"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Batch processing failed with exception: {e}")
    