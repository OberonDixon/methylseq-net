from methylseqnet.io_handlers import *
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
    def sequential_region_process(self):
        with open(self.regions_bed) as f:
            for line in f:
                fields = line.split('\t')
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                self.multitask_io_handler.process_batch(
                    [(chrom,start,end)],
                    self.dataset_writer,
                )

    