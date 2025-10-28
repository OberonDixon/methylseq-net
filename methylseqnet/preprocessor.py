import argparse
import sys
import traceback
from methylseqnet.io_handlers import *
from methylseqnet.datawriter import *
from methylseqnet.methylseqnn import *
from methylseqnet.dataset import *
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
import multiprocessing
from multiprocessing import Manager
from collections import defaultdict
import gin
import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

@gin.configurable
class PreprocessingPipeline:
    def __init__(
            self,
            sample_generator,
            multitask_io_handler,
            dataset_writer_class,
            samples_per_batch=1,
            output_directory=None,
            ):
        # run sampler
        # create dataset_writer
        # create IO handler

        self.sample_generator = sample_generator
        self.multitask_io_handler = multitask_io_handler
        self.dataset_writer_class = dataset_writer_class
        self.samples_per_batch = samples_per_batch
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def create_sample_batches(self):
        """
        This function itself calls create_sample_lists, which assumes that either set_sample_regions or
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
        sample_list_by_split = self.sample_generator.create_samples()
        self.sample_batches_by_split = {split:[] for split in sample_list_by_split.keys()}
        for split,sample_list in sample_list_by_split.items():
            batch_indices = []
            batch_samples = []
            for sample_index,sample_dict in enumerate(sample_list):
                batch_indices.append(sample_index)
                batch_samples.append(sample_dict)
                if len(batch_samples)>=self.samples_per_batch or sample_index>=len(sample_list)-1:
                    self.sample_batches_by_split[split].append((batch_indices,batch_samples))
                    batch_indices = []
                    batch_samples = []  
                    
    def initialize_dataset_writer(self,split_name):
        num_tracks = self.multitask_io_handler.num_tracks
        output_path = Path(self.output_directory) / (split_name.strip() + ".h5")
        dataset_writer = self.dataset_writer_class(
            output_path=output_path,
            num_tracks=num_tracks,
            io_mappings_list=self.multitask_io_handler.io_mappings_list,
        )
        return dataset_writer

    def process_samples(
        self,
        subset = 'all',
        mode = 'sequential',
        max_workers = 1,
    ):
        self.create_sample_batches()
        if subset=='all' and subset not in self.sample_batches_by_split.keys():
            self.splits = sorted(self.sample_batches_by_split.keys(), 
                key=lambda k: len(self.sample_batches_by_split[k]))
        else:
            self.splits = [subset]
        manager = Manager()  # Create a manager
        self.lock = manager.Lock()  # Create a lock via the manager
        print(f'splits to process:{self.splits}')
        for split in self.splits:
            batches = self.sample_batches_by_split[split]
            # Each key here is a data split that will want its own dataset. Most recent one is stored as class attribute
            self.dataset_writer = self.initialize_dataset_writer(split)  
            if mode=='sequential':
                for indices_list,sample_list in tqdm(batches,
                                                      desc=f'processing and writing {split} to {self.dataset_writer.output_path}'):
                    self.multitask_io_handler.process_batch(
                        indices_list,
                        sample_list,
                        self.dataset_writer,
                        self.lock,
                    )          
            elif mode=='parallel':
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(
                        self.multitask_io_handler.process_batch, 
                        indices_list, 
                        sample_list, 
                        self.dataset_writer, 
                        self.lock,
                    ) for indices_list,sample_list in batches]
                    for future in tqdm(as_completed(futures), total=len(futures), 
                                       desc=f"processing and writing {split} to {self.dataset_writer.output_path}"):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Parallel processing failed with exception: {e}")
                            traceback.print_exc()    
            else:
                raise NotImplementedError(f"No running mode implemented for {mode}")

def pretrained_model_embeddings():
    parser = argparse.ArgumentParser(description="Save model embeddings for one or more datasets.")
    parser.add_argument("--config", required=True, help="Path to the the gin config file for a MethylSeqNN trainer containing a pretrained model.")
    parser.add_argument("--embeddings-shape", nargs="+", type=int, help="Space separated integers for shape of embeddings.")
    parser.add_argument("--input-datasets-directory", required=True, help="Path to the input directory from which to run datasets.")
    parser.add_argument("--subset", required=False, nargs="*", help="Dataset subset(s) to run. Defaults to all.")
    parser.add_argument("--output-datasets-directory", required=True, help="Path to an output directory for saving the embeddings datasets.")
    parser.add_argument("--batch-size", required=False, type=int, default=8, help="Size of batches for passing through model.")
    parser.add_argument("--write-batch-size", required=False, type=int, default=16, help="Size of batches for writing to disk.")
    parser.add_argument("--append-to-existing", action="store_true", help="Append to existing dataset if this is a restarted job.")
    
    args = parser.parse_args(sys.argv[2:])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if str(args.input_datasets_directory)==str(args.output_datasets_directory):
        raise ValueError(f"Cannot set same input and out directories - this would overwrite files and break the pipeline.")

    if not os.path.exists(args.output_datasets_directory):
        os.makedirs(args.output_datasets_directory)

    gin.parse_config_file(args.config)

    embeddings_shape = tuple(args.embeddings_shape)

    model = MethylSeqNN()

    model.seq_output_head = None
    model.mode = "pretrained-only"
    model.eval()
    model.to(device)

    if args.subset:
        dataset_files = [Path(args.input_datasets_directory) / (filename + ".h5") for filename in args.subset]
    else:
        dataset_files = [f for f in Path(args.input_datasets_directory).iterdir() if f.suffix.lower() in ['.h5', '.hdf5']]

    print(dataset_files)
    
    for dataset_file in dataset_files:
        output_file = Path(args.output_datasets_directory) / Path(dataset_file).name
        seq_embeddings_writer = SeqEmbeddingsWriter(
            embeddings_shape = embeddings_shape,
            output_path = output_file,
            append = args.append_to_existing,
        )
        try:
            dataset = MultiMethylDataset(
                dataset_file,
                batch_size=args.batch_size,
                return_specifiers=True,
            )
        except:
            dataset = CustomH5Dataset(
                dataset_file,
                batch_size=args.batch_size,
                return_specifiers=True,
            )
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

        def writer_worker(write_queue):
            """Worker function that processes items from the queue."""
            while True:
                item = write_queue.get()
                if item is None:  # Sentinel value to signal termination.
                    write_queue.task_done()
                    break
                indices_list, sample_specifiers_list, embeddings_list = item
                print(f'writing regions: {sample_specifiers_list}')
                seq_embeddings_writer.write_chunk(indices_list, sample_specifiers_list, embeddings_list)
                write_queue.task_done()
        
        # Create a bounded queue with a maximum number of pending tasks.
        write_queue = Queue(maxsize=10)
        
        # Start the writer thread.
        writer_thread = Thread(target=writer_worker, args=(write_queue,))
        writer_thread.start()
        
        with torch.no_grad():
            indices_list = []
            embeddings_list = []
            sample_specifiers_list = []
            
            for batch_idx, (inputs, _, _, specifiers) in enumerate(tqdm(dataloader)):
                inputs = inputs.to(device)
                embeddings = model(inputs)
                
                # Convert embeddings to numpy arrays if needed.
                embeddings_list += [emb.cpu().numpy() for emb in embeddings]
                
                batch_start_idx = batch_idx * args.batch_size
                batch_end_idx = batch_start_idx + len(specifiers.tolist())
                indices_list += list(range(batch_start_idx, batch_end_idx))
                sample_specifiers_list += specifiers.tolist()
                
                # When we accumulate enough samples, push the work to the queue.
                if len(embeddings_list) > args.write_batch_size:
                    # This put() will block if the queue already has maxsize items.
                    write_queue.put((indices_list, sample_specifiers_list, embeddings_list))
                    # Reset lists for the next chunk.
                    indices_list = []
                    sample_specifiers_list = []
                    embeddings_list = []
        
            # If there is any remaining data after the loop, write it as well.
            if embeddings_list:
                write_queue.put((indices_list, sample_specifiers_list, embeddings_list))
        
        # Wait until all items in the queue have been processed.
        write_queue.join()
        
        # Signal the writer thread to exit by putting a sentinel value.
        write_queue.put(None)
        writer_thread.join()

def main():
    parser = argparse.ArgumentParser(description="Run PreprocessingPipeline")
    parser.add_argument("--config", required=True, help="Path to the gin config file. No default.")
    parser.add_argument("--subset", required=False, default='all', help="Subset to process (e.g., train, test, validation, or all). Defaults to all.")
    parser.add_argument("--mode", required=False, default="sequential", help="Processing mode, sequential or parallel. Defaults to sequential.")
    parser.add_argument("--workers", required=False, default="all", help="max_workers across which to parallelize. Defaults to all.")

    args = parser.parse_args(sys.argv[1:])

    # Initialize gin-config with the provided config file
    gin.parse_config_file(args.config)

    # Create an instance of the PreprocessingPipeline class
    pipeline = PreprocessingPipeline()
    
    # Run the pipeline with the specified subset
    cores_avail = multiprocessing.cpu_count()
    if args.mode=='parallel':
        if args.workers=='all':
            cores = cores_avail
        else:
            try:
                workers = int(args.workers)
                cores = min(workers, cores_avail)  # Ensure cores don't exceed available cores
            except ValueError:
                print(f"Invalid value for --workers: {args.workers}, defaulting to available cores")
                cores = cores_avail
    else:
        cores=1
    print(f"running {args.mode}. {cores_avail} available cores, using {cores}")
    pipeline.process_samples(subset=args.subset,mode=args.mode,max_workers=cores)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pretrained_model_embeddings":
        pretrained_model_embeddings()
    else:
        main()