from methylseqnet.dataset import CustomH5Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

train_dataset = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/train_chr1.h5'

dataset = CustomH5Dataset(train_dataset)

# Create a DataLoader instance
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for i, (inputs, targets) in tqdm(enumerate(dataloader)):
    continue
    # print(f'Batch {i+1}:')
    # print(f'Inputs: {inputs.shape}')
    # print(f'Targets: {targets.shape}')
    # if i > 200:  # Stop after first 200 batches to avoid too much output
    #     break

print('Dataset loaded successfully and batches retrieved.')