from methylseqnet.dataset import CustomH5Dataset
from methylseqnet.methylseqnn import MethylSeqNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

batch_size = 2048
num_epochs = 100
input_channels = 5
seq_length = 896
output_channels = 1

hyperparams = {
    "ConvDNA":{
        "in_channels":input_channels,
        "filters":50,
        "kernel_size":17,
        "pool_size":3,   
    },
    "ConvTower":{
        "in_channels":50,
        "filters_init":50,
        "filters_end":100,
        "divisible_by":16,
        "kernel_size":5,
        "pool_size":2,
        "repeat":6,
    },
    "ConvBlock":{
        "in_channels":100,
        "filters":256,
        "kernel_size":1,
    },
    "DenseBlock":{
        "in_features":1024,
        "units":768,
        "dropout":0.2,
    },
    "Final":{
        "in_features":768,
        "units":output_channels,
    },
}


train_dataset = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/train_(0.05,).h5'
validation_dataset = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/validation_0.05.h5'
test_dataset = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/test_0.05.h5'



train_dataset = CustomH5Dataset(train_dataset,batch_size=batch_size)
validation_dataset = CustomH5Dataset(validation_dataset,batch_size=batch_size)
# # test_dataset = CustomH5Dataset(test_dataset)

# Create a DataLoader instance
train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=1)
validation_dataloader = DataLoader(validation_dataset, batch_size=None, shuffle=False, num_workers=1)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# # Iterate through the DataLoader
# for i, (inputs, targets) in tqdm(enumerate(train_dataloader)):
#     continue
# for i, (inputs, targets) in tqdm(enumerate(validation_dataloader)):
#     continue
# for i, (inputs, targets) in tqdm(enumerate(test_dataloader)):
#     continue

# print('Dataset loaded successfully and batches retrieved.')

model = MethylSeqNN(hyperparams)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.98)
criterion = nn.BCEWithLogitsLoss()

patience = 3
best_val_loss = float('inf')
epochs_since_improvement = 0
batchwise_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for phase in ['train','val']:
        if phase == 'train':
            model.train()
            dataloader = train_dataloader
        else:
            model.eval()
            dataloader = validation_dataloader

        running_loss = 0.0
        running_corrects = 0

        for inputs, targets in tqdm(dataloader,unit='batch',desc='model passes'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            batchwise_losses.append(loss.item() * inputs.size(0))

        epoch_loss = running_loss / len(dataloader.dataset)
        if phase=='train':
            training_loss = epoch_loss
            # # Save model with additional metadata
            # model_metadata = {
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'training_loss': training_loss,
            #     'validation_loss': validation_loss,
            #     'num_epochs': num_epochs,
            #     'last_epoch':epoch,
            #     'train_file':f"{datasets_dir}{train_file}",
            #     'architecture_hparam':(input_channels, seq_length, output_channels),
            #     'batchwise_losses':batchwise_losses
            # }
            # model_save_path = f'{models_dir}model_{train_file[0:-3]}_{time_str}_epoch{epoch}.pth'
            # torch.save(model_metadata, model_save_path)
        elif phase=='val':
            validation_loss = epoch_loss
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement+=1
        print(f"{phase} Loss: {epoch_loss:.4f}")
    if epochs_since_improvement>patience:
        print(f"Early stopping after {epoch+1} epochs")
        break
print("Training complete.")
    