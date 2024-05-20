from methylseqnet.dataset import CustomH5Dataset
from methylseqnet.methylseqnn import MethylSeqNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score
import numpy as np
import argparse
from datetime import datetime as dt
from pathlib import Path
import json

parser = argparse.ArgumentParser(description="Trains methylseq-net. Provide in_channels and script commit.")

parser.add_argument("-i","--in_channels",help="5: seq+cpg, 4: seq only, 1: cpg only")
parser.add_argument("-c","--commit",help="git commit id string for script version")

args = parser.parse_args()
commit = args.commit
batch_size = 2048
num_epochs = 100
in_channels = int(args.in_channels)
seq_length = 896
output_channels = 1

training_epoch_stop = 100000000000

hyperparams = {
    "ConvDNA":{
        "in_channels":in_channels,
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


train_dataset_file = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/train_th0.05_fd6d75e.h5'
validation_dataset_file = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/validation_th0.05_fd6d75e.h5'
test_dataset_file = '/clusterfs/nilah/oberon/datasets/methylseq-net_deep-ctcf/test_th0.05_fd6d75e.h5'



train_dataset = CustomH5Dataset(train_dataset_file,batch_size=batch_size)
validation_dataset = CustomH5Dataset(validation_dataset_file,batch_size=batch_size)
# # test_dataset = CustomH5Dataset(test_dataset_file)

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
pos_weight = torch.tensor([96 / 4]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

patience = 5
best_val_loss = float('inf')
epochs_since_improvement = 0

metadata_list = [hyperparams]
batchwise_losses = []

time_str = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

for epoch in range(num_epochs):
    print(f"\n\nEpoch {epoch+1}/{num_epochs}")
    
    performance_dict = {}
    
    for phase in ['train','val']:
        if phase == 'train':
            model.train()
            dataloader = train_dataloader
        else:
            model.eval()
            dataloader = validation_dataloader

        running_loss = 0.0
        running_corrects = 0
        
        targets_list = []
        outputs_list = []

        for inputs, targets in tqdm(dataloader,unit='batch',desc='model passes'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                targets_list.extend(targets.cpu().detach().numpy().tolist())
                outputs_list.extend(outputs.cpu().detach().numpy().tolist())

            running_loss += loss.item() * inputs.size(0)
            batchwise_losses.append(loss.item() * inputs.size(0))
            
            if len(targets_list)>training_epoch_stop:
                break

        epoch_loss = running_loss / len(dataloader.dataset)
        if phase=='train':
            training_loss = epoch_loss
        elif phase=='val':
            validation_loss = epoch_loss
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement+=1
        print(f"{phase} Loss: {epoch_loss:.4f}")
        # print('targets:',targets_list,'outputs:',outputs_list)
        probabilities = torch.sigmoid(torch.tensor(outputs_list))
        predictions = (probabilities >= 0.5).int()
        # Convert to numpy arrays for use with sklearn
        targets = torch.tensor(targets_list).numpy()
        predictions = predictions.numpy()

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(targets, predictions)
        precision_overall = precision_score(targets, predictions,zero_division=0)
        recall_overall = recall_score(targets, predictions,zero_division=0)
        # Calculate precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        # Calculate the area under the precision-recall curve
        pr_auc = auc(recall, precision)
        f1 = f1_score(targets, predictions, average='binary')
        
        print('targets=True:',np.sum(targets))
        print('predictions=True:',np.sum(predictions))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_overall:.4f}")
        print(f"Recall: {recall_overall:.4f}")
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision-Recall AUC: {pr_auc:.4f}')
        
        performance_dict[phase] = {
            "epoch":epoch,
            "loss":epoch_loss,
            "accuracy":accuracy,
            "precision":precision_overall,
            "recall":recall_overall,
            "f1":f1,
            "prc_auc":pr_auc,
        }
        
    # Save epoch performance
    metadata_list.append(performance_dict)
    performance_save_path = Path(train_dataset_file).parent / 'models' / f'{commit}_{time_str}_in{in_channels}_meta.json'
    # Save performance data as JSON
    with open(performance_save_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    
    # Save model with additional metadata
    if epochs_since_improvement == 0:
        model_metadata = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'num_epochs': num_epochs,
            'last_epoch':epoch,
            'train_file':train_dataset,
        }
        model_save_path = Path(train_dataset_file).parent / 'models' / f'{commit}_{time_str}_in{in_channels}_state.pth'
        torch.save(model_metadata, model_save_path)
        
    if epochs_since_improvement>patience:
        print(f"Early stopping after {epoch+1} epochs")
        break
print("Training complete.")
    