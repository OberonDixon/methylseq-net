import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score
import numpy as np
import json
import gin
from datetime import datetime as dt
from pathlib import Path
import argparse
from methylseqnet.dataset import CustomH5Dataset
from methylseqnet.methylseqnn import MethylSeqNN

@gin.configurable
class Trainer:
    def __init__(
        self,  
        in_channels,
        train_dataset_file, 
        validation_dataset_file, 
        batch_size=32, 
        num_epochs=100, 
        learning_rate=0.005, 
        momentum=0.98, 
        pos_weight=100, 
        patience=5, 
        training_epoch_stop=100000000000
    ):
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.patience = patience
        
        self.train_dataset_file = train_dataset_file
        self.validation_dataset_file = validation_dataset_file
        
        self.training_epoch_stop = training_epoch_stop
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_weight = torch.tensor([pos_weight]).to(self.device)
        # if in_channels was provided in the command line, pass it down
        # if not, then it is assumed that in_channels will be pulled from the gin config file
        # if in_channels is not provided in either location, it will default to 5, i.e. seq+cpg
        self.model = MethylSeqNN( 
                                 **({'in_channels':self.in_channels} 
                                  if self.in_channels is not None else {})
                                ).to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        
        self.best_val_loss = float('inf')
        self.epochs_since_improvement = 0
        self.metadata_list = []
        self.batchwise_losses = []
        self.train_dataloader = self.get_dataloader(self.train_dataset_file, shuffle=True)
        self.validation_dataloader = self.get_dataloader(self.validation_dataset_file, shuffle=False)
        self.time_str = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

    def get_dataloader(self, dataset_file, shuffle):
        dataset = CustomH5Dataset(dataset_file, batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=1)
    
    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    dataloader = self.train_dataloader
                else:
                    self.model.eval()
                    dataloader = self.validation_dataloader

                running_loss = 0.0
                targets_list, outputs_list = [], []

                for inputs, targets, mask in tqdm(dataloader, unit='batch', desc=f'{phase} phase for epoch {epoch}'):
                    inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device) if mask is not None else None
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        targets = targets.unsqueeze(-1) if len(targets.shape) == 2 else targets
                        targets = targets.permute(0, 2, 1)

                        if mask is not None:
                            mask = mask.permute(0, 2, 1)
                            outputs = outputs[mask]
                            targets = targets[mask]
                            
                        loss = self.criterion(outputs, targets)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        targets_list.extend(targets.cpu().detach().numpy().tolist())
                        outputs_list.extend(outputs.cpu().detach().numpy().tolist())

                        running_loss += loss.item() * inputs.size(0)
                        self.batchwise_losses.append(loss.item() * inputs.size(0))
                        
                        if len(targets_list) > self.training_epoch_stop:
                            break

                epoch_loss = running_loss / len(dataloader.dataset)
                if phase == 'train':
                    self.training_loss = epoch_loss
                elif phase == 'val':
                    validation_loss = epoch_loss
                    if validation_loss < self.best_val_loss:
                        self.best_val_loss = validation_loss
                        self.epochs_since_improvement = 0
                    else:
                        self.epochs_since_improvement += 1

                print(f"{phase} Loss: {epoch_loss:.4f}")
                self.evaluate_phase(phase, targets_list, outputs_list, epoch_loss, epoch)

                if self.epochs_since_improvement > self.patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        print("Training complete.")
    
    def evaluate_phase(self, phase, targets_list, outputs_list, epoch_loss, epoch):
        probabilities = torch.sigmoid(torch.tensor(outputs_list))
        predictions = (probabilities >= 0.5).int()
        targets = torch.tensor(targets_list).numpy()
        predictions = predictions.numpy()

        accuracy = accuracy_score(targets, predictions)
        precision_overall = precision_score(targets, predictions, zero_division=0)
        recall_overall = recall_score(targets, predictions, zero_division=0)
        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        pr_auc = auc(recall, precision)
        f1 = f1_score(targets, predictions, average='binary')

        print('targets=True:', np.sum(targets))
        print('predictions=True:', np.sum(predictions))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_overall:.4f}")
        print(f"Recall: {recall_overall:.4f}")
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision-Recall AUC: {pr_auc:.4f}')

        performance_dict = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "precision": precision_overall,
            "recall": recall_overall,
            "f1": f1,
            "prc_auc": pr_auc,
        }

        self.metadata_list.append({phase: performance_dict})
        self.save_metadata()

        if phase == 'val' and self.epochs_since_improvement == 0:
            self.save_model(epoch, epoch_loss, self.training_loss)

    def save_metadata(self):
        performance_save_path = Path(self.train_dataset_file).parent / 'models' / f'{self.time_str}_meta.json'

        # Ensure the directory exists
        os.makedirs(os.path.dirname(performance_save_path), exist_ok=True)
        
        with open(performance_save_path, 'w') as f:
            json.dump(self.metadata_list, f, indent=4)
    
    def save_model(self, epoch, validation_loss, training_loss):
        model_metadata = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'num_epochs': self.num_epochs,
            'last_epoch': epoch,
            'gin_file': gin.operative_config_str(),
        }
        model_save_path = Path(self.train_dataset_file).parent / 'models' / f'{self.time_str}_in{self.in_channels}_state.pth'
        torch.save(model_metadata, model_save_path)

def main(config,in_channels):
    gin.parse_config_file(config)
    trainer = Trainer(in_channels=in_channels)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MethylSeqNN model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the gin config file.')
    parser.add_argument('--in_channels', type=int, default=None, help='Number of input channels (5: seq+cpg, 4: seq only, 1: cpg only)')
    args = parser.parse_args()
    main(args.config, args.in_channels)