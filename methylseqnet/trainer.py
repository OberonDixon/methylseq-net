from torch.utils.data import DataLoader
import torch

from methylseqnet.methylseqnn import *
from methylseqnet.dataset import *

print('cuda is available ',torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device for training',device)

# train_dataset = CustomH5Dataset(f"{datasets_dir}{train_file}")
# val_dataset = CustomH5Dataset(f"{datasets_dir}{validation_file}")
# test_dataset = CustomH5Dataset(f"{datasets_dir}{test_file}")

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = BassetCpG(input_channels, seq_length, output_channels)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.98)
# criterion = nn.L1Loss()

# time_str = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
# training_loss = float('inf')
# validation_loss = float('inf')

# patience = 3
# best_val_loss = float('inf')
# epochs_since_improvement = 0
# batchwise_losses = []

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#             dataloader = train_loader
#         else:
#             model.eval()
#             dataloader = val_loader

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             batchwise_losses.append(loss.item() * inputs.size(0))

#         epoch_loss = running_loss / len(dataloader.dataset)
#         if phase=='train':
#             training_loss = epoch_loss
#             # Save model with additional metadata
#             model_metadata = {
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'training_loss': training_loss,
#                 'validation_loss': validation_loss,
#                 'num_epochs': num_epochs,
#                 'last_epoch':epoch,
#                 'train_file':f"{datasets_dir}{train_file}",
#                 'architecture_hparam':(input_channels, seq_length, output_channels),
#                 'batchwise_losses':batchwise_losses
#             }
#             model_save_path = f'{models_dir}model_{train_file[0:-3]}_{time_str}_lin-relu-l1_epoch{epoch}.pth'
#             torch.save(model_metadata, model_save_path)
#         elif phase=='val':
#             validation_loss = epoch_loss
#             if validation_loss < best_val_loss:
#                 best_val_loss = validation_loss
#                 epochs_since_improvement = 0
#             else:
#                 epochs_since_improvement+=1
#         print(f"{phase} Loss: {epoch_loss:.4f}")
#     if epochs_since_improvement>patience:
#         print(f"Early stopping after {epoch+1} epochs")
#         break
# print("Training complete.")

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
# model_save_path = f'{models_dir}model_{train_file[0:-3]}_{time_str}_lin-relu-l1_final.pth'
# torch.save(model_metadata, model_save_path)