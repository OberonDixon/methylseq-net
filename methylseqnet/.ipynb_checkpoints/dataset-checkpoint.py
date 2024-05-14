import h5py
from torch.utils.data import Dataset

class CustomH5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f['sequence'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            input_data = f['sequence'][idx]
            target = f['tracks'][idx]

        if self.transform:
            input_data = self.transform(input_data)

        input_data = np.transpose(input_data, (1,0))
        input_data = torch.tensor(input_data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return input_data, target