import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import random

class GolfDB(Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.files = os.listdir(csv_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        a = self.files[idx]  # annotation info
        fp = os.path.join(self.csv_dir, a)
        df = pd.read_csv(fp, sep=',')

        # the columns of df are as follows: frame_number, X1, Y1, X2, Y2, ..., X17, Y17, label
        # I need them to be like this: FrameNumber, X1, X2, ..., X17, Y1, Y2, ..., Y17, label
        # so that I can reshape the tensor to [seq_length, 2, 17]

        # Reorder columns
        df = df[['frame_number'] + [f'x{i}' for i in range(1, 18)] + [f'y{i}' for i in range(1, 18)] + ['label']]

        images = df.iloc[:, 1:35].values  # shape: [seq_length, 34]
        labels = df.iloc[:, 35].values  # shape: [seq_length]

        # Convert images and labels to torch.tensor
        images = torch.tensor(images, dtype=torch.float32)  # Ensure images are float
        labels = torch.tensor(labels, dtype=torch.long)

        # Reshape images to [seq_length, 2, 17]
        images = images.view(-1, 2, 17)

        sample = {'images': images, 'labels': labels}
        return sample