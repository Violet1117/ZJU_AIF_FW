import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class FER2013(Dataset):
    def __init__(self, data_fame, transform=None):
        self.data_frame = data_fame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.data_frame.pixels.iloc[idx]
        img = np.array(img.split()).astype('float').reshape(48, 48)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.data_frame.emotion.iloc[idx]
        label = torch.tensor(label)
        return img, label
    
if __name__ == '__main__':
    df_data = pd.read_csv('dataset/fer2013.csv')
    data_train = df_data[df_data['Usage'] == 'Training']
    data_val = df_data[df_data['Usage'] == 'PrivateTest']
    data_test = df_data[df_data['Usage'] == 'PublicTest']
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_data = FER2013(data_train, transform)
    val_data = FER2013(data_val, transform)
    test_data = FER2013(data_test, transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    for img, label in train_loader:
        print(img.shape)
        print(label)
        break