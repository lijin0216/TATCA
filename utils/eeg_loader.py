import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal  

try:
    from moabb.datasets import BNCI2014_001
except ImportError:
    from moabb.datasets import BNCI2014001 as BNCI2014_001
from moabb.paradigms import MotorImagery

class BCI42aDataset(Dataset):
    def __init__(self, train=True, subject_id=None):
        # 1. Defined Paradigm: Focus on 4-38Hz
        paradigm = MotorImagery(n_classes=4, events=['left_hand', 'right_hand', 'feet', 'tongue'], 
                                tmin=0, tmax=4.0, fmin=4, fmax=38)
        
        # 2. Loading data
        dataset = BNCI2014_001()
        dataset.subject_list = [subject_id] if subject_id else list(range(1, 10))
        
        print(f"Loading BCI IV-2a (Train={train})...")
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list)
        
        # 3. Divide the training/testing
        if train:
            mask = metadata['session'].str.contains('train') | metadata['session'].str.contains('session_T')
        else:
            mask = metadata['session'].str.contains('test') | metadata['session'].str.contains('session_E')
            
        self.data = X[mask] * 1e6 
        self.labels = y[mask]
        
        # 4. Label coding
        self.class_map = {k: v for v, k in enumerate(np.unique(self.labels))}
        self.numeric_labels = np.array([self.class_map[l] for l in self.labels])
        
        target_length = 128 
        print(f"Resampling from {self.data.shape[-1]} to {target_length}...")
        self.data = signal.resample(self.data, target_length, axis=-1)
        
        self.data = self.data.astype(np.float32)
        print(f"Data Loaded: {self.data.shape} (N, 22, {target_length})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.numeric_labels[idx]
        return x, y

def get_dataloader(batch_size=64, subject_id=1): 
    print(f"Loading data for Subject {subject_id} ONLY...")
    
    train_set = BCI42aDataset(train=True, subject_id=subject_id)
    test_set = BCI42aDataset(train=False, subject_id=subject_id)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_set.data.shape[-1]