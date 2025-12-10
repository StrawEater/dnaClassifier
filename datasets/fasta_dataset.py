import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import numpy as np

class FastaDataset(Dataset):
    """Dataset for loading sequences from FASTA files organized by phylum"""
    
    def __init__(self, sequences: List[str], labels: List[int], max_length: int = 750):
        """
        Args:
            sequences: List of DNA sequences
            labels: List of integer labels (phylum indices)
            max_length: Maximum sequence length to use
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        return sequence, label      

def add_ambiguity(sequence, pct=0.1):
    """Replace pct% of positions with 'N'."""
    seq = list(sequence)
    n = int(len(seq) * pct)
    positions = random.sample(range(len(seq)), n)

    for p in positions:
        seq[p] = "N"
    return "".join(seq)

def augment_sequences_with_ambiguity(sequences, labels):

    new_sequences = []
    new_labels = []
    
    for idx, sequence in tqdm(enumerate(sequences), total=len(sequences)):
        for i in range(2):
            new_sequences.append(add_ambiguity(sequence))
            new_labels.append(labels[idx].copy())
    
    return (sequences + new_sequences), (labels + new_labels)
    
def separate_train_val_test(sequences, labels, test_size, val_size, random_state = 42):
  
  relative_val_size = val_size/(1-test_size)

  X_temp, X_test, y_temp, y_test = train_test_split(
    sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
  )

  X_train, X_val, y_train, y_val = train_test_split(
      X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
  )

  return X_train, X_val, X_test, y_train, y_val, y_test

def pre_process_batch(batch):
    """Custom collate function for batching sequences"""
    sequences, labels = zip(*batch)
    labels = torch.tensor(np.asarray(labels), dtype=torch.long)
    return list(sequences), labels