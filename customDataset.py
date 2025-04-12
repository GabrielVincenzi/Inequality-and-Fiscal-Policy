import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

class ChunkIterableDataset(IterableDataset):
    def __init__(self, csv_path, state_columns, reward_columns, chunk_size=100000, shuffle=False):
        self.csv_path = csv_path
        self.state_columns = state_columns
        self.reward_columns = reward_columns
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.total_rows = sum(1 for _ in open(csv_path)) - 1  # Total rows minus header
        self.header = pd.read_csv(csv_path, nrows=0).columns.tolist()

    def process_chunk(self, chunk):
        """Converts a chunk to NumPy arrays for fast indexing."""
        states = chunk[self.state_columns].to_numpy(dtype=np.float32)
        rewards = chunk[self.reward_columns].to_numpy(dtype=np.float32)
        return states, rewards

    def chunk_generator(self):
        """Generator that yields data from the CSV file in chunks."""
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            if self.shuffle:
                chunk = chunk.sample(frac=1).reset_index(drop=True)  # Shuffle within the chunk
            states, rewards = self.process_chunk(chunk)
            for i in range(len(chunk)):  # Yield each sample in the chunk
                yield torch.tensor(states[i]), torch.tensor(rewards[i])

    def __iter__(self):
        return self.chunk_generator()
    

class TrainingDataset(Dataset):
    def __init__(self, data, state_columns, reward_columns):
        self.states = torch.tensor(data[state_columns].values, dtype=torch.float32)
        self.rewards = torch.tensor(data[reward_columns].values, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.rewards[idx]
