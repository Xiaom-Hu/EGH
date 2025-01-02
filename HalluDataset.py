import torch
from torch.utils.data import Dataset, DataLoader

class HallDataset(Dataset):
    def __init__(self, embedding, gradient, label):
        self.embedding = embedding
        self.gradient = gradient
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.embedding[index], self.gradient[index], self.label[index]


def hallu_collate_fn(batch):
    embedding = []
    gradient = []
    label = []
    for sample in batch:
        embedding.append(sample[0])
        gradient.append(sample[1])
        label.append(sample[2])
    return embedding, gradient, torch.tensor(label)
