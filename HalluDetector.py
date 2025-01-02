import torch
import torch.nn as nn
import torch.nn.functional as F 

class HallModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, w=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.w = w
    def forward(self, embedding, gradient):
        def avg(vector):
            tem = list(map(lambda x: torch.mean(x, dim=0).squeeze(0), vector))
            return torch.tensor([item.cpu().detach().numpy() for item in tem])
        embedding = avg(embedding)
        gradient = avg(gradient)
        x = self.w * embedding + (1 - self.w) * gradient
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
    