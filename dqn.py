import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return  self.fc4(x)
    
class DeepQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DeepQNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    

class RobustNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RobustNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim)
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return torch.sigmoid(x)
    