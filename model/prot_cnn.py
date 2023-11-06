import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """
    
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()   
        
        # Initialize the required layers
        self.skip = torch.nn.Sequential()
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=3, bias=False, padding=1)
        
    def forward(self, x):
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))
        
        return x2 + self.skip(x)
    
class ProtCNN(pl.LightningModule):
    def __init__(self, x, num_classes):
        super(ProtCNN, self).__init__()
        self.embedding = nn.Embedding(len(x), 128)
        # self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, padding=0)
        self.residual_block = nn.Sequential(
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3)
        )
        self.pooling = nn.MaxPool1d(3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(7680, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # x,_ = self.self_attn(x,x,x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.residual_block(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
