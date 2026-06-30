import torch
import config
from torch import nn
from .ConvLSTMCell import ConvLSTMCell

device = torch.device(config.cuda_device)

class _ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        batch_size, seq_len, _, height, width = X.size()

        output = torch.zeros(batch_size, seq_len, self.out_channels, 
        height, width, device=device)
        
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        C = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:,time_step], H, C)
            output[:,time_step,:] = H

        return output
    
class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        batch_size, seq_len, _, height, width = X.size()

        output = torch.zeros(batch_size, seq_len, self.out_channels, 
        height, width, device=device)
        
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        C = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:,time_step], H, C)
            output[:,time_step] = H

        return output