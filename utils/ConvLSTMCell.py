# FROM 'https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582'
import torch
from torch import nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = out_channels
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def _forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell Output
        C = forget_gate*C_prev + input_gate*self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)
        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
    
    def forward(self, X, H_prev, C_prev):
        combined = torch.cat([X, H_prev], dim=1)
        combined_conv = self.conv(combined)

        # Limitar valores antes de las activaciones
        combined_conv = torch.clamp(combined_conv, -50, 50)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * C_prev + i * g
        # Limitar la celda de memoria
        c_next = torch.clamp(c_next, -20, 20)
        
        h_next = o * self.activation(c_next)
        
        return h_next, c_next