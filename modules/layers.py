import torch
import torch.nn as nn
import torch.nn.functional as F

class TATCA(nn.Module):
    """
    Task-Adaptive Temporal-Channel Attention.
    """
    def __init__(self, kernel_size_t: int = 3, T: int = 8, channel: int = 128, reduction: int = 4):
        super().__init__()

        self.temporal_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size_t, padding='same', groups=channel, bias=False)

        self.channel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.gate_params = nn.Parameter(torch.ones(2))
        #self.gate_params = nn.Parameter(torch.tensor([2.0, 0.0]))

    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        x_pooled = torch.mean(x_seq, dim=[3, 4])
        x_permuted = x_pooled.permute(1, 2, 0) 

        t_attn = self.temporal_conv(x_permuted)
        c_attn = self.channel_avg_pool(x_permuted)
        c_attn = self.channel_mlp(c_attn)

        gates = F.softmax(self.gate_params, dim=0)
        gated_t_attn = t_attn * gates[0]
        gated_c_attn = c_attn * gates[1]


        total_attn = self.sigmoid(gated_t_attn + gated_c_attn)
        total_attn = total_attn.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
        y_seq = x_seq * total_attn
        
        return y_seq

class TA(nn.Module):
    """
    only includes Temporal attention.
    """
    def __init__(self, kernel_size_t: int = 3, T: int = 8, channel: int = 128, reduction: int = 4):
        super().__init__()
        self.temporal_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size_t, padding='same', groups=channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        x_pooled = torch.mean(x_seq, dim=[3, 4])
        x_permuted = x_pooled.permute(1, 2, 0)
        t_attn = self.temporal_conv(x_permuted)
        total_attn = self.sigmoid(t_attn)
        total_attn = total_attn.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
        y_seq = x_seq * total_attn
        return y_seq

class CA(nn.Module):
    """
    only includes channel attention.
    """
    def __init__(self, kernel_size_t: int = 3, T: int = 8, channel: int = 128, reduction: int = 4):
        super().__init__()
        self.channel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        x_pooled = torch.mean(x_seq, dim=[3, 4])
        x_permuted = x_pooled.permute(1, 2, 0)
        c_attn = self.channel_avg_pool(x_permuted)
        c_attn = self.channel_mlp(c_attn)
        total_attn = self.sigmoid(c_attn)
        total_attn = total_attn.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
        y_seq = x_seq * total_attn
        return y_seq

class TATCA_1D(nn.Module):
    """
    The 1D version of Temporal-Channel Attention.
    """
    def __init__(self, kernel_size_t: int = 3, T: int = 16, channel: int = 128, reduction: int = 4):
        super().__init__()
        self.temporal_conv = nn.Conv1d(in_channels=channel, out_channels=channel, 
                                       kernel_size=kernel_size_t, padding='same', 
                                       groups=channel, bias=False)

        self.channel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.gate_params = nn.Parameter(torch.ones(2))

    def forward(self, x_seq: torch.Tensor):
        x_permuted = x_seq.permute(1, 2, 0) # (N, C, T)
        t_attn = self.temporal_conv(x_permuted)
        
        c_attn = self.channel_avg_pool(x_permuted)
        c_attn = self.channel_mlp(c_attn) # (N, C, 1)

        gates = F.softmax(self.gate_params, dim=0)
        gated_t_attn = t_attn * gates[0]
        gated_c_attn = c_attn * gates[1]

        total_attn = self.sigmoid(gated_t_attn + gated_c_attn) # (N, C, T)
        self.last_attn_map = total_attn.detach().cpu()

        total_attn = total_attn.permute(2, 0, 1)
        y_seq = x_seq * total_attn
        
        return y_seq

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)
