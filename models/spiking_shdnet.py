import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate, functional
from modules.layers import TATCA_1D 

class SpikingSHDNet_MLP(nn.Module):
    def __init__(self, input_channels=700, hidden_channels=128, num_classes=20, T=16, tau=2.0, attn_type='tatca'):
        super().__init__()
        self.T = T
    
        self.part1 = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            nn.Linear(hidden_channels, hidden_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        )

        self.attn = TATCA_1D(attn_type, kernel_size_t=3, T=T, channel=hidden_channels)

        self.part2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            nn.Linear(hidden_channels, num_classes) 
        )

    def forward(self, x):
        # x: (N, T, 700)
        N, T_steps, _ = x.shape
        x_seq = x.permute(1, 0, 2) # (T, N, 700)
        
        feat_list = []
        for t in range(T_steps):
            feat = self.part1(x_seq[t])
            feat_list.append(feat)
        features = torch.stack(feat_list, dim=0) # (T, N, Hidden)
        
        # Attention
        attended_features = self.attn(features)

        # Classifier
        output_list = []
        for t in range(T_steps):
            out = self.part2(attended_features[t])
            output_list.append(out)
            
        outputs = torch.stack(output_list, dim=0)
        return outputs.mean(0), features, attended_features


class SpikingSHDNet_CNN(nn.Module):
    def __init__(self, input_channels=700, hidden_channels=128, num_classes=20, T=16, tau=2.0, attn_type='tatca'):
        super().__init__()
        self.T = T
        self.conv_block = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False), 
            nn.BatchNorm1d(32),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            
            # Layer 2
            nn.Conv1d(32, hidden_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            
            # Global Pool
            nn.AdaptiveAvgPool1d(1) 
        )

        self.attn = TATCA_1D(attn_type, kernel_size_t=3, T=T, channel=hidden_channels)

        self.part2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x):
        # x: (N, T, 700)
        N, T, C = x.shape
        # Reshape for Conv1d: (N*T, 1, 700)
        x_reshaped = x.reshape(N * T, 1, C)
        # Conv -> (N*T, Hidden, 1)
        feat_flat = self.conv_block(x_reshaped)
        # Reshape back -> (T, N, Hidden)
        features = feat_flat.view(N, T, -1).permute(1, 0, 2)
        
        # Attention
        attended_features = self.attn(features)
        
        # Classifier
        output_list = []
        for t in range(T):
            out = self.part2(attended_features[t])
            output_list.append(out)
            
        outputs = torch.stack(output_list, dim=0)
        return outputs.mean(0), features, attended_features


class SpikingSHDNet_RNN(nn.Module):
    def __init__(self, input_channels=700, hidden_channels=128, num_classes=20, T=16, tau=2.0, attn_type='tatca'):
        super().__init__()
        self.T = T
        self.hidden_channels = hidden_channels
        
        # Recurrent Layer
        self.fc_in = nn.Linear(input_channels, hidden_channels)
        self.fc_rec = nn.Linear(hidden_channels, hidden_channels)
        self.lif1 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.lif2 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())

        self.attn = TATCA_1D(attn_type, kernel_size_t=3, T=T, channel=hidden_channels)

        self.part2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x):
        # x: (N, T, 700)
        N, T_steps, _ = x.shape
        
        feat_list = []
        spikes_last = torch.zeros(N, self.hidden_channels, device=x.device)
        
        for t in range(T_steps):
            frame = x[:, t, :] # (N, 700)
            
            # Recurrent input
            inp = self.fc_in(frame) + self.fc_rec(spikes_last)
            spikes_t = self.lif1(inp)
            spikes_last = spikes_t
            
            # Forward Layer 2
            x2 = self.fc2(spikes_t)
            feat = self.lif2(x2)
            
            feat_list.append(feat)
        
        features = torch.stack(feat_list, dim=0) # (T, N, C)
        
        # Attention
        attended_features = self.attn(features)
        
        # Classifier
        output_list = []
        for t in range(T_steps):
            out = self.part2(attended_features[t])
            output_list.append(out)
            
        outputs = torch.stack(output_list, dim=0)
        return outputs.mean(0), features, attended_features


def create_shd_model(arch_type, **kwargs):
    arch_type = arch_type.lower()
    if arch_type == 'mlp':
        return SpikingSHDNet_MLP(**kwargs)
    elif arch_type == 'cnn':
        return SpikingSHDNet_CNN(**kwargs)
    elif arch_type == 'rnn':
        return SpikingSHDNet_RNN(**kwargs)
    else:
        raise ValueError(f"Unknown SHD architecture: {arch_type}")