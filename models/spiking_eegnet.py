import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate, functional
from modules.layers import TATCA


class EEG_CNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=22, T=64, signal_length=63, use_tatca=True, **kwargs):
        super().__init__()
        self.T = T
        self.use_tatca = use_tatca
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(8),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(input_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        if self.use_tatca:
            self.tatca = TATCA(kernel_size_t=3, T=T, channel=16)

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        # Auto calculate dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_channels, signal_length)
            out = self.conv1(dummy)
            out = self.conv2(out)
            out = self.conv3(out)
            self.flat_dim = out.flatten(1).shape[1]
            
        self.classifier = nn.Linear(self.flat_dim, num_classes)

    def forward(self, x):
        # x: (N, 22, L)
        x = x.unsqueeze(1) # (N, 1, 22, L)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        out = functional.multi_step_forward(x_seq, self.conv1)
        out = functional.multi_step_forward(out, self.conv2)
        

        if self.use_tatca:
            out = self.tatca(out)
            
        out = functional.multi_step_forward(out, self.conv3)
        out = out.flatten(2)
        out = functional.multi_step_forward(out, self.classifier)
        return out.mean(0)


class EEG_MLP(nn.Module):
    def __init__(self, num_classes=4, input_channels=22, T=64, signal_length=63, use_tatca=True, **kwargs):
        super().__init__()
        self.T = T
        self.use_tatca = use_tatca
        
        input_dim = input_channels * signal_length
        hidden_dim = 64 
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.drop1 = nn.Dropout(0.5) 
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) 
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.drop2 = nn.Dropout(0.5)
        
        if self.use_tatca:
            self.tatca = TATCA(kernel_size_t=3, T=T, channel=hidden_dim)
            
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (N, 22, L)
        N = x.shape[0]
        # Flatten input: (N, 22*L)
        x_flat = x.flatten(1)
        # Repeat for time steps: (T, N, Input_Dim)
        x_seq = x_flat.unsqueeze(0).repeat(self.T, 1, 1)
        
        # Step-by-step forward 
        out_list = []
        for t in range(self.T):
            xt = x_seq[t]
            
            # Layer 1
            out = self.fc1(xt)
            out = self.bn1(out)
            out = self.lif1(out)
            out = self.drop1(out)
            
            # Layer 2
            out = self.fc2(out)
            out = self.bn2(out)
            out = self.lif2(out)
            out = self.drop2(out)
            
            out_list.append(out)
            
        # Stack -> (T, N, Hidden)
        out_seq = torch.stack(out_list, dim=0)
        
        if self.use_tatca:
            # Reshape (T, N, C) -> (T, N, C, 1, 1)
            out_seq = out_seq.unsqueeze(-1).unsqueeze(-1)
            out_seq = self.tatca(out_seq)
            # Reshape back -> (T, N, C)
            out_seq = out_seq.squeeze(-1).squeeze(-1)
            
        # Classification
        out_seq = functional.multi_step_forward(out_seq, self.classifier)
        return out_seq.mean(0)


class EEG_RNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=22, T=64, signal_length=63, use_tatca=True, **kwargs):
        super().__init__()
        self.T = T
        self.use_tatca = use_tatca
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(16),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_channels, signal_length)
            out = self.feature_extractor(dummy) 
            # out shape: (1, 16, 22, 15) 
            self.extracted_shape = out.shape[1:] # (16, 22, 15)
            self.extracted_dim = out.flatten(1).shape[1]
            
        if self.use_tatca:
            self.tatca = TATCA(kernel_size_t=3, T=T, channel=16)
            
        self.hidden_size = 64
        
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.extracted_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            nn.Dropout(0.5)
        )
        
        # 4. LSTM
        self.rnn_cell = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # x: (N, 22, L)
        x = x.unsqueeze(1)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # (T, N, 1, 22, L)
        T, N, _, _, _ = x_seq.shape
        
        # input: (T * N, 1, 22, L)
        x_reshaped = x_seq.view(T * N, 1, x_seq.shape[-2], x_seq.shape[-1])
        cnn_feat = self.feature_extractor(x_reshaped) 
        # output: (T * N, 16, 22, 15)
        
        # BACK (T, N, 16, 22, 15)
        cnn_feat = cnn_feat.view(T, N, *self.extracted_shape)
        

        if self.use_tatca:
            cnn_feat = self.tatca(cnn_feat)
            

        h = torch.zeros(N, self.hidden_size).to(x.device)
        c = torch.zeros(N, self.hidden_size).to(x.device)
        
        out_list = []
        
        for t in range(T):
            feat_t = cnn_feat[t] # (N, 16, 22, 15)
            
            feat_low = self.proj(feat_t) # (N, 64)
            
            h, c = self.rnn_cell(feat_low, (h, c))
            out_list.append(h)
            
        out_seq = torch.stack(out_list, dim=0)
        
        out_seq = functional.multi_step_forward(out_seq, self.classifier)
        return out_seq.mean(0)


def create_eeg_model(arch_type, **kwargs):
    arch_type = arch_type.lower()
    if arch_type == 'cnn':
        return EEG_CNN(**kwargs)
    elif arch_type == 'mlp':
        return EEG_MLP(**kwargs)
    elif arch_type == 'rnn':
        return EEG_RNN(**kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch_type}")