import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)



class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        dim_content=64, dim_style=1, dim_spk=256, dim_pre=512
        
        self.lstm1 = nn.LSTM(dim_content + dim_style + dim_spk, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

        self.max_seq_len = config["max_seq_len"]

    def forward(self, x, mel_masks):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        ############## attention(fastspeech2) not written ###########
        # TODO : mel_masks is for attention 
        # -- Forward from fastspeech2 decoder
        batch_size, max_len = x.shape[0], x.shape[1]        
        if not self.training and x.shape[1] > self.max_seq_len:
            slf_attn_mask = mel_masks.unsqueeze(1).expand(-1, max_len, -1)
        else:
            max_len = min(max_len, self.max_seq_len)            
            mel_masks = mel_masks[:, :max_len]


        return decoder_output, mel_masks
    