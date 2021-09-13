import torch
import torch.nn as nn
import torch.nn.functional as F
################# adin vc https://github.com/jjery2243542/adaptive_voice_conversion   ###########
from torch.nn.utils import spectral_norm

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()   

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out
def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2]
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out
def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up

class Decoder(nn.Module):
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.c_in = model_config["decoder"]["c_in"]
        self.c_cond = model_config["decoder"]["c_cond"] 
        self.c_h = model_config["decoder"]["c_h"] 
        self.c_out = model_config["decoder"]["c_out"] 
        self.kernel_size = model_config["decoder"]["kernel_size"] 
        self.n_conv_blocks = model_config["decoder"]["n_conv_blocks"] 
        self.upsample = model_config["decoder"]["upsample"] 
        self.act = get_act(model_config["decoder"]["act"])
        self.sn = model_config["decoder"]["sn"] 
        self.dropout_rate = model_config["decoder"]["dropout_rate"]         
        f = spectral_norm if self.sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(self.c_in, self.c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size)) for _ \
                in range(self.n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [f(nn.Conv1d(self.c_h, self.c_h * up, kernel_size=self.kernel_size)) \
                for _, up in zip(range(self.n_conv_blocks), self.upsample)])
        self.norm_layer = nn.InstanceNorm1d(self.c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
                [f(nn.Linear(self.c_cond, self.c_h * 2)) for _ in range(self.n_conv_blocks*2)])
        self.out_conv_layer = f(nn.Conv1d(self.c_h, self.c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, z):
        # print("z shape", z.size())    #([16, 51, 512])三者拼接
        z = z.transpose(1,2)
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            # y = self.norm_layer(y)
            # y = append_cond(y, self.conv_affine_layers[l*2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            # y = self.norm_layer(y)
            # y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                out = y + out
        # print("out shape", out.size()) #([16, 128, 408])  
        out = pad_layer(out, self.out_conv_layer)
        # print("out shape", out.size()) # ([16, 512, 408])  
        return out
    ########### adin ###########
    # def forward(self, z, cond):
    #     out = pad_layer(z, self.in_conv_layer)
    #     out = self.norm_layer(out)
    #     out = self.act(out)
    #     out = self.dropout_layer(out)
    #     # convolution blocks
    #     for l in range(self.n_conv_blocks):
    #         y = pad_layer(out, self.first_conv_layers[l])
    #         y = self.norm_layer(y)
    #         y = append_cond(y, self.conv_affine_layers[l*2](cond))
    #         y = self.act(y)
    #         y = self.dropout_layer(y)
    #         y = pad_layer(y, self.second_conv_layers[l])
    #         if self.upsample[l] > 1:
    #             y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
    #         y = self.norm_layer(y)
    #         y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
    #         y = self.act(y)
    #         y = self.dropout_layer(y)
    #         if self.upsample[l] > 1:
    #             out = y + upsample(out, scale_factor=self.upsample[l]) 
    #         else:
    #             out = y + out
    #     print("out shape", out.size())
    #     out = pad_layer(out, self.out_conv_layer)
    #     print("out shape", out.size())
    #     return out
    ########### adin ###########


################# adin vc https://github.com/jjery2243542/adaptive_voice_conversion   ###########

################# autovc  ###########
# class LinearNorm(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
#         super(LinearNorm, self).__init__()
#         self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

#         torch.nn.init.xavier_uniform_(
#             self.linear_layer.weight,
#             gain=torch.nn.init.calculate_gain(w_init_gain))

#     def forward(self, x):
#         return self.linear_layer(x)

# class ConvNorm(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
#                  padding=None, dilation=1, bias=True, w_init_gain='linear'):
#         super(ConvNorm, self).__init__()
#         if padding is None:
#             assert(kernel_size % 2 == 1)
#             padding = int(dilation * (kernel_size - 1) / 2)

#         self.conv = torch.nn.Conv1d(in_channels, out_channels,
#                                     kernel_size=kernel_size, stride=stride,
#                                     padding=padding, dilation=dilation,
#                                     bias=bias)

#         torch.nn.init.xavier_uniform_(
#             self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

#     def forward(self, signal):
#         conv_signal = self.conv(signal)
#         return conv_signal

# class Decoder(nn.Module):
#     """Decoder module:
#     """
#     def __init__(self, model_config):
#         super(Decoder, self).__init__()
#         self.dim_content = model_config["decoder"]["dim_content"]
#         self.dim_style = model_config["decoder"]["dim_style"]
#         self.dim_spk = model_config["spks"]["spk_embed_dim"]
#         self.dim_pre = model_config["decoder"]["dim_pre"]
#         self.n_lstm_layers = model_config["decoder"]["n_lstm_layers"]
#         self.lstm1 = nn.LSTM(self.dim_content + self.dim_style + self.dim_spk, self.dim_pre, 1, batch_first=True)
        
#         self.act = model_config["decoder"]["act"]   #'relu'
#         self.kernel_size= model_config["decoder"]["kernel_size"]
#         self.stride= model_config["decoder"]["stride"]
#         self.padding= model_config["decoder"]["padding"]
#         self.dilation= model_config["decoder"]["dilation"]

#         convolutions = []
#         for i in range(3):
#             conv_layer = nn.Sequential(
#                 ConvNorm(self.dim_pre,
#                          self.dim_pre,
#                          kernel_size = self.kernel_size, 
#                          stride = self.stride,
#                          padding = self.padding,
#                          dilation = self.dilation, w_init_gain=self.act),
#                 nn.BatchNorm1d(self.dim_pre))
#             convolutions.append(conv_layer)
#         self.convolutions = nn.ModuleList(convolutions)
        
#         self.lstm2 = nn.LSTM(self.dim_pre, 1024, 2, batch_first=True)
        
#         self.linear_projection = LinearNorm(1024, 80)

#         self.max_seq_len = model_config["max_seq_len"]

#     def forward(self, x, mel_masks):
        
#         #self.lstm1.flatten_parameters()
#         x, _ = self.lstm1(x)
#         x = x.transpose(1, 2)
        
#         for conv in self.convolutions:
#             x = F.relu(conv(x))
#         x = x.transpose(1, 2)
        
#         outputs, _ = self.lstm2(x)
        
#         decoder_output = self.linear_projection(outputs)

#         ############## attention(fastspeech2) not written ###########
#         # TODO : mel_masks is for attention 
#         # -- Forward from fastspeech2 decoder
#         batch_size, max_len = x.shape[0], x.shape[1]        
#         if not self.training and x.shape[1] > self.max_seq_len:
#             slf_attn_mask = mel_masks.unsqueeze(1).expand(-1, max_len, -1)
#         else:
#             max_len = min(max_len, self.max_seq_len)            
#             mel_masks = mel_masks[:, :max_len]


#         return decoder_output, mel_masks
################# autovc   ###########    