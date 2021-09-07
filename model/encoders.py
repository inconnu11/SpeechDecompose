import torch
import torch.nn as nn
import torch.nn.functional as F

    
# conten encoder from https://github.com/jjery2243542/adaptive_voice_conversion
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
    
def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()   

class ContentEncoder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = model_config["content_encoder"]["n_conv_blocks"]
        self.subsample = model_config["content_encoder"]["subsample"]
        self.act = get_act(model_config["content_encoder"]["act"])    #'relu'
        self.c_in = model_config["content_encoder"]["c_in"]
        self.c_h = model_config["content_encoder"]["c_h"]
        self.c_out= model_config["content_encoder"]["c_out"]
        self.kernel_size= model_config["content_encoder"]["kernel_size"]
        self.bank_size= model_config["content_encoder"]["bank_size"]
        self.bank_scale= model_config["content_encoder"]["bank_scale"]
        self.c_bank= model_config["content_encoder"]["c_bank"]
        self.subsample= model_config["content_encoder"]["subsample"]
        self.dropout_rate= model_config["content_encoder"]["dropout_rate"]

        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(self.c_in, self.c_bank, kernel_size=k) for k in range(self.bank_scale, self.bank_size + 1, self.bank_scale)])
        in_channels = self.c_bank * (self.bank_size // self.bank_scale) + self.c_in
        self.in_conv_layer = nn.Conv1d(in_channels, self.c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size) for _ \
                in range(self.n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size, stride=sub) 
            for sub, _ in zip(self.subsample, range(self.n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(self.c_h, affine=False)
        self.mean_layer = nn.Conv1d(self.c_h, self.c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(self.c_h, self.c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        out = self.conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        mu = pad_layer(out, self.mean_layer)
        log_sigma = pad_layer(out, self.std_layer)
        return out, mu, log_sigma
