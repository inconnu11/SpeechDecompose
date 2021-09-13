import torch
import torch.nn as nn
import torch.nn.functional as F

    
# RuntimeError: Given groups=1, weight of size 128 512 1, 
# expected input[16, 406, 80] to have 512 channels, but got 406 channels instead  
def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    # print("kernel_size", kernel_size)  # 1
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # print("pad shape", pad) #(0,0)
    # print("before padding, inp shape", inp.size())
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    # print("after padding, inp shape", inp.size())       
    out = layer(inp)
    return out
    
def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()   
def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

class VQEmbeddingEMA(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding) # only change during forward
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def inference(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        residual = x - quantized
        return quantized, residual, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0) # calculate the distance between each ele in embedding and x

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training: # EMA based codebook learning
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss
        
        residual = x - quantized
        
        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, residual, loss, perplexity


# conten encoder from https://github.com/jjery2243542/adaptive_voice_conversion 
# + vq from https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py

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
        ####### vq layer ######
        self.n_embeddings = model_config["content_encoder"]["VQencoder"]["n_embeddings"]
        self.z_dim = model_config["content_encoder"]["VQencoder"]["z_dim"]
        self.c_dim = model_config["content_encoder"]["VQencoder"]["c_dim"]


        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(self.c_in, self.c_bank, kernel_size=k) for k in range(self.bank_scale, self.bank_size + 1, self.bank_scale)])
        in_channels = self.c_bank * (self.bank_size // self.bank_scale) + self.c_in
        self.in_conv_layer = nn.Conv1d(in_channels, self.c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size) for _ \
                in range(self.n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size, stride=sub) 
            for sub, _ in zip(self.subsample, range(self.n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(self.c_h, affine=False)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        self.codebook = VQEmbeddingEMA(self.n_embeddings, self.z_dim)
        self.rnn = nn.LSTM(self.z_dim, self.c_dim, batch_first=True)


    def forward(self, x):
        # print("x shape", x.size()) # ([16, 406, 80])  
        x = x.transpose(1,2)
        out = conv_bank(x, self.conv_bank, act=self.act)
        # print("put shape", out.size())
        # out = conv_bank(x, self.conv_bank)
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
        # 前面的网络时间轴上下采样了, 1/2 原本输入特征是([16, 128, 406])(batch, feature, frame)，输出的out是([16, 128, 51]) 
        z_beforeVQ = out.transpose(1,2)
        z, r, loss, perplexity = self.codebook(z_beforeVQ) 
        # print("out shape",out.size())  # batch, feature, frame ([16, 128, 51])
        # print("z_beforeVQ shape", z_beforeVQ.size()) # batch, frame, feature ([16, 51, 128])  
        # print("z_afterVQ shape", z.size())  # ([16, 51, 128])  
        # print("r shape", r.size())   # ([16, 51, 128])  
        # print("loss", loss)
        # print("perplexity", perplexity)      
        c, _ = self.rnn(z) # (16, 51, 128) -> (64, 51, 256) 
        # print("c shape", c.size()) #([16, 51, 256])  
        #input.size(-1) must be equal to input_size. Expected 64, got 51  
        return z, c, z_beforeVQ, loss, perplexity
        # mu = pad_layer(out, self.mean_layer)
        # log_sigma = pad_layer(out, self.std_layer)
        # return out, mu, log_sigma

    def inference(self, x):
        print("x shape", x.size())
        x = x.transpose(1,2)        
        out = conv_bank(x, self.conv_bank, act=self.act)
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
        z_beforeVQ = out.transpose(1,2)
        z, r, indices = self.codebook.inference(z_beforeVQ) # z: (bz, 128/2, 64)
        c, _ = self.rnn(z) # (64, 140/2, 64) -> (64, 140/2, 256)
        return z, c, z_beforeVQ, indices




# adin vc speaker encoder
class StyleEncoder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(StyleEncoder, self).__init__()
        self.c_in = model_config["style_encoder"]["c_in"]
        self.c_h = model_config["style_encoder"]["c_h"]
        self.c_out= model_config["style_encoder"]["c_out"]        
        self.kernel_size= model_config["style_encoder"]["kernel_size"]    
        self.n_conv_blocks = model_config["style_encoder"]["n_conv_blocks"]            
        self.n_dense_blocks = model_config["style_encoder"]["n_dense_blocks"]    
        self.subsample= model_config["style_encoder"]["subsample"]        
        self.act = get_act(model_config["style_encoder"]["act"])    #'relu'
        self.bank_size= model_config["style_encoder"]["bank_size"]
        self.bank_scale= model_config["style_encoder"]["bank_scale"]
        self.c_bank= model_config["style_encoder"]["c_bank"]
        self.dropout_rate= model_config["style_encoder"]["dropout_rate"]   

        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(self.c_in, self.c_bank, kernel_size=k) for k in range(self.bank_scale, self.bank_size + 1, self.bank_scale)])
        in_channels = self.c_bank * (self.bank_size // self.bank_scale) + self.c_in
        self.in_conv_layer = nn.Conv1d(in_channels, self.c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size) for _ \
                in range(self.n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(self.c_h, self.c_h, kernel_size=self.kernel_size, stride=sub) 
            for sub, _ in zip(self.subsample, range(self.n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(self.c_h, self.c_h) for _ in range(self.n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(self.c_h, self.c_h) for _ in range(self.n_dense_blocks)])
        self.output_layer = nn.Linear(self.c_h, self.c_out)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.mean_layer = nn.Conv1d(self.c_h, self.c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(self.c_h, self.c_out, kernel_size=1)     

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):
        # print("x shape", x.size())
        x = x.transpose(1,2)        
        out = conv_bank(x, self.conv_bank, act=self.act)
        # out = conv_bank(x, self.conv_bank)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # print("after pooling layer shape", out.shape)  # ([16, 128])  
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        # print("style encoder final style embedding shape", out.shape)  # ([16, 128]) 
        out_for_mean_var = out.unsqueeze(1).transpose(1,2)
        # print("out_for_mean_var shape", out_for_mean_var.size())   #([16, 128, 1]) 
        mu = pad_layer(out_for_mean_var, self.mean_layer).squeeze()    # ([16, 128, 1]) -> ([16, 1, 128]) 
        log_sigma = pad_layer(out_for_mean_var, self.std_layer).squeeze() 
        # print("mu shape", mu.size()) ([16, 128]) 
        # print("log_sigma shape", log_sigma.size()) ([16, 128]) 
        return out, mu, log_sigma        
        # return out