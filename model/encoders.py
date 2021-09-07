import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(ContentEncoder, self).__init__()
        if self.use_instance_norm:
            self.norm_layer = torch.nn.InstanceNorm1d(bottle_neck_feature_dim, affine=False)