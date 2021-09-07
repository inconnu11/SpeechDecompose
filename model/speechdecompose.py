import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformer import Encoder, Decoder, PostNet
from transformer import PostNet
from model.encoders import ContentEncoder, StyleEncoder
from model.decoders import Decoder
from model.nets_utils import to_gpu
# from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

class SpeechDecompose(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(SpeechDecompose, self).__init__()
        self.model_config = model_config
        ##### encoders, decoder #####
        self.encoder_content = ContentEncoder(model_config)
        self.encoder_style = StyleEncoder(model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        ##### encoders, decoder #####
        self.multi_spk = self.model_config["spk"]["multi_spk"]
        self.use_spk_dvec = self.model_config["spk"]["use_spk_dvec"]
        self.num_speakers = self.model_config["spk"]["num_speakers"]
        self.spk_embed_dim = self.model_config["spk"]["spk_embed_dim"]
        self.multi_styles = self.model_config["styles"]["multi_styles"]
        self.num_styles = self.model_config["styles"]["num_styles"]
        self.style_embed_dim = self.model_config["styles"]["style_embed_dim"]
         
        ## spk ##
        if self.multi_spk:
            if not self.use_spk_dvec:         
                self.spk_embedding = nn.Embedding(self.num_speakers, self.spk_embed_dim) 

        ## style ##         
        if self.multi_styles:
            self.style_embedding = nn.Embedding(self.num_styles, self.style_embed_dim)
            # projection_input_size += self.style_embed_dim                

    def parse_batch(self, batch):
        # text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch[:5]
        mel, mel_lens, max_mel_len = batch[:3]
        mel = to_gpu(mel).long()
        mel_lens = to_gpu(mel_lens).long()
        max_mel_len = to_gpu(max_mel_len).float()
        inputs = (mel, mel_lens, max_mel_len)

        if self.multi_spk:
            if self.use_spk_dvec:
                speaker_embeddings = batch[3]
                speaker_embeddings = to_gpu(speaker_embeddings).float()
                stop_tokens = batch[4] 
                stop_tokens = to_gpu(stop_tokens).long()
                inputs = (mel, mel_lens, max_mel_len, speaker_embeddings, stop_tokens)
            else:
                speaker_ids = batch[3]
                speaker_ids = to_gpu(speaker_ids).long()
                stop_tokens = batch[4] 
                stop_tokens = to_gpu(stop_tokens).long()
                inputs = (mel, mel_lens, max_mel_len, speaker_ids, stop_tokens)
                       

        return (inputs)

    def forward(self, mel = None, mel_lens = None, max_mel_len=None, spembs = None, styleembs = None):
        B, T, _ = mel.size()
        out, mu, log_sigma  = self.encoder_content(mel)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        content_embs = mu + torch.exp(log_sigma / 2) * eps
        x = content_embs

        if self.multi_spk:
            assert spembs is not None
            # spk_embs = self.spk_embedding(torch.LongTensor([0,]*ppg.size(0)).to(ppg.device))
            if not self.use_spk_dvec:
                spk_embs = self.spk_embedding(spembs)
                spk_embs = torch.nn.functional.normalize(
                    spk_embs).unsqueeze(1).expand(-1, T, -1)
            else:
                spk_embs = torch.nn.functional.normalize(
                    spembs).unsqueeze(1).expand(-1, T, -1)
            print("spk_embs shape", spk_embs.shape)
            x = torch.cat([content_embs, spk_embs], dim=2)   


        if self.multi_styles:
            if styleembs is not None:
                style_embs = self.style_embedding(styleembs)
                style_embs = torch.nn.functional.normalize(
                    style_embs).unsqueeze(1).expand(-1, T, -1)
                
            else:
                style_embs = self.encoder_style(mel)
                style_embs = torch.nn.functional.normalize(
                        style_embs).unsqueeze(1).expand(-1, T, -1)
            print("style_embs shape", style_embs.shape)
            x = torch.cat([x, style_embs], dim=2)

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )  
        output, mel_masks = self.decoder(x, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output      
        return (
            output,
            postnet_output,
            mel_masks,
            mel_lens, 
            predict_stop_token,
            mu,
            log_sigma
        )

