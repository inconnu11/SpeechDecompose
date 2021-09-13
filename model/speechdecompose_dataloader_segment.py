import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformer import Encoder, Decoder, PostNet
from transformer.Layers import PostNet
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
        self.encoder_content = ContentEncoder(preprocess_config, model_config)
        self.encoder_style = StyleEncoder(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        # self.mel_linear = nn.Linear(
        #     model_config["transformer"]["decoder_hidden"],
        #     preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        # )
        self.mel_linear = nn.Linear(
            model_config["decoder"]["c_out"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )        
        self.postnet = PostNet()
        ##### encoders, decoder #####
        self.multi_spk = self.model_config["spks"]["multi_speaker"]
        self.use_spk_dvec = self.model_config["spks"]["use_spk_dvec"]
        self.num_speakers = self.model_config["spks"]["num_speakers"]
        self.spk_embed_dim = self.model_config["spks"]["spk_embed_dim"]
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

    def parse_batch(self, mel, batch, device):
        # text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch[:5]
        #  (mels_padded, torch.LongTensor(mel_lengths), torch.LongTensor(max_mel_len), spk_ids, stop_tokens)
        # print("tt", tt.dtype)
        # mel = batch[:1].to(device)
        # mel = to_gpu(mel).long()
        # mel_lens = to_gpu(mel_lens).long()
        # print("mel", mel.size())
        # mel = to_gpu(mel).float()
        # mel_lens = to_gpu(mel_lens).long()        
        # max_mel_len = to_gpu(max_mel_len).long()
        # print("parse batch max_mel_len", max_mel_len)
        inputs = (mel)

        if self.multi_spk:
            if self.use_spk_dvec:
                speaker_embeddings = batch[1]
                # speaker_embeddings = to_gpu(speaker_embeddings).float()
                speaker_embeddings = speaker_embeddings.to(device)
                # stop_tokens = batch[4] 
                # stop_tokens = to_gpu(stop_tokens).long()
                inputs = (mel, speaker_embeddings)
            else:
                speaker_ids = batch[1]
                # speaker_ids = to_gpu(speaker_ids).long()
                speaker_ids = speaker_ids.to(device)
                # stop_tokens = batch[4] 
                # stop_tokens = to_gpu(stop_tokens).long()
                inputs = (mel, speaker_ids)
                       

        return (inputs)

    def forward(self, mel = None, spembs = None, fid = None, styleembs = None):
        # print("mel size", mel.size())
        # print("mel shape", mel.shape)  # [406, 80]
        # T, _ = mel.size()
        B, T, _ = mel.size()  # [16, 406, 80]
        z, c, z_beforeVQ, vq_loss, perplexity = self.encoder_content(mel)
        content_embs = z    # (16, 51, 128)
        T_down = content_embs.shape[1]
        x = content_embs

        if self.multi_spk:
            assert spembs is not None
            # spk_embs = self.spk_embedding(torch.LongTensor([0,]*ppg.size(0)).to(ppg.device))
            if not self.use_spk_dvec:
                spk_embs = self.spk_embedding(spembs)
                spk_embs = torch.nn.functional.normalize(
                    spk_embs).unsqueeze(1).expand(-1, T_down, -1)
            else:
                spk_embs = torch.nn.functional.normalize(
                    spembs).unsqueeze(1).expand(-1, T_down, -1)
            # print("spk_embs shape", spk_embs.shape)
            # print("spk emb", spk_embs)
            x = torch.cat([content_embs, spk_embs], dim=2)   
            # print("x shape", x.size())   # 384 = content 128 + spk 256
            # Sizes of tensors must match except in dimension 2. 
            # Got 406 and 51 in dimension 1 


        if self.multi_styles:
            if styleembs is not None:
                # print("styleembs is not None")
                style_embs = self.style_embedding(styleembs)
                style_embs = torch.nn.functional.normalize(
                    style_embs).unsqueeze(1).expand(-1, T_down, -1)
                
            else:
                # print("styleembs is None")
                # style_embs = self.encoder_style(mel)
                out, mu, log_sigma  = self.encoder_style(mel)
                eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
                style_embs = mu + torch.exp(log_sigma / 2) * eps                
                style_embs = torch.nn.functional.normalize(
                        style_embs).unsqueeze(1).expand(-1, T_down, -1)
            # print("style_embs shape", style_embs.shape)
            x = torch.cat([x, style_embs], dim=2)
            # print("x shape", x.size())   ([16, 51, 512])  512 = 384(content 128 + spk 256) + 128 (style)

        # print("max_mel_len",max_mel_len)    
        # print("mel_lens",mel_lens)   
        # mel_masks = (
        #     get_mask_from_lengths(mel_lens, max_mel_len)
        #     if mel_lens is not None
        #     else None
        # )  
        # print("mel masks shape", mel_masks.size())
        # output, mel_masks = self.decoder(x, mel_masks)
        output = self.decoder(x)
        # print("decoder output shape ",output.size()) #([16, 512, 408])  
        output = self.mel_linear(output.transpose(1, 2))
        # print("after mel_linear output shape",output.size()) #([16, 408, 80])  

        postnet_output = self.postnet(output) + output 
        # print("postnet_output shape", postnet_output.size())  # ([16, 408, 80])   
        return (
            output,
            postnet_output,
            vq_loss,
            mu,
            log_sigma
        )

