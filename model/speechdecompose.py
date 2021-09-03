import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformer import Encoder, Decoder, PostNet
from transformer import PostNet
from model.encoders import Encoder_content, Encoder_style
from model.decoders import Decoder
from model.nets_utils import to_gpu
# from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

class SpeechDecompose(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(SpeechDecompose, self).__init__()
        self.model_config = model_config
        self.encoder_content = Encoder_content(model_config)
        self.encoder_style = Encoder_style(model_config)
        # self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.multi_spk = self.model_config["spk"]["multi_spk"]
        self.use_spk_dvec = self.model_config["spk"]["use_spk_dvec"]
        self.num_speakers = self.model_config["spk"]["num_speakers"]
        self.spk_embed_dim = self.model_config["spk"]["spk_embed_dim"]
        self.multi_styles = self.model_config["styles"]["multi_styles"]
        self.num_styles = self.model_config["styles"]["num_styles"]
        self.style_embed_dim = self.model_config["styles"]["style_embed_dim"]
        if self.multi_spk:
            if not self.use_spk_dvec:         
                self.spk_embedding = nn.Embedding(self.num_speakers, self.spk_embed_dim)  
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
                inputs = (mel, mel_lens, max_mel_len, speaker_embeddings)
            else:
                speaker_ids = batch[3]
                speaker_ids = to_gpu(speaker_ids).long()
                inputs = (mel, mel_lens, max_mel_len, speaker_ids)

        return (inputs)

    def forward(self, mel = None, mel_lens = None, max_mel_len=None, spembs = None, styleembs = None):
        B, T, _ = mel.size()
        content_embs = self.encoder_content(mel)
        
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
            mel_lens
        )






# class FastSpeech2(nn.Module):
#     """ FastSpeech2 """

#     def __init__(self, preprocess_config, model_config):
#         super(FastSpeech2, self).__init__()
#         self.model_config = model_config

#         self.encoder = Encoder(model_config)
#         self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
#         self.decoder = Decoder(model_config)
#         self.mel_linear = nn.Linear(
#             model_config["transformer"]["decoder_hidden"],
#             preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
#         )
#         self.postnet = PostNet()

#         self.speaker_emb = None
#         if model_config["multi_speaker"]:
#             with open(
#                 os.path.join(
#                     preprocess_config["path"]["preprocessed_path"], "speakers.json"
#                 ),
#                 "r",
#             ) as f:
#                 n_speaker = len(json.load(f))
#             self.speaker_emb = nn.Embedding(
#                 n_speaker,
#                 model_config["transformer"]["encoder_hidden"],
#             )

#     def forward(
#         self,
#         speakers,
#         texts,
#         src_lens,
#         max_src_len,
#         mels=None,
#         mel_lens=None,
#         max_mel_len=None,
#         p_targets=None,
#         e_targets=None,
#         d_targets=None,
#         p_control=1.0,
#         e_control=1.0,
#         d_control=1.0,
#     ):
#         src_masks = get_mask_from_lengths(src_lens, max_src_len)
#         mel_masks = (
#             get_mask_from_lengths(mel_lens, max_mel_len)
#             if mel_lens is not None
#             else None
#         )

#         output = self.encoder(texts, src_masks)

#         if self.speaker_emb is not None:
#             output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
#                 -1, max_src_len, -1
#             )

#         (
#             output,
#             p_predictions,
#             e_predictions,
#             log_d_predictions,
#             d_rounded,
#             mel_lens,
#             mel_masks,
#         ) = self.variance_adaptor(
#             output,
#             src_masks,
#             mel_masks,
#             max_mel_len,
#             p_targets,
#             e_targets,
#             d_targets,
#             p_control,
#             e_control,
#             d_control,
#         )

#         output, mel_masks = self.decoder(output, mel_masks)
#         output = self.mel_linear(output)

#         postnet_output = self.postnet(output) + output

#         return (
#             output,
#             postnet_output,
#             p_predictions,
#             e_predictions,
#             log_d_predictions,
#             d_rounded,
#             src_masks,
#             mel_masks,
#             src_lens,
#             mel_lens,
#         )