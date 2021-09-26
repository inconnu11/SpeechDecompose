from speaker_encoder.audio import preprocess_wav
from utils.tools import plot_mel
import hydra
import hydra.utils as utils

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

import soundfile as sf

# from model_encoder import Encoder, Encoder_lf0
# from model_decoder import Decoder_ac
# from model_encoder import SpeakerEncoder as Encoder_spk
from utils.model import get_model, get_vocoder
import torch.nn.functional as F
import os
import random

from glob import glob
import subprocess
# from spectrogram import logmelspectrogram
import kaldiio
import librosa
import resampy
import pyworld as pw
import audio as Audio
import argparse
import yaml
from speaker_encoder.voice_encoder import SpeakerEncoder
from scipy.io import wavfile
from utils.model import vocoder_infer
from matplotlib import pyplot as plt
######################## adopted from VQMIVC(my modified version) #########################
def select_wavs(paths, min_dur=2, max_dur=8):
    pp = []
    for p in paths:
        x, fs = sf.read(p)
        if len(x)/fs>=min_dur and len(x)/fs<=8:
            pp.append(p)
    return pp
def utt_make_frames(x):
    frame_size = 128
    # remains = x.size(0) % frame_size 
    remains = x.size(1) % frame_size 
    # print("remains", remains)
    if remains != 0:
        x = F.pad(x, (0, 128-remains))
    # out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
    # print("out ", out.shape)
    # return out
    return x

def bin_level_min_max_norm(melspec, preprocess_config):
    # frequency bin level min-max normalization to [-4, 4]
    print("[INFO] Min-Max normalize Melspec.")
    print("melspec", melspec)
    mel_min = preprocess_config["data"]["mel_min"]
    mel_max = preprocess_config["data"]["mel_max"]
    mel = (melspec - mel_min) / (mel_max - mel_min) * 8.0 - 4.0
    print("np.clip(mel, -4., 4.) ", np.clip(mel, -4., 4.) )
    return np.clip(mel, -4., 4.) 
def denorm_bin_level_min_max(melspec, preprocess_config):
    print("[INFO] Min-Max DeNormalize Melspec.")
    print("melspec", melspec)
    mel_min = preprocess_config["data"]["mel_min"]
    mel_max = preprocess_config["data"]["mel_max"]
    mel = (melspec + 4.0 ) / 8.0 * (mel_max - mel_min) + mel_min
    print("mel", mel)
    return mel

def extract_mel_fs2_d_vector(wav_path, preprocess_config):
        # Read and trim wav files
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        wav, fs = sf.read(wav_path)
        # wav, _ = librosa.effects.trim(wav, top_db=60)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        # print("fs", fs)
        # print("sampling_rate",sampling_rate)
        if fs != sampling_rate:
            wav = resampy.resample(wav, fs, sampling_rate, axis=0)
        # wav = wav / max(abs(wav)) * max_wav_value
        wav = wav.astype(np.float32)

        # Compute mel-scale spectrogram and energy
        tacotron_stft = Audio.stft.TacotronSTFT(
                preprocess_config["preprocessing"]["stft"]["filter_length"],
                preprocess_config["preprocessing"]["stft"]["hop_length"],
                preprocess_config["preprocessing"]["stft"]["win_length"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                preprocess_config["preprocessing"]["mel"]["mel_fmin"],
                preprocess_config["preprocessing"]["mel"]["mel_fmax"],
            )        
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, tacotron_stft)
        # pp = wav_path.split('/')[-1]
        # print("pp", pp)  #p225_062.wav
        # Compute d-vector speaker embedding
        weights_fpath = preprocess_config["preprocessing"]["spk_emb"]["pretrained"]
        encoder = SpeakerEncoder(weights_fpath)
        speaker_embedding = encoder.embed_utterance(wav)
        return (mel_spectrogram, speaker_embedding)


# @hydra.main(config_path="config/convert.yaml")
# def convert(cfg):
def convert(args, configs):
    preprocess_config, model_config, train_config = configs
    # src_wav_paths = glob('./Dataset/VCTK-Corpus/wav48_silence_trimmed/p225/*mic1.flac') # modified to absolute wavs path, can select any unseen speakers
    # tar1_wav_paths = glob('./Dataset/VCTK-Corpus/wav48_silence_trimmed/p231/*mic1.flac') # can select any unseen speakers
    # tar2_wav_paths = glob('./Dataset/VCTK-Corpus/wav48_silence_trimmed/p243/*mic1.flac') # can select any unseen speakers
    src_wav_paths = glob('/home/v-jiewang/data/VCTK-corpus/VCTK-Corpus/wav48/p225/*.wav') # modified to absolute wavs path, can select any unseen speakers
    tar1_wav_paths = glob('/home/v-jiewang/data/VCTK-corpus/VCTK-Corpus/wav48/p231/*.wav') # can select any unseen speakers
    tar2_wav_paths = glob('/home/v-jiewang/data/VCTK-corpus/VCTK-Corpus/wav48/p243/*.wav') # can select any unseen speakers    
    src_wav_paths = select_wavs(src_wav_paths)
    tar1_wav_paths = select_wavs(tar1_wav_paths)
    tar2_wav_paths = select_wavs(tar2_wav_paths)
    # print("tar1_wav_paths",tar1_wav_paths)
    # print("tar1_wav_paths shape", tar1_wav_paths.size())
    tar1_wav_paths = [sorted(tar1_wav_paths)[0]]
    tar2_wav_paths = [sorted(tar2_wav_paths)[0]]

    # print("src_wav_paths", src_wav_paths)
    # print("tar1_wav_paths", tar1_wav_paths)
    # print("tar2_wav_paths", tar2_wav_paths)

    print('len(src):', len(src_wav_paths), 'len(tar1):', len(tar1_wav_paths), 'len(tar2):', len(tar2_wav_paths)) # 214, 1, 1

    checkpoint_path = args.model_path
    # print("checkpoint_path", checkpoint_path)  ./ckpt_from_azure/100000.pth.tar  
    tmp = checkpoint_path.split('/')
    # print("tmp", tmp)  ['.', 'ckpt_from_azure', '100000.pth.tar']   
    # steps = tmp[-1].split('-')[-1].split('.')[0]
    steps = tmp[-1].split('.')[0]
    # out_dir = f'converted_results/{tmp[-3]}-{tmp[-2]}-{steps}'
    out_dir = f'converted_results/autoencoder-{steps}'
    out_dir = Path(utils.to_absolute_path(out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get model
    model = get_model(args, configs, device, train=False)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"]) 

    ############## get fs2 vocoder ###########
    vocoder = get_vocoder(model_config, device) 
    ############## get fs2 vocoder ###########   
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    min_max_norm_mel = preprocess_config["data"]["min_max_norm_mel"]
    # feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir)+'/feats.1'))
    for i, src_wav_path in tqdm(enumerate(src_wav_paths, 1)):
        if i>10:
            break
        mel, speaker_source = extract_mel_fs2_d_vector(src_wav_path, preprocess_config)
        # print("mel shape", mel.shape)    #(80, 401)  
        # print("speaker_source", speaker_source)
        # print("speaker_source shape", np.array(speaker_source).shape) (256, )

        if i % 2 == 1:
            ref_wav_path = random.choice(tar2_wav_paths)
            tar = 'tarMale_'
        else:
            ref_wav_path = random.choice(tar1_wav_paths)
            tar = 'tarFemale_'
        ref_mel, speaker_target = extract_mel_fs2_d_vector(ref_wav_path, preprocess_config)
        ######################### lsx min-max norm ####################
        if min_max_norm_mel:
            mel = bin_level_min_max_norm(mel, preprocess_config)
            ref_mel = bin_level_min_max_norm(ref_mel, preprocess_config)
        ######################### lsx min-max norm ####################

        ######################### VQMIVC mean-std norm ####################
        # mel_stats = np.load('./preprocessed_data/VCTK_22050_trim30/mel_stats.npy')
        # mean = mel_stats[0]
        # std = mel_stats[1]
        # print("mean", mean.shape)      
        # mel_norm = (mel_origin.T - mean) / (std + 1e-8)   #(80, 401) -> (401,80) 
        # ref_mel_norm = (ref_mel_origin.T - mean) / (std + 1e-8)
        # mel = mel_norm.T
        # ref_mel = ref_mel_norm.T
        ######################### VQMIVC mean-std norm ####################

        # print("orginal mel", torch.FloatTensor(mel_origin).size())   #([80, 401])  
        


        ############################## padding to 128 * #########################
        # mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)
        # ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
        # mel, ref_mel = length_check(torch.FloatTensor(mel), torch.FloatTensor(ref_mel))
        # mel = utt_make_frames(torch.FloatTensor(mel)) 
        mel = torch.FloatTensor(mel)
        # # # print("after utt_make_frames mel", mel.size())   #([80, 512]) 
        # ref_mel = utt_make_frames(torch.FloatTensor(ref_mel)) 
        ref_mel = torch.FloatTensor(ref_mel)  
        ############################## padding to 128 * #########################  
        out_filename = os.path.basename(src_wav_path).split('.')[0] 
        print("out_filename", out_filename)  
        out_filename_ref = os.path.basename(ref_wav_path).split('.')[0] 
        print("out_filename_ref", out_filename_ref)   


        tacotron_stft = Audio.stft.TacotronSTFT(
                preprocess_config["preprocessing"]["stft"]["filter_length"],
                preprocess_config["preprocessing"]["stft"]["hop_length"],
                preprocess_config["preprocessing"]["stft"]["win_length"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                preprocess_config["preprocessing"]["mel"]["mel_fmin"],
                preprocess_config["preprocessing"]["mel"]["mel_fmax"],
            )            
        Audio.tools.inv_mel_spec(mel=mel, out_filename='./converted_results/autoencoder-5000_GL/' + 'mel-' + out_filename + '.wav', _stft=tacotron_stft)
        Audio.tools.inv_mel_spec(mel=ref_mel, out_filename='./converted_results/autoencoder-5000_GL/' + 'refmel-' + out_filename_ref + '.wav', _stft=tacotron_stft)
        mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)   #(80, 401) -> (401, 80) -> (1, 401, 80)
        ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)  
 
        # print("mel", mel.shape)    # ([1, 512, 80])  
        # print("ref_mel", ref_mel.shape)     # ([1, 128, 80]) 
        # print(speaker_target.shape) # (256,)
        speaker_source = torch.FloatTensor(speaker_source).unsqueeze(0).to(device) #(256, )->(1,256)
        speaker_target = torch.FloatTensor(speaker_target).unsqueeze(0).to(device) #(256, )->(1,256)
        # print(speaker_target.shape)  #torch.Size([1, 256])
      
        # batch = (mel_content, mel_spk, mel_style, mel_autoencoder, speaker_embeddings, fid)
        batch_reconstruct = (mel, mel, mel, mel, speaker_source)
        batch_convert_spk = (mel, ref_mel, mel, mel, speaker_target)   # actually, the speaker is controlled by speaker embedding
        batch_convert_style = (mel, mel, ref_mel, mel, speaker_source)
        # batch_convert_content = (ref_mel, mel, mel, mel, speaker_source)
        with torch.no_grad():
            # z, _, _, _ = encoder.encode(mel)
            # lf0_embs = encoder_lf0(lf0)
            # spk_embs = encoder_spk(ref_mel)
            # output = decoder(z, lf0_embs, spk_embs)
            output_reconstruct, post_output_reconstruct, _, _, _ = model.inference(*(batch_reconstruct))
            output_convert_spk, post_output_convert_spk, _, _, _ = model.inference(*(batch_convert_spk))
            output_convert_style, post_output_convert_style, _, _, _ = model.inference(*(batch_convert_style))
            # output_reconstruct = model(*(batch_reconstruct))
            # output_convert_spk = model(*(batch_convert_spk))
            # output_convert_style = model(*(batch_convert_style))  
            # print("output_reconstruct", output_reconstruct)        
            # logmel = output.squeeze(0).cpu().numpy()
            # logmel_reconstruct = output_reconstruct.squeeze(0).cpu().numpy()
            # logmel_convert_spk = output_convert_spk.squeeze(0).cpu().numpy()
            # logmel_convert_style = output_convert_style.squeeze(0).cpu().numpy()
            print("mel", mel.shape)
            print("ref mel", ref_mel.shape)
            print("output_reconstruct", output_reconstruct.shape)
            print("post_output_reconstruct", post_output_reconstruct.shape)
            print("output_convert_spk", output_convert_spk.shape)
            print("post_output_convert_spk", post_output_convert_spk.shape)
            print("output_convert_style", output_convert_style.shape)
            print("post_output_convert_style", post_output_convert_style.shape)


            # feat_writer[out_filename+'_reconstruct'] = logmel_reconstruct
            # feat_writer[out_filename+'_convert_spk'] = logmel_convert_spk
            # feat_writer[out_filename+'_convert_style'] = logmel_convert_style


            # print("mel to synthesize", mel.size())  # ([1, 256, 80])   ([1, 128, 80]) ([1, 384, 80])  
            # feat_writer[out_filename+'_src'] = mel.squeeze(0).cpu().numpy().T
            # feat_writer[out_filename+'_ref'] = ref_mel.squeeze(0).cpu().numpy().T
            # print("mel.cpu().numpy().T", mel.cpu().numpy().T.shape)  # (1, 512, 80)  -> (80, 512, 1) 
            # print("mel.cpu().numpy().T", mel.cpu().numpy().T)
            # print("mel", mel.shape)
            
            ######################### lsx min-max norm ####################
            if min_max_norm_mel:
                mel = denorm_bin_level_min_max(mel, preprocess_config)
                ref_mel = denorm_bin_level_min_max(ref_mel, preprocess_config)
                # post_output_reconstruct = denorm_bin_level_min_max(post_output_reconstruct, preprocess_config)
                # post_output_convert_spk = denorm_bin_level_min_max(post_output_convert_spk, preprocess_config)
                # post_output_convert_style = denorm_bin_level_min_max(post_output_convert_style, preprocess_config)
                # output_reconstruct = denorm_bin_level_min_max(output_reconstruct, preprocess_config)
                # output_convert_spk = denorm_bin_level_min_max(output_convert_spk, preprocess_config)
                # output_convert_style = denorm_bin_level_min_max(output_convert_style, preprocess_config)               
            ######################### lsx min-max norm ####################

            ############################### input different shape mel ################################################################
            # print("mel", mel.shape)   # torch.Size([1, 512, 80])  
            # print("ref_mel", ref_mel.shape)  # torch.Size([1, 128, 80])
            # print("post_output_reconstruct", post_output_reconstruct.shape)  # torch.Size([1, 512, 80])
            # print("post_output_convert_spk", post_output_convert_spk.shape) # torch.Size([1, 512, 80])
            # print("post_output_convert_style", post_output_convert_style.shape) # torch.Size([1, 512, 80])
            # print("output_reconstruct", output_reconstruct.shape) # torch.Size([1, 512, 80])
            # print("output_convert_spk", output_convert_spk.shape) # torch.Size([1, 512, 80])
            # print("output_convert_style", output_convert_style.shape)  # torch.Size([1, 512, 80])


            ################# (1, frame, 80) -> [1, 80, frame] #################
            if vocoder is not None:
                wav_src_gen = vocoder_infer(
                    mel.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]
                wav_ref_gen = vocoder_infer(
                    ref_mel.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]  
                ########################## postnet output #####################
                postnet_wav_reconstruction = vocoder_infer(
                    post_output_reconstruct.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]
                postnet_wav_convert_spk = vocoder_infer(
                    post_output_convert_spk.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]   
                postnet_wav_convert_style = vocoder_infer(
                    post_output_convert_style.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0] 
                ########################## postnet output #####################

                wav_reconstruction = vocoder_infer(
                    output_reconstruct.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]
                wav_convert_spk = vocoder_infer(
                    output_convert_spk.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]   
                wav_convert_style = vocoder_infer(
                    output_convert_style.transpose(1, 2),
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]                                     
            else:
                wav_reconstruction = wav_prediction = None

            # path = '/home/v-jiewang/SpeechDecompose/converted_results/autoencoder-100000-melgan-denorm-fs2'
 
            wavfile.write(os.path.join(out_dir, "{}_src_gen.wav".format(out_filename)), sampling_rate, wav_src_gen)
            wavfile.write(os.path.join(out_dir, "{}_ref_gen.wav".format(out_filename)), sampling_rate, wav_ref_gen)
            ######################### postnet output #####################
            wavfile.write(os.path.join(out_dir, "{}_reconstruct_postnet.wav".format(out_filename)), sampling_rate, postnet_wav_reconstruction)
            wavfile.write(os.path.join(out_dir, "{}_convert_spk_postnet.wav".format(out_filename)), sampling_rate, postnet_wav_convert_spk)
            wavfile.write(os.path.join(out_dir, "{}_convert_style_postnet.wav".format(out_filename)), sampling_rate, postnet_wav_convert_style)  
            ########################## postnet output #####################
            wavfile.write(os.path.join(out_dir, "{}_reconstruct.wav".format(out_filename)), sampling_rate, wav_reconstruction)
            wavfile.write(os.path.join(out_dir, "{}_convert_spk.wav".format(out_filename)), sampling_rate, wav_convert_spk)
            wavfile.write(os.path.join(out_dir, "{}_convert_style.wav".format(out_filename)), sampling_rate, wav_convert_style)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    # convert(args, model_config)
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--model_path", type=str, default='./ckpt_from_azure/no_norm_0923_v4/5000.pth.tar'
    )
    parser.add_argument(
        "-p", "--preprocess_config", type=str, default='./config/VCTK/preprocess.yaml'
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default='./config/VCTK/model.yaml'
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default='./config/VCTK/train.yaml'
    )    
    args = parser.parse_args()  

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
  
    convert(args, configs)

