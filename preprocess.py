import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

import argparse
import yaml
from preprocessor.preprocessor import Preprocessor
import os
import soundfile as sf
import resampy


def prepare_align(config):
    in_dir = config["data"]["vctk_wav_dir"]
    out_dir = config["data"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    # cleaners = config["preprocessing"]["text"]["text_cleaners"]

    # for libritts
    # for speaker in tqdm(os.listdir(in_dir)):
    #     for chapter in os.listdir(os.path.join(in_dir, speaker)):
    #         for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
    for speaker in tqdm(os.listdir(in_dir)):
        for file_name in os.listdir(os.path.join(in_dir, speaker)):
            # for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):    
            if file_name[-4:] != ".wav":
                continue
            base_name = file_name[:-4]

            wav_path = os.path.join(
                in_dir, speaker, "{}.wav".format(base_name)
            )

            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            # wav, _ = librosa.load(wav_path, sampling_rate)
            wav, fs = sf.read(wav_path)
            # wav, _ = librosa.effects.trim(wav, top_db=60)
            wav, _ = librosa.effects.trim(wav, top_db=30)
            # print("fs", fs)
            # print("sampling_rate",sampling_rate)
            if fs != sampling_rate:
                wav = resampy.resample(wav, fs, sampling_rate, axis=0)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/VCTK/preprocess.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    prepare_align(config)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
