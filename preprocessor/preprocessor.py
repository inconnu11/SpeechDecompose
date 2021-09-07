import os
import random
import json
import glob2
import math
import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["data"]["raw_path"]
        self.out_dir = config["data"]["preprocessed_path"]
        # self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]


        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]        
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        print("Processing Data ...")
        n_frames = 0
        energy_scaler = StandardScaler()
        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            if "p" not in speaker:
                continue
            speakers[speaker] = i
            wav_dir = os.path.join(self.in_dir, speaker)
            wav_file_list = glob2.glob(f"{wav_dir}/*.wav")
            count_spk = len(wav_file_list)
            cnt_spk = 0
            out = list()            
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue
                basename = wav_name.split(".")[0]
                ret = self.process_utterance(speaker, basename)
                info, energy, n = ret
                out.append(info)
                n_frames += n  
                cnt_spk += 1   
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))                                            
                if cnt_spk <= math.ceil(count_spk * 0.98):
                    with open(os.path.join(self.out_dir, "train.txt"), "a", encoding="utf-8") as f:
                        # for m in out:
                        f.write(info + "\n")                   
                elif cnt_spk <= math.ceil(count_spk * 0.99):
                    with open(os.path.join(self.out_dir, "val.txt"), "a", encoding="utf-8") as f:
                        # for m in out:
                        f.write(info + "\n")  
                else:
                    with open(os.path.join(self.out_dir, "test.txt"), "a", encoding="utf-8") as f:
                        # for m in out:
                        f.write(info + "\n")                        
                            
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )            
        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))
        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        # with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
        #     for m in out[self.val_size :]:
        #         f.write(m + "\n")
        # with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
        #     for m in out[: self.val_size]:
        #         f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav.astype(np.float32)

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)


        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker]),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
