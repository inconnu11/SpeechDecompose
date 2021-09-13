import torch
import numpy as np
import random


def read_fids(fid_list_f):
    with open(fid_list_f, 'r') as f:
        fids = [l.strip().split('|')[0] for l in f if l.strip()]
    return fids   



class OneshotVcDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        vctk_wav_dir: str,
        vctk_spk_dvec_dir: str,
        vctk_mel_dir: str,
        min_max_norm_mel: bool = False,
        mel_min: float = None,
        mel_max: float = None,
        wav_file_ext: str = "wav",
        mel_file_ext: str = "npy",
    ):
        self.fid_list = read_fids(meta_file)
        self.vctk_wav_dir = vctk_wav_dir
        self.vctk_mel_dir = vctk_mel_dir
        self.vctk_spk_dvec_dir = vctk_spk_dvec_dir
        self.wav_file_ext = wav_file_ext
        self.mel_file_ext = mel_file_ext
        self.n_sample_frames = 128
        self.min_max_norm_mel = min_max_norm_mel
        if min_max_norm_mel:
            print("[INFO] Min-Max normalize Melspec.")
            assert mel_min is not None
            assert mel_max is not None
            self.mel_max = mel_max
            self.mel_min = mel_min
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)
    
    def get_spk_dvec(self, fid):
        ################ lsx : one embedding for one speaker? ##########
        # spk_name = fid.split("_")[0]
        # if spk_name.startswith("p"):
        #     spk_dvec_path = f"{self.vctk_spk_dvec_dir}/{spk_name}.npy"
        # else:
        #     spk_dvec_path = f"{self.libri_spk_dvec_dir}/{spk_name}.npy"
        # return torch.from_numpy(np.load(spk_dvec_path))
        # spk_name = fid.split("_")[0]
        ################ now : one embedding for one utterance? ##########        
        if fid.startswith("p"):
            spk_dvec_path = f"{self.vctk_spk_dvec_dir}/{fid}.npy"
        else:
            spk_dvec_path = f"{self.libri_spk_dvec_dir}/{fid}.npy"
        return torch.from_numpy(np.load(spk_dvec_path))


    def bin_level_min_max_norm(self, melspec):
        # frequency bin level min-max normalization to [-4, 4]
        mel = (melspec - self.mel_min) / (self.mel_max - self.mel_min) * 8.0 - 4.0
        return np.clip(mel, -4., 4.)   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        # 1. Load features
        if fid.startswith("p"):
            spk_id = fid.split('_')[0]
            mel = np.load(f"{self.vctk_mel_dir}/{spk_id}-{'mel'}-{fid}.{self.mel_file_ext}")
        else:
            # libritts
            mel = np.load(f"{self.libri_wav_dir}/{fid}.{self.wav_file_ext}")
        if self.min_max_norm_mel:
            mel = self.bin_level_min_max_norm(mel)
        
        spk_dvec = self.get_spk_dvec(fid)

        # print("mel", mel)
        mel = mel.T  #  -> (80, )
        # print("mel", mel.size())
        melt = mel
        while mel.shape[-1] < self.n_sample_frames:
            mel = np.concatenate([mel, melt], -1)
        # print("mel", mel.size())
        pos = random.randint(0, mel.shape[-1] - self.n_sample_frames)
        # print("pos", pos)
        mel = mel[:, pos:pos + self.n_sample_frames]            
        # print("mel", mel.size())
        mel = torch.from_numpy(mel.T)
        # print("mel", mel.size()) ([80, 128]) - >  ([128, 80])
        # return (ppg, lf0_uv, mel, spk_dvec, fid)
        return (torch.from_numpy(np.array(mel)), spk_dvec, fid)
        # return (torch.from_numpy(np.array(mel)), spk_dvec)
