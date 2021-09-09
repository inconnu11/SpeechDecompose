import numpy as np


file_path = "/home/v-jiewang/data/VCTK-corpus/VCTK-Corpus/wav16_downsample_trim_topdb30_d_vector_flatten/GE2E_spkEmbed_step_5805000/p225_001.npy"
file_con = np.load(file_path)
print(file_con)