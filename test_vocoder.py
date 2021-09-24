

from utils.model import vocoder_infer
wav_src_gen = vocoder_infer(
    mel.transpose(1, 2),
    vocoder,
    model_config,
    preprocess_config,
)[0]

