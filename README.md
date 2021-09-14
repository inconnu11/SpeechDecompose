Decompose speech into content, style and speaker
# 1. data preprocess
## extract mel spectrogram
Following [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2), ```python3 preprocess.py config/LJSpeech/preprocess.yaml```

## extract d-vector
Following [liusongxiang's ppg-vc](https://github.com/liusongxiang/ppg-vc), ```python3 3_compute_spk_dvecs_no_flatten.py```

### verify the d-vector using t-sne




# 2 train
local GCR debug : ``` python3 -m torch.distributed.launch train_ddp_dataloader_segment_submit.py --model-dir ./model_debug_dir --log-dir ./log_debug_dir -p config/VCTK/preprocess.yaml -m config/VCTK/model.yaml -t config/VCTK/train.yaml```

WebIDE : ```        ```


# Reference



