#!/bin/bash

# Copyright  2020  Microsoft (author: Ke Wang)

# set -euo pipefail

# cluster="itp-v100-scus"  # itp-p40-eus-01, a100-scus, itp-v100-scus-2, itp-v100-scus, itp-p100-wus2, v100-32gb-wus2-2, v100-16gb-scus
# cluster="v100-32gb-wus2-2"
# num_gpu=8
cluster="itp-v100-scus-2"
# cluster="v100-16gb-scus"
num_gpu=4
vc="speech-itp-tts"        # speech-itp-tts, speech-itp-tts-prod, speech-itp-default
# vc="speech-itp-tts-prod"
distributed="true"   # enable distributed training or not
dist_method="torch"  # torch or horovod

project_name="SPEECHDECOMPOSE_amlt"  # project name (e.g., tacotron/fastspeech)
exp_name="CONTENT_STYLE_SPK_0923_VQ_KL_sampling_22050_top30_3mel_autoencoder_mel-no-norm_mel_forinference_v5"    # experimental name (e.g., Evan/Guy/Jessa)
# exp_name="CONTENT_STYLE_SPK_0914_VQ_KL_PLOT_mel_energy_pitch_but_mismatch_vocoder_scus2_v6"
data_dir="/datablob"
# if the packages not installed in the docker, you can install them here
# extra_env_setup_cmd="pip install tensorflow-gpu==1.13.1 tensorboardX chainer librosa==0.8.0"
# extra_env_setup_cmd="pip install tensorflow-gpu==1.15.0 chainer"
# tensorflow-gpu==1.15.0
extra_env_setup_cmd="pip install pyworld tensorflow-gpu joblib parallel_wavegan"
extra_params="--distributed ${distributed}"
extra_params=${extra_params}" --dist-method ${dist_method}"
extra_params=${extra_params}" --data-dir ${data_dir}"

# add some personal config
# extra_params=${extra_params}" --config ${config}"
preemptible="false"  # if ture, can access resources outside your quota


# python third_party/Submitter/utils/pt_submit.py \
python third_party_new_0922/Submitter/utils/amlt_submit.py \
	    --service "amlk8s" --cluster ${cluster} --virtual-cluster ${vc} \
	    --gpu ${num_gpu} --distributed ${distributed} --preemptible ${preemptible} \
		--image-registry "docker.io" --image-repo "wangkenpu" \
		--image-name "pytorch:1.7.0-py38-cuda10.2-cudnn8-ubuntu18.04" \
		--data-container-name "philly-ipgsp" --model-container-name "philly-ipgsp" \
		--extra-env-setup-cmd "${extra_env_setup_cmd}" --local-code-dir "$(pwd)" \
		--amlt-project ${project_name} --exp-name ${exp_name} \
		--run-cmd "python -u train_ddp_dataloader_segment_submit_success_v21.py"
        # --run-cmd "python -m torch.distributed.launch --nproc_per_node=8 train_ddp.py --model-dir /modelblob/v-jiewang/MODELS/TRANSFORMER/" 
		# --image-name "pytorch1.3.0-hvd-apex-py37-cuda10.0-cudnn7:latest" \
				# --run-cmd "sleep 300000000;echo"
