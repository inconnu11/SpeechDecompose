#!/bin/bash

# Copyright  2020  Microsoft (author: Ke Wang)

# set -euo pipefail

# cluster="itp-v100-scus-2"  # itp-p40-eus-01, a100-scus, itp-v100-scus-2, itp-v100-scus, itp-p100-wus2, v100-32gb-wus2-2, v100-16gb-scus
# cluster="v100-32gb-wus2-2"
# num_gpu=8
cluster="itp-v100-scus-2"
num_gpu=4
vc="speech-itp-tts"        # speech-itp-tts, speech-itp-tts-prod, speech-itp-default
distributed="true"   # enable distributed training or not
dist_method="torch"  # torch or horovod

project_name="SPEECHDECOMPOSE"  # project name (e.g., tacotron/fastspeech)
exp_name="CONTENT_STYLE_SPK_0913_VQ_KL_v0"    # experimental name (e.g., Evan/Guy/Jessa)

data_dir="/datablob"
# if the packages not installed in the docker, you can install them here
# extra_env_setup_cmd="pip install tensorflow-gpu==1.13.1 tensorboardX chainer librosa==0.8.0"
# extra_env_setup_cmd="pip install tensorflow-gpu==1.15.0 chainer"
# tensorflow-gpu==1.15.0
extra_env_setup_cmd="pip install tensorflow-gpu joblib parallel_wavegan"
extra_params="--distributed ${distributed}"
extra_params=${extra_params}" --dist-method ${dist_method}"
extra_params=${extra_params}" --data-dir ${data_dir}"

# add some personal config
# extra_params=${extra_params}" --config ${config}"
preemptible="false"  # if ture, can access resources outside your quota


python third_party/Submitter/utils/pt_submit.py \
	    --service "amlk8s" --cluster ${cluster} --virtual-cluster ${vc} \
	    --gpu ${num_gpu} --distributed ${distributed} --preemptible ${preemptible} \
		--image-registry "docker.io" --image-repo "wangkenpu" \
		--image-name "pytorch:1.7.0-py38-cuda10.2-cudnn8-ubuntu18.04" \
		--data-container-name "philly-ipgsp" --model-container-name "philly-ipgsp" \
		--extra-env-setup-cmd "${extra_env_setup_cmd}" --local-code-dir "$(pwd)" \
		--pt-project ${project_name} --exp-name ${exp_name} \
		--run-cmd "python train_ddp_dataloader_segment.py"
        # --run-cmd "python -m torch.distributed.launch --nproc_per_node=8 train_ddp.py --model-dir /modelblob/v-jiewang/MODELS/TRANSFORMER/" 
		# --image-name "pytorch1.3.0-hvd-apex-py37-cuda10.0-cudnn7:latest" \
				# --run-cmd "sleep 300000000;echo"