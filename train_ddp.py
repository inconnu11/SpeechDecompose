import argparse
import os
import math
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.datasets import OneshotVcDataset
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
# from model import FastSpeech2Loss
from model.loss import DisentangleLoss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_parallel_model(model):
    if isinstance(model, torch.nn.DataParallel) or \
       isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return True
    else:
        return False


def main(args, configs):
    preprocess_config, model_config, train_config = configs

    ############################# setting ddp #########################

    torch.backends.cudnn.enabled = train_config["ddp"]["cudnn_enabled"]
    torch.backends.cudnn.benchmark = train_config["ddp"]["cudnn_benchmark"]


    rank=0
    world_size=1
    local_rank='group name'

    # init distributed env
    if train_config["ddp"]["distributed_run"]:
        # dist_init(backend='nccl')
        torch.distributed.init_process_group(backend='nccl')
        rank = ompi_rank()
        # local_rank = ompi_local_rank()
        local_rank = torch.distributed.get_rank()
        # world_size = torch.cuda.device_count()
        world_size = ompi_size()
        if rank == 0:
            print('[Rank 0]: DistributedDataParallel PyTorch Method')
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(rank, local_rank, world_size, flush=True)    
    if rank == 0:
        # print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
        print("Distributed Run:", train_config["ddp"]["distributed_run"])
        print("cuDNN Enabled:", train_config["ddp"]["cudnn_enabled"])
        print("cuDNN Benchmark:", train_config["ddp"]["cudnn_benchmark"])
        print('Final parsed hparams:')
        # pprint(hps.values())
    ############################# setting ddp #########################



    print("Prepare training ...")
    torch.manual_seed(train_config["ddp"]["seed"])
    torch.cuda.manual_seed(train_config["ddp"]["seed"])    
    # Get dataset
    # dataset = Dataset(
    #     "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    # )
    # batch_size = train_config["optimizer"]["batch_size"]
    # group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    # assert batch_size * group_size < len(dataset)
    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size * group_size,
    #     shuffle=True,
    #     collate_fn=dataset.collate_fn,
    # )
    # train_set = VCDataset(hps.Audio.data_dir, hps.Audio.train_meta_file)
    train_set = OneshotVcDataset(
        meta_file= preprocess_config["data"]["train_fid_list"],
        vctk_wav_dir= preprocess_config["data"]["vctk_wav_dir"],
        vctk_spk_dvec_dir= preprocess_config["data"]["vctk_spk_dvec_dir"],
        min_max_norm_mel = preprocess_config["data"]["min_max_norm_mel"],
        mel_min = None,
        mel_max = None,
        wav_file_ext = "wav",
    )


    if train_config["ddp"]["distributed_run"]:
        train_sampler = DistributedSampler(train_set)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=train_config["optimizer"]["batch_size"], sampler=train_sampler,
                                  shuffle=shuffle,
                                  num_workers=train_config["ddp"]["num_workers"],
                                  collate_fn=train_set.collate_fn)

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    print_rank("trying to move the model to GPU")
    # Move it to GPU if you can
    # model.cuda() if torch.cuda.is_available() else model.cpu()
    model.to(device)
    print_rank("moved the model to GPU")
    print_rank("model: {}".format(model))
    if train_config["ddp"]["distributed_run"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    num_param = get_param_num(model)
    Loss = DisentangleLoss(preprocess_config, model_config).to(device)
    print("Number of SpeechDecompose Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        if train_config["ddp"]["distributed_run"]:
            train_loader.sampler.set_epoch(epoch)        
        inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in train_loader:
            for batch in batchs:   # 每一个sample？
                if is_parallel_model(model):
                    batch = model.module.parse_batch(batch)
                else:
                    batch = model.parse_batch(batch)                
                # batch = to_device(batch, device)

                # Forward
                # output = model(*(batch[2:]))
                output = model(*(batch))

                # Cal Loss
                losses = Loss(batch, output)
                # losses : total_loss, mel_loss, postnet_mel_loss
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                # total_loss.backward()
                if train_config["ddp"]["fp16_run"]:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                if step % grad_acc_step == 0:
                    if train_config["ddp"]["fp16_run"]:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), grad_clip_thresh)
                        is_overflow = math.isnan(grad_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_thresh)
                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()                            
                
                # if step % grad_acc_step == 0:
                #     # Clipping gradients to avoid gradient explosion
                #     nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                #     # Update weights
                #     optimizer.step_and_update_lr()
                #     optimizer.zero_grad()

                learning_rate = optimizer._get_lr_scale()
                if (rank == 0):
                    if step % log_step == 0:
                        losses = [l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}".format(
                            *losses
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses, learning_rate=learning_rate)

                    if step % synth_step == 0:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                            learning_rate=learning_rate,
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                            learning_rate=learning_rate,
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                            learning_rate=learning_rate,
                        )

                    if step % val_step == 0:
                        model.eval()
                        message = evaluate(model, step, configs, val_logger, vocoder)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )

    ############# distributed ###############
    parser.add_argument('--model-dir',type=str, required=True,
                        help='directory to save checkpoints')
    parser.add_argument('--log-dir', type=str,default=None,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true', 
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--hparams_json', type=str,
                        required=False, help='hparams json file')
    parser.add_argument('--local_rank', type=int, default=1,
                        required=False, help='rank of current gpu')
    parser.add_argument('--n_gpus', type=int, default=4,
                        required=False, help='number of gpus')
    ############# distributed ###############  
    # 
    ######### submitter make model and log path #########
    args = parser.parse_args()
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(args.model_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)   
    ######### submitter make model and log path  #########


    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
