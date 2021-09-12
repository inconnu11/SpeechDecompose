import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
# from model import FastSpeech2Loss
# from dataset import Dataset
from datasets.datasets import OneshotVcDataset
from model.loss import DisentangleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    # evalset = Dataset(
    #     "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    # )
    # batch_size = train_config["optimizer"]["batch_size"]
    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=dataset.collate_fn,
    # )

    validate_set = OneshotVcDataset(
        meta_file= preprocess_config["data"]["validate_fid_list"],
        vctk_wav_dir= preprocess_config["data"]["vctk_wav_dir"],
        vctk_spk_dvec_dir= preprocess_config["data"]["vctk_spk_dvec_dir"],
        min_max_norm_mel = preprocess_config["data"]["min_max_norm_mel"],
        mel_min = None,
        mel_max = None,
        wav_file_ext = "wav",
    )


    validate_loader = DataLoader(validate_set, batch_size=train_config["optimizer"]["batch_size"],
                                shuffle=False, num_workers=train_config["ddp"]["num_workers"],
                                collate_fn=validate_set.collate_fn)
    # Get loss function
    # Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    Loss = DisentangleLoss(preprocess_config, model_config).to(device)
    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in validate_loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(validate_set) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
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
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)