import torch
import torch.nn as nn


class DisentangleLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(DisentangleLoss, self).__init__()
        # self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
        #     "feature"
        # ]
        # self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
        #     "feature"
        # ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        # inputs: (mel, mel_lens, max_mel_len, speaker_embeddings)
        (
            mel_targets,
            _,
            _,
            _
        ) = inputs[4:]
        
        # predictions (output, postnet_output,mel_masks, mel_lens)
        (
            mel_predictions,
            postnet_mel_predictions,
            mel_masks,
            _,
        ) = predictions

        mel_masks = ~mel_masks

        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        mel_targets.requires_grad = False

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        total_loss = (
            mel_loss + postnet_mel_loss 
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
        )
