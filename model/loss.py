import torch
import torch.nn as nn


class DisentangleLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(DisentangleLoss, self).__init__()
        # self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
        #     "feature"
        # ]
        # self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
        #     "feature"
        # ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.stop_loss_criterion = nn.BCEWithLogitsLoss(reduction='none') 
        self.frames_per_step =  model_config["decoder"]["frames_per_step"]
        self.train_config = train_config

    def forward(self, inputs, predictions, step):
        # inputs: (mel, mel_lens, max_mel_len, speaker_embeddings)
        (mel_targets, mel_lens, max_mel_len, speaker_embeddings, stop_tokens) = inputs
        # inputs = (mel, mel_lens, max_mel_len, speaker_embeddings, stop_tokens)
        
        # predictions (output, postnet_output,mel_masks, mel_lens) 
        (mel_predictions, postnet_mel_predictions, mel_masks, mel_lens, content_vq_loss, mu, log_sigma) = predictions

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

        # stop token loss
        # B = stop_tokens.size(0)
        # stop_tokens = stop_tokens.reshape(B, -1, self.frames_per_step)[:, :, 0]
        # stop_lengths = torch.ceil(mel_lens.float() / self.frames_per_step).long()
        # stop_mask = self.get_mask(stop_lengths, int(mel_targets.size(1)/self.frames_per_step))        
        # stop_loss = torch.sum(self.stop_loss_criterion(predict_stop_token, stop_tokens) * stop_mask) / stop_mask.sum()

        style_kl_loss = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
        if step >= self.train_config["lambda"]["annealing_iters"]:
            lambda_kl = self.train_config['lambda']['lambda_kl']
        else:
            lambda_kl = self.train_config['lambda']['lambda_kl'] * (step + 1) / self.train_config["annealing_iters"]     
        ## total loss ##
        mel_total_loss = (mel_loss + postnet_mel_loss)

        total_loss = self.train_config['lambda']['lambda_rec'] * mel_total_loss + lambda_kl * style_kl_loss + self.train_config['lambda']['lambda_vq'] * content_vq_loss
        # total_loss = (
        #     mel_loss + postnet_mel_loss 
        # )

        # total_loss = (
        #     mel_loss + postnet_mel_loss + stop_loss
        # )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            style_kl_loss,
            lambda_kl, 
            vq_loss
        )
