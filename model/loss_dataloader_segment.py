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
        # self.frames_per_step =  model_config["decoder"]["frames_per_step"]
        self.train_config = train_config

    def forward(self, inputs, predictions, step):
        # inputs: (mel, mel_lens, max_mel_len, speaker_embeddings)
        # (mel_targets, _, _) = inputs
        mel_targets = inputs[0]
        # print("mel target ", mel_targets.size())
        # inputs = (mel, mel_lens, max_mel_len, speaker_embeddings, stop_tokens)
        
        # predictions (output, postnet_output,mel_masks, mel_lens) 
        (mel_predictions, postnet_mel_predictions, content_vq_loss, mu, log_sigma) = predictions
        ################ fastspeech 2 mel mask ##############
        # mel_masks = ~mel_masks
        # mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        # mel_masks = mel_masks[:, :mel_masks.shape[1]]
        # mel_targets.requires_grad = False

        # mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        # postnet_mel_predictions = postnet_mel_predictions.masked_select(
        #     mel_masks.unsqueeze(-1)
        # )
        # mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        ################ fastspeech 2 mel mask ##############

        ########### ppg-vc-lsx MaskedMSELoss ############
        # (B, T, 1)
        # mask = self.get_mask(mel_lens).unsqueeze(-1)
        # # (B, T, D)
        # mask_ = mask.expand_as(mel_targets)
        # print("mask_", mask_)
        # print("mel_predictions", mel_predictions)
        # print("mel_targets", mel_targets)
        # # return ((loss * mask_).sum()) / mask_.sum()
        # # mel_predictions [16, 408, 80], mel_targets ([16, 406, 80])
        # mel_loss = ((self.mae_loss(mel_predictions, mel_targets) * mask_).sum()) / mask_.sum()
        # postnet_mel_loss = ((self.mae_loss(postnet_mel_predictions, mel_targets) * mask_).sum()) / mask_.sum()        
        ########### ppg-vc-lsx MaskedMSELoss ############

        ########### directly mae loss  ############
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
            lambda_kl = self.train_config['lambda']['lambda_kl'] * (step + 1) / self.train_config['lambda']["annealing_iters"]     
        ## total loss ##
        mel_total_loss = (mel_loss + postnet_mel_loss)

        total_loss = self.train_config['lambda']['lambda_rec'] * mel_total_loss + lambda_kl * style_kl_loss + self.train_config['lambda']['lambda_vq'] * content_vq_loss
        # total_loss = (
        #     mel_loss + postnet_mel_loss 
        # )

        # total_loss = (
        #     mel_loss + postnet_mel_loss + stop_loss
        # )

        return (total_loss, mel_total_loss, mel_loss, postnet_mel_loss, style_kl_loss, content_vq_loss), lambda_kl



    def get_mask(self, lengths):
        # lengths: [B,]
        # print("mel len", lengths)
        max_len = torch.max(lengths)
        # print("max len", max_len)  # max len tensor(406., device='cuda:0')  
        batch_size = lengths.size(0)
        # print("batch_size", batch_size)  # 16
        seq_range = torch.arange(0, max_len).long()
        # print("seq_range", seq_range)   # [0,1,2, ... ,405]
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).to(lengths.device)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()        
