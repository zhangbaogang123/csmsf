import torch
import torch.nn as nn
import lightning.pytorch as pl
import time
class TransfomerDecoder(nn.Module):
    def __init__(self, dim=768, num_output_tokens=77, nhead=1, num_layers=2, dropout=0.2, dim_feedforward=1024):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_output_tokens, dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, batch_first=True, dropout=dropout,
                                                   norm_first=True, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory):  # memory: (bs, 257, 768)
        bs = memory.size(0)
        tgt = self.queries.expand(bs, -1, -1)
        out = self.decoder(tgt, memory)  # (bs, 77, 768)
        return out


class ElementwiseComparisonLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(ElementwiseComparisonLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            raise ValueError("The shapes of both tensors must be the same.")

        mse_loss = nn.MSELoss()
        mse_loss_value = mse_loss(tensor1, tensor2)
        cos_loss = CosineSimilarityLoss()
        cos_loss_value = cos_loss(tensor1, tensor2)

        mse_cos_loss = 0.5 * mse_loss_value * 1000 + cos_loss_value

        return {"mse_cos_loss": mse_cos_loss, "mse_loss": mse_loss_value,
                "cos_loss_value": cos_loss_value}


class CosineSimilarityLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        similarity = nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)
        similarity = (similarity + 1.0) / 2.0
        loss = 1.0 - similarity
        return loss.mean()


def cos_loss(preds, targs):
    similarity = nn.functional.cosine_similarity(preds, targs, dim=-1)
    similarity = (similarity + 1.0) / 2.0
    loss = 1.0 - similarity
    return loss.mean()


class FineTunedModel(pl.LightningModule):
    def __init__(self, lr, weight_decay, pct, tf_drop, multi_scale_size, use_sub_info=False, subject_list=[1, 2, 5, 7]):
        super().__init__()
        self.multi_scale_size = multi_scale_size
        self.pretrained = pct
        self.decoder = TransfomerDecoder(dim=768, num_output_tokens=77, nhead=1, num_layers=2, dropout=tf_drop,
                                         dim_feedforward=1024)
        self.loss_1_weight = 1
        self.loss_2_weight = 500
        self.use_sub_info = use_sub_info
        self.subject_list = subject_list
        self.lr = lr
        self.weight_decay = weight_decay
        self.tf_drop = tf_drop
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(tf_drop),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(tf_drop),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 768),
        )
        self.loss_fn = ElementwiseComparisonLoss()
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def forward(self, fmri_dict, subj_id):
        features = []
        if self.use_sub_info:
            for num_windows, window_size in self.multi_scale_size:
                key_input = f"{num_windows}_{window_size}"
                x = fmri_dict[key_input]  # [B, N, D]
                B = x.shape[0]

                unique_subj_ids = torch.unique(subj_id)
                group_outputs = [None] * B

                for s_id in unique_subj_ids:
                    s_id = s_id.item()
                    key_model = f"sub{s_id}_{num_windows}_{window_size}"
                    indices = (subj_id == s_id).nonzero(as_tuple=False).squeeze(1)
                    x_group = x[indices]  # [n_i, N, D]
                    out_group = self.pretrained.multi_scales[key_model](x_group)
                    for i, idx in enumerate(indices):
                        group_outputs[idx] = out_group[i:i + 1]

                feature_tensor = torch.cat(group_outputs, dim=0)
                features.append(feature_tensor)
        else:
            for num_windows, window_size in self.multi_scale_size:
                key = f"{num_windows}_{window_size}"
                fmri_input = fmri_dict[key]  # [B, N, D]
                feature = self.pretrained.multi_scales[key](fmri_input)
                features.append(feature)
        multi_scale_feature = torch.cat(features, dim=1)
        batch_size, _, _ = multi_scale_feature.shape

        cls_token = torch.zeros(batch_size, 1, 768, device=self.device)
        multi_scale_feature = torch.cat((multi_scale_feature, cls_token), dim=1)
        x = self.pretrained.pt_last(multi_scale_feature)
        x_1 = self.decoder(x)
        x_2 = self.projector(x_1)
        return x_1, x_2

    def training_step(self, batch, batch_idx):

        fmri_dict, text, subject_id = batch
        label = text
        label = label.squeeze()
        label = label.squeeze()
        x_1, x_2 = self(fmri_dict, subject_id)

        loss_1 = cos_loss(x_1, label)
        loss = self.loss_fn(x_2, label)
        mse_cos_loss = loss['mse_cos_loss']
        mse_loss = loss['mse_loss']

        loss = self.loss_2_weight * mse_loss + (self.loss_1_weight * loss_1)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("mse_cos_loss", mse_cos_loss, prog_bar=True, sync_dist=True)

        self.log("mse_loss", mse_loss, prog_bar=True, sync_dist=True)

        torch.cuda.synchronize()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        fmri_dict, text, subject_id = batch
        label = text
        label = label.squeeze()
        x_1, x_2 = self(fmri_dict, subject_id)
        loss_1 = cos_loss(x_1, label)
        loss = self.loss_fn(x_2, label)
        mse_loss = loss['mse_loss']
        mse_cos_loss = loss['mse_cos_loss']
        loss = self.loss_2_weight * mse_loss + (self.loss_1_weight * loss_1)

        self.log("val_mse_cos_loss", mse_cos_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "pos_embed", "scale_embed", "norm", "ln", "cls_token", "scale_w"]):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.lr)

        total_steps = self.trainer.num_training_batches * self.trainer.max_epochs
        print("max_epochs:", self.trainer.max_epochs)
        print("estimated_stepping_batches:", self.trainer.estimated_stepping_batches)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                final_div_factor=25,
                pct_start=2 / self.trainer.max_epochs,
                anneal_strategy='cos'
            ),
            "interval": "step",  # 注意，OneCycleLR一定是按step
            "frequency": 1,
            "name": "one_cycle_lr"
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
