# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn

import lightning.pytorch as pl

import torch
import torch.nn.functional as F



class local_Transformer(pl.LightningModule):
    def __init__(self, channels):
        super(local_Transformer, self).__init__()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        self.sa1 = SA_Layer(channels)


    def forward(self, x):
        batch_size, _, N = x.size()

        # B, D, N
        x = self.sa1(x)
        x = self.gelu(self.bn2(self.conv2(x)))

        return x


class TransformerWithCLS(pl.LightningModule):
    def __init__(self, in_channels, out_channels, args, channels):
        super(TransformerWithCLS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 嵌入层将输入维度转换为模型期望的输出维度
        self.embedding = nn.Linear(in_channels, out_channels)
        self.local_transformer = local_Transformer(channels=out_channels)


    def forward(self, x):
        b, n, s, d = x.size()
        x = x.reshape(-1, s, d)

        x = self.embedding(x)

        x = x.permute(0, 2, 1)
        x = self.local_transformer(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=1)

        out = x.view(b, n, -1)

        return out


class Multi_scale(pl.LightningModule):
    def __init__(self, n_windows, window_size, drop_num,use_xyz):
        super(Multi_scale, self).__init__()
        self.n_windows = n_windows
        self.window_size = window_size
        self.use_xyz=use_xyz
        self.gather_local = Local_op(in_channels=window_size, out_channels=256, drop_num=drop_num)
        if n_windows == 512:
            self.conv_downsample = nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
            )
        elif n_windows == 1024:
            self.conv_downsample = nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
            )
        elif n_windows == 256:
            self.conv_downsample = nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.SiLU(),
            )
        if use_xyz:
            self.pos_encoder = FeatureWithRelativePosition(n_point=n_windows, feature_dim=256)
            xyz_l_roi = np.load(f"/root/data-tmp/data/groups/group_{n_windows}_{window_size}/xyz_l_roi.npy")
            xyz_r_roi = np.load(f"/root/data-tmp/data/groups/group_{n_windows}_{window_size}/xyz_r_roi.npy")
            self.xyz = np.concatenate([xyz_l_roi, xyz_r_roi], axis=0)
            self.xyz = torch.tensor(self.xyz, dtype=torch.float32)
        self.layer_norm = nn.LayerNorm(256)
        self.scale_embed = torch.nn.Parameter(torch.randn(256))
        self.projector_256_768 = nn.Sequential(nn.Conv1d(256, 768, kernel_size=3, bias=False, padding=1),
                                               nn.BatchNorm1d(768),
                                               nn.SiLU(),
                                               nn.Dropout(drop_num))

    def forward(self, x):
        batch_size, _, _, _ = x.size()

        feature = self.gather_local(x)
        if self.use_xyz:
            xyz = self.xyz.to(self.device)
            xyz = xyz.unsqueeze(0).repeat(batch_size, 1, 1)
            pos_feature = self.pos_encoder(xyz)
            feature = feature + pos_feature
        feature = feature + self.layer_norm(self.scale_embed.view(1, 1, -1)) * 0.1
        feature = feature.permute(0, 2, 1).contiguous()
        feature = self.conv_downsample(feature)
        feature = self.projector_256_768(feature)
        feature = feature.permute(0, 2, 1).contiguous()
        return feature


class Local_op(pl.LightningModule):
    def __init__(self, in_channels, out_channels, drop_num):
        super(Local_op, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(drop_num)

        )

    def forward(self, x):
        x = x[:, :, :, 3]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)

        return x

class CosineSimilarityLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        # 计算余弦相似度
        similarity = nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)  # 假设输入是两个形状为 (batch_size, 256, 768) 的张量

        # 将余弦相似度映射到 [0, 1] 范围内
        similarity = (similarity + 1.0) / 2.0

        # 计算损失（可以根据需求选择不同的损失函数）
        loss = 1.0 - similarity  # 例如，使用 1 减去相似性作为损失

        return loss.mean()

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


class FeatureWithRelativePosition(nn.Module):
    def __init__(self, n_point, feature_dim):
        super(FeatureWithRelativePosition, self).__init__()
        self.feature_dim = feature_dim

        self.distance_linear = nn.Linear(n_point, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.silu = nn.SiLU()

    def compute_distance_matrix(self, positions):
        """
        计算相对距离矩阵
        参数:
            positions: 每个点的位置坐标，形状为 (bs, num_points, 3)
        返回:
            距离矩阵，形状为 (bs, num_points, num_points)
        """
        # positions 形状为 (bs, num_points, 3)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # 计算两两点之间的坐标差，形状为 (bs, num_points, num_points, 3)
        distance_matrix = torch.sqrt((diff ** 2).sum(-1))  # 计算欧氏距离，形状为 (bs, num_points, num_points)
        return distance_matrix

    def forward(self, positions):
        """
        参数:
            features: 输入特征，形状为 (bs, 1024, 64)
            positions: 每个点的位置坐标，形状为 (bs, 1024, 3)
        返回:
            包含相对位置信息的特征，形状为 (bs, 1024, 64)
        """
        # 计算距离矩阵
        distance_matrix = self.compute_distance_matrix(positions)  # (bs, 1024, 1024)

        # 将距离矩阵通过 Linear 层处理，得到形状 (bs, 1024, 64)
        distance_features = self.distance_linear(distance_matrix)  # (bs, 1024, 64)
        distance_features = self.layer_norm(distance_features)
        distance_features = self.silu(distance_features)

        return distance_features


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_feat_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_feat_dim, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.d_model = d_model

    def forward(self, src):
        output = self.transformer_encoder(src)
        # output = output.permute(1, 0, 2)
        # output = self.output_embedding(output)
        return output





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


def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2) / 2
    return loss


class Pct(pl.LightningModule):
    def __init__(self, tf_drop, lr, weight_decay, multi_scale_size, use_sub_info=False, subject_list=[1, 2, 5, 7],use_xyz=True):
        super(Pct, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_sub_info = use_sub_info
        self.subject_list = subject_list
        # self.sub_num = sub_num
        self.weight_decay = weight_decay
        self.tf_drop = tf_drop
        self.multi_scale_size = multi_scale_size
        self.loss_1_weight = 30
        self.loss_2_weight = 1
        self.use_xyz=use_xyz
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
        self.multi_scales = nn.ModuleDict()
        if use_sub_info:
            for subj_id in self.subject_list:
                for num_windows, window_size in self.multi_scale_size:
                    key = f"sub{subj_id}_{num_windows}_{window_size}"
                    self.multi_scales[key] = Multi_scale(n_windows=num_windows, window_size=window_size,
                                                         drop_num=tf_drop,use_xyz=use_xyz)
        else:
            for num_windows, window_size in self.multi_scale_size:
                key = f"{num_windows}_{window_size}"
                self.multi_scales[key] = Multi_scale(n_windows=num_windows, window_size=window_size, drop_num=tf_drop,use_xyz=use_xyz)

        self.pt_last = TransformerEncoderModel(input_feat_dim=768, d_model=1024, nhead=1, dim_feedforward=1024,
                                               num_encoder_layers=2,
                                               dropout=tf_drop)
        self.decoder = TransfomerDecoder(dim=768, num_output_tokens=257, nhead=1, num_layers=2, dropout=tf_drop,
                                         dim_feedforward=1024)


    def forward(self, fmri_dict, subj_id):

        features = []
        if self.use_sub_info:
            for num_windows, window_size in self.multi_scale_size:
                key_input = f"{num_windows}_{window_size}"
                x = fmri_dict[key_input]  # [B, N, D]
                B = x.shape[0]

                # 每个受试者一个 mask
                unique_subj_ids = torch.unique(subj_id)
                group_outputs = [None] * B  # 用于恢复顺序

                for s_id in unique_subj_ids:
                    s_id = s_id.item()
                    key_model = f"sub{s_id}_{num_windows}_{window_size}"

                    # 找出当前受试者对应的样本位置
                    indices = (subj_id == s_id).nonzero(as_tuple=False).squeeze(1)  # shape: [n_i]
                    x_group = x[indices]  # [n_i, N, D]
                    out_group = self.multi_scales[key_model](x_group)  # [n_i, N, D']

                    # 保存到 group_outputs 的正确位置
                    for i, idx in enumerate(indices):
                        group_outputs[idx] = out_group[i:i + 1]  # 保留 batch dim

                # 重新拼接为 [B, N, D']
                feature_tensor = torch.cat(group_outputs, dim=0)
                features.append(feature_tensor)
        else:
            for num_windows, window_size in self.multi_scale_size:
                key = f"{num_windows}_{window_size}"
                fmri_input = fmri_dict[key]  # [B, N, D]
                feature = self.multi_scales[key](fmri_input)
                features.append(feature)
        # multi_scale_feature = torch.stack(features, dim=0).sum(dim=0)
        multi_scale_feature = torch.cat(features, dim=1)
        batch_size, _, _ = multi_scale_feature.shape

        cls_token = torch.zeros(batch_size, 1, 768, device=self.device)
        multi_scale_feature = torch.cat((multi_scale_feature, cls_token), dim=1)
        x_1 = self.pt_last(multi_scale_feature)  # (2,8192,4096)
        x_1 = self.decoder(x_1)
        x_2 = self.projector(x_1)
        return x_1, x_2

    def training_step(self, batch, batch_idx):
        fmri_dict, img, subject_id = batch

        label = img

        label = label.squeeze()
        label = label.squeeze()
        # data = data.permute(0, 2, 1).contiguous()
        x_1, x_2 = self(fmri_dict, subject_id)


        clip_voxels_norm = nn.functional.normalize(x_1.flatten(1), dim=-1)
        clip_target_norm = nn.functional.normalize(label.flatten(1), dim=-1)
        loss_1 = soft_clip_loss(
            clip_voxels_norm,
            clip_target_norm,
            temp=0.006)

        loss = self.loss_fn(x_2, label)
        mse_cos_loss = loss['mse_cos_loss']
        mse_loss = loss['mse_loss']

        loss = self.loss_2_weight * mse_loss + (self.loss_1_weight * loss_1)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("mse_cos_loss", mse_cos_loss, prog_bar=True, sync_dist=True)
        self.log("mse_loss", mse_loss, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        fmri_dict, img, subject_id = batch
        label = img
        label = label.squeeze()
        x_1, x_2 = self(fmri_dict, subject_id)
        clip_voxels_norm = nn.functional.normalize(x_1.flatten(1), dim=-1)
        clip_target_norm = nn.functional.normalize(label.flatten(1), dim=-1)
        loss_1 = soft_clip_loss(
            clip_voxels_norm,
            clip_target_norm,
            temp=0.006)


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
                print("88888888888888800000000000000:", name)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.lr)

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


class Point_Transformer_Last(pl.LightningModule):
    def __init__(self, args, channels=1024):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = x1 + x2 + x3 + x4
        return x


class SA_Layer(pl.LightningModule):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
