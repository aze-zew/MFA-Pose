
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]

class EfficientLKA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 5),
                      padding=(0, 4), dilation=(1, 2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (5, 1),
                      padding=(4, 0), dilation=(2, 1), groups=in_channels),
            nn.GroupNorm(max(1, in_channels // 8), in_channels),
            nn.SiLU()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(in_channels, in_channels, kernel_size=1),

            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.pw_proj = nn.Conv2d(in_channels, out_channels, 1)
        self._init_weights()

    def _init_weights(self):

        nn.init.normal_(self.pw_proj.weight, std=0.01)
        if self.pw_proj.bias is not None:
            nn.init.constant_(self.pw_proj.bias, 0)
        conv_layer = self.channel_att[1]
        if isinstance(conv_layer, nn.Conv2d):
            nn.init.normal_(conv_layer.weight, std=0.01)
            if conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0)

        for layer in self.spatial_att:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # 深度特征提取
        x_dw = self.dw_conv(x)
        # 通道注意力（保持维度）
        ca = self.channel_att(x_dw)

        # 空间注意力（保持维度）
        sa = self.spatial_att(x_dw)
        # 特征融合（与原始结构相同的广播机制）
        x_att = identity * (ca + sa)
        # 最终投影（确保输出通道与原始一致）
        return self.pw_proj(x_att)

class PureDSConvLKA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # ----------------- 深度卷积部分 -----------------
        # 第一层：3x3深度卷积（保持空间尺寸）
        self.dw_conv1 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )
        # 第二层：3x3扩张深度卷积（dilation=2）
        self.dw_dilated = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3,
            padding=2,  # (3-1)*2//2=2
            dilation=2,
            groups=in_channels
        )
        # 第三层：3x3深度卷积
        self.dw_conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )
        # ----------------- 通道注意力（轻量化设计） -----------------
        self.gate = nn.Sequential(
            # 5x5深度卷积（空间聚合）
            nn.Conv2d(in_channels, in_channels, 5,
                      padding=2, groups=in_channels),
            # 1x1逐点卷积（可分组）
            nn.Conv2d(in_channels, in_channels, 1, groups=8),
            nn.Sigmoid()
        )
        self.pw_proj = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1  # 确保输出通道与原模型一致
        )
        for layer in [self.dw_conv1, self.dw_dilated, self.dw_conv2, self.gate[0], self.gate[1]]:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)  # 与原Conv2d初始化一致
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.pw_proj.weight, std=0.01)
        nn.init.constant_(self.pw_proj.bias, 0)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # 深度卷积路径（9x9感受野）
        x = self.dw_conv1(x)
        x = self.dw_dilated(x)
        x = self.dw_conv2(x)
        # 注意力门控
        attn = self.gate(x)
        x = identity * attn
        # 通道投影
        x = self.pw_proj(x)
        return x


@MODELS.register_module()
class RTMCCHead(BaseHead):
    def __init__(
            self,
            in_channels: Union[int, Sequence[int]],
            out_channels: int,
            input_size: Tuple[int, int],
            in_featuremap_size: Tuple[int, int],
            simcc_split_ratio: float = 2.0,
            final_layer_kernel_size: int = 1,
            gau_cfg: ConfigType = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
            decoder: OptConfigType = None,
            init_cfg: OptConfigType = None,
            use_efficient_lka: bool = False  # 新增开关控制模块选择
        ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')
            
        # 通过use_efficient_lka开关选择模块类型
        lka_class = EfficientLKA if use_efficient_lka else PureDSConvLKA
        self.final_layer = lka_class(
            in_channels=in_channels,
            out_channels=out_channels,
       )

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:

        input_feats = feats[-1]
        feats = self.final_layer(input_feats)  # -> B, K, H, W
        
        #feats = torch.flatten(feats, 2)
        feats = feats.permute(0, 2, 3, 1).contiguous()
        feats = feats.view(feats.size(0), -1, feats.size(-1))
        feats = feats.transpose(1, 2)
        
        feats = self.mlp(feats)  # -> B, K, hidden

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y

    def predict(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            test_cfg: OptConfigType = {},
    ) -> InstanceList:

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
            dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
            dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
