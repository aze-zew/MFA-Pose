
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
import math
from mmengine import MODELS
class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应计算卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = max(t if t % 2 else t + 1, 3)  # 确保≥3且为奇数

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return identity * y
@MODELS.register_module()
class Bi_A(BaseModule):
    def __init__(self,
                 in_channels,  # 输入各层的通道数（如 [256, 512, 1024]）
                 out_channels=1024,  # 输出统一通道数
                 num_layers=2,  # BiFPN堆叠层数
                 epsilon=1e-4,  # 防止除零的小量
                 attention=True,  # 是否启用注意力

                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.attention = attention

        # 输入投影层
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels:
            print(f"Creating lateral conv: {ch} -> {out_channels}")  # 调试输出
            self.lateral_convs.append(
                ConvModule(
                    ch,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=None
                )
            )

        # 权重参数
        self.weights = nn.ParameterList()
        for _ in range(len(in_channels) - 1):
            self.weights.append(nn.Parameter(torch.ones(2, dtype=torch.float32)))

        # 构建BiFPN层
        self.bifpn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'top_down': nn.ModuleList(),
                'bottom_up': nn.ModuleList(),
                'attention': nn.ModuleList()  # 新增注意力模块
            })

            # 为每个特征层级添加卷积块和注意力
            for i in range(len(in_channels) - 1):
                layer['top_down'].append(self._build_ds_conv_block(out_channels))
                layer['bottom_up'].append(self._build_ds_conv_block(out_channels))

            # 为每个输出层级添加注意力模块
            if attention:
                for i in range(len(in_channels)):
                    layer['attention'].append(
                        EfficientChannelAttention(out_channels)
                    )
            else:
                # 如果不启用注意力，添加空模块
                for _ in range(len(in_channels)):
                    layer['attention'].append(nn.Identity())

            self.bifpn_layers.append(layer)

    def _build_ds_conv_block(self, channels):
        """构建标准卷积块"""
        return DepthwiseSeparableConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dw_norm_cfg=dict(type='BN'),
            dw_act_cfg=dict(type='SiLU'),
            pw_norm_cfg=dict(type='BN'),
            pw_act_cfg=dict(type='SiLU')
        )

    def forward(self, inputs):
        # 初始投影
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        # 逐层处理
        for bifpn_layer in self.bifpn_layers:
            # ----------------- Top-Down 路径 -----------------
            top_down_path = [feats[-1]]  # 从最高级特征开始
            # 从P5->P4->P3
            for i in range(len(feats) - 2, -1, -1):
                # 融合权重
                w = F.relu(self.weights[i])
                w_sum = w.sum() + self.epsilon
                # 上采样
                up_feat = F.interpolate(
                    top_down_path[-1],
                    size=feats[i].shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
                # 特征融合
                fused_feat = (w[0] * feats[i] + w[1] * up_feat) / w_sum
                # 卷积处理
                fused_feat = bifpn_layer['top_down'][i](fused_feat)
                top_down_path.append(fused_feat)
            top_down_path = list(reversed(top_down_path))  # 反转顺序 [P3, P4, P5]
            # ----------------- Bottom-Up 路径 -----------------
            bottom_up_path = [top_down_path[0]]  # 从最低级特征开始
            for i in range(len(top_down_path) - 1):
                # 融合权重
                w = F.relu(self.weights[i])
                w_sum = w.sum() + self.epsilon
                # 下采样
                target_size = top_down_path[i + 1].shape[2:]
                down_feat = F.interpolate(
                    bottom_up_path[-1],
                    size=target_size,
                    mode='bilinear',
                    align_corners=True
                )
                # 特征融合
                fused_feat = (w[0] * down_feat + w[1] * top_down_path[i + 1]) / w_sum
                # 卷积处理
                fused_feat = bifpn_layer['bottom_up'][i](fused_feat)
                bottom_up_path.append(fused_feat)
            # 应用注意力机制
            feats = []
            for i, feat in enumerate(bottom_up_path):
                # 使用对应层级的注意力模块
                feats.append(bifpn_layer['attention'][i](feat))
        return feats
