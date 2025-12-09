
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from torch import Tensor
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from mmpose.utils.typing import ConfigType, OptConfigType, OptMultiConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

class EDDA(nn.Module):
    def __init__(self, in_channels, reduction=16, expand_ratio=0.25, groups=4):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        mid_channels = max(8, in_channels // reduction)

        # 轻量双轴特征提取（共享权重）
        self.axis_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),  # 深度可分离卷积
            nn.Conv2d(in_channels, 4, 1),  # 通道扩展
            nn.GELU(),#Mish
            nn.Conv2d(4, 1, 1)  # 通道压缩
        )

        # 动态核融合技术
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, 3)  # 生成3个核权重
        )

        # 多尺度空洞卷积（保持双轴结构）
        self.dilated_x_convs = nn.ModuleList([
            nn.Conv1d(1, 1, 3, padding=d, dilation=d, bias=False)
            for d in (1, 2, 3)
        ])
        self.dilated_y_convs = nn.ModuleList([
            nn.Conv1d(1, 1, 3, padding=d, dilation=d, bias=False)
            for d in (1, 2, 3)
        ])

        # 高效位置精炼（5×5替代7×7）
        self.position_refine = nn.Sequential(
            nn.Conv2d(2, 4, 5, padding=2),  # 减半通道
            nn.GELU(),
            nn.Conv2d(4, 1, 5, padding=2),
            nn.Sigmoid()
        )

        # 分组通道增强
        self.group_channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // groups, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // groups, in_channels, 1),
            nn.Sigmoid()
        )

        # 深度可分离特征融合
        self.depthwise_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # 1. 双轴特征提取
        x_pool = x.mean(dim=2, keepdim=True)  # [b, c, 1, w]
        y_pool = x.mean(dim=3, keepdim=True)  # [b, c, h, 1]

        # 共享权重处理
        x_pool = self.axis_conv(x_pool)  # [b, 1, 1, w]
        y_pool = self.axis_conv(y_pool)  # [b, 1, h, 1]

        # 2. 动态核融合
        kernel_weights = F.softmax(self.kernel_gen(x), dim=1)  # [b, 3]

        # 3. 多尺度处理（保持双轴分离）
        # X轴处理
        x_feats = []
        for i, conv in enumerate(self.dilated_x_convs):
            feat = x_pool.squeeze(2)  # [b, c, w]
            feat = conv(feat).unsqueeze(2)  # [b, c, 1, w]
            # 应用动态权重
            weighted_feat = feat * kernel_weights[:, i].view(b, 1, 1, 1)
            x_feats.append(weighted_feat)

        # Y轴处理
        y_feats = []
        for i, conv in enumerate(self.dilated_y_convs):
            feat = y_pool.squeeze(3)  # [b, c, h]
            feat = conv(feat).unsqueeze(3)  # [b, c, h, 1]
            # 应用动态权重
            weighted_feat = feat * kernel_weights[:, i].view(b, 1, 1, 1)
            y_feats.append(weighted_feat)

        # 4. 特征融合
        x_feat = sum(x_feats).expand(-1, -1, h, -1)  # [b, c, h, w]
        y_feat = sum(y_feats).expand(-1, -1, -1, w)  # [b, c, h, w]

        # 位置精炼（双轴特征融合）
        spatial_att = self.position_refine(torch.cat([
            x_feat.mean(dim=1, keepdim=True),  # 空间特征
            y_feat.mean(dim=1, keepdim=True)
        ], dim=1))  # [b, 1, h, w]

        # 5. 分组通道注意力
        channel_att = self.group_channel_att(x)  # [b, c, 1, 1]

        # 6. 特征融合与增强
        out = identity * (1 + spatial_att) * (1 + channel_att)  # 残差增强
        out = self.depthwise_fusion(out)

        return out


class DarknetBottleneck(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out

class CSPNeXtBlock(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer_gai(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 
                 dca_attention: bool = False,  # 启用DCA注意力
                 dca_reduction: int = 16,  # 缩减比例
                 #dca_kernel_sizes: list = [3, 5],  # 卷积核尺寸
                 dca_groups: int = 4,  # 分组数
                 
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)

        #self.channel_attention = channel_attention

        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

        self.attention = None
        if dca_attention:
            self.attention = EDDA(
                2 * mid_channels,
                reduction=dca_reduction,
                #kernel_sizes=dca_kernel_sizes,
                groups=dca_groups
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.attention is not None:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)