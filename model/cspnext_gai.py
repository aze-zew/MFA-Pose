
import math
from typing import Optional, Sequence, Tuple
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmcv.ops import DeformConv2dPack

from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType
from ..utils import CSPLayer_gai
from .csp_darknet import SPPBottleneck
from ..necks.Bi_A import Bi_A


class DCNInputAdapter(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.dcn = DeformConv2dPack(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            deform_groups=1, # 添加分组参数
            im2col_step= 64  # 显式设置更小的步长
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 动态设置 im2col_step 为当前 batch size
        batch_size = x.size(0)
        self.dcn.im2col_step = batch_size  # 确保整除
        # 直接使用 DCN，偏移量由内部生成
        x_dcn = self.dcn(x)
        x_res = self.conv(x)
        x = x_res + x_dcn  # 残差连接
        return self.act(self.bn(x))

@MODELS.register_module()
class CSPNeXt_gai(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        
        # DCA注意力参数 (使用新前缀避免冲突)
        dca_attention: bool = False,  # 是否启用DCA
        dca_reduction: int = 16,  # 缩减比例
        #dca_kernel_sizes: list = [3, 5],  # 卷积核尺寸
        dca_groups: int = 4,  # 分组数
        
        use_input_dcn: bool = True,  # DCN开关

        use_bifpn: bool = False,  # 新增：是否启用BiFPN
        bifpn_out_channels: int = 1024,  # BiFPN输出通道
        bifpn_num_layers: int = 2,  # BiFPN堆叠层数

        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        init_cfg: Optional[ConfigType] = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.use_input_dcn = use_input_dcn
        # 初始化DCN输入适配器
        if self.use_input_dcn:
            self.input_adapter = DCNInputAdapter(3, 64)  # 输出通道64匹配Stem输入
        else:
            self.input_adapter = nn.Identity()

        # 计算实际输出通道数（考虑widen_factor）
        self.bifpn_input_channels = [
            int(ch * widen_factor)
            for _, ch, _, _, _ in arch_setting[len(arch_setting) - len(out_indices):]
        ]

        # 计算BiFPN输入通道
        self._calculate_bifpn_channels(arch_setting, widen_factor)
        self.use_bifpn = use_bifpn
        if self.use_bifpn:
            self.bifpn = Bi_A(
                in_channels=self.bifpn_input_channels,
                out_channels=bifpn_out_channels,
                num_layers=bifpn_num_layers,
                attention=True,  # 传入注意力开关
            )
            # 更新输出通道数（头部需要适配）
            self.out_channels = [bifpn_out_channels] * len(self.out_indices)

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem = nn.Sequential(
            ConvModule(
                64,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer_gai(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,

                dca_attention=dca_attention,
                dca_reduction=dca_reduction,
                #dca_kernel_sizes=dca_kernel_sizes,
                dca_groups=dca_groups,

                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _calculate_bifpn_channels(self, arch_setting, widen_factor):
        """根据out_indices计算各stage的输出通道"""
        self.bifpn_input_channels = []
        for idx in self.out_indices:
            if idx == 0:  # stem的输出
                ch = int(arch_setting[0][0] * widen_factor)
            else:  # stage的输出（索引从1开始）
                stage_idx = idx - 1
                ch = int(arch_setting[stage_idx][1] * widen_factor)
            self.bifpn_input_channels.append(ch)

        
        print(f"BiFPN input channels: {self.bifpn_input_channels}")  # 调试输出


    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        x = self.input_adapter(x)  # 先通过DCN模块
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:

                #print(f"Layer {layer_name} output shape: {x.shape}")  # 打印尺寸

                outs.append(x)

        if self.use_bifpn:
            outs = self.bifpn(outs)

        return tuple(outs)