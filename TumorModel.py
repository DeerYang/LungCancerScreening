import math
import numpy as np

import torch
from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import random

# 对输入进行3D数据增强
def augment3d(inp):
    # 初始化变换矩阵
    transform_t = torch.eye(4, dtype=torch.float32)
    for i in range(3):
        if True: #'flip' in augmentation_dict:
            # 以0.5的概率进行翻转
            if random.random() > 0.5:
                transform_t[i,i] *= -1
        if True: #'offset' in augmentation_dict:
            # 定义偏移量
            offset_float = 0.1
            # 生成随机偏移值
            random_float = (random.random() * 2 - 1)
            transform_t[3,i] = offset_float * random_float
    if True:
        # 生成随机旋转角度
        angle_rad = random.random() * np.pi * 2
        s = np.sin(angle_rad)
        c = np.cos(angle_rad)

        # 构建旋转矩阵
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        # 合并变换矩阵
        transform_t @= rotation_t
    # 生成仿射网格
    affine_t = torch.nn.functional.affine_grid(
            transform_t[:3].unsqueeze(0).expand(inp.size(0), -1, -1).cuda(),
            inp.shape,
            align_corners=False,
        )

    # 进行网格采样，得到增强后的数据
    augmented_chunk = torch.nn.functional.grid_sample(
            inp,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )
    return augmented_chunk

# 定义Luna模型类，继承自torch.nn.Module
class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        # 定义尾部批量归一化层
        self.tail_batchnorm = nn.BatchNorm3d(1)

        # 定义四个卷积块
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # 定义全连接层
        self.head_linear = nn.Linear(1152, 2)
        # 定义softmax激活函数
        self.head_activation = nn.Softmax(dim=1)

        # 初始化模型参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                # 使用Kaiming初始化方法初始化权重
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    # 计算扇入和扇出
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    # 计算边界值
                    bound = 1 / math.sqrt(fan_out)
                    # 初始化偏置
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        # 进行批量归一化
        bn_output = self.tail_batchnorm(input_batch)

        # 依次通过四个卷积块
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        # 将卷积输出展平
        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        # 通过全连接层
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_activation(linear_output)

# 定义Luna块类，继承自torch.nn.Module
class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        # 定义第一个卷积层
        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        # 定义第一个ReLU激活函数
        self.relu1 = nn.ReLU(inplace=True)
        # 定义第二个卷积层
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        # 定义第二个ReLU激活函数
        self.relu2 = nn.ReLU(inplace=True)

        # 定义最大池化层
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        # 通过第一个卷积层
        block_out = self.conv1(input_batch)
        # 通过第一个ReLU激活函数
        block_out = self.relu1(block_out)
        # 通过第二个卷积层
        block_out = self.conv2(block_out)
        # 通过第二个ReLU激活函数
        block_out = self.relu2(block_out)

        # 通过最大池化层
        return self.maxpool(block_out)