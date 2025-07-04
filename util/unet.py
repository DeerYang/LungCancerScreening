# From https://github.com/jvanvugt/pytorch-unet
# https://raw.githubusercontent.com/jvanvugt/pytorch-unet/master/unet.py

# MIT License
#
# Copyright (c) 2018 Joris
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        # 调用父类的构造函数
        super(UNet, self).__init__()
        # 确保上采样模式为'upconv'或'upsample'
        assert up_mode in ('upconv', 'upsample')
        # 记录是否使用填充
        self.padding = padding
        # 记录网络深度
        self.depth = depth
        # 记录上一层的通道数
        prev_channels = in_channels
        # 定义下采样路径的模块列表
        self.down_path = nn.ModuleList()
        for i in range(depth):
            # 添加下采样块
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            # 更新上一层的通道数
            prev_channels = 2**(wf+i)

        # 定义上采样路径的模块列表
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            # 添加上采样块
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            # 更新上一层的通道数
            prev_channels = 2**(wf+i)

        # 定义最后一层卷积层
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # 存储下采样过程中的特征图
        blocks = []
        for i, down in enumerate(self.down_path):
            # 通过下采样块进行前向传播
            x = down(x)
            if i != len(self.down_path)-1:
                # 保存特征图
                blocks.append(x)
                # 进行平均池化下采样
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            # 通过上采样块进行前向传播
            x = up(x, blocks[-i-1])

        # 通过最后一层卷积层得到输出
        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        # 调用父类的构造函数
        super(UNetConvBlock, self).__init__()
        # 定义卷积块的模块列表
        block = []

        # 添加第一个卷积层
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        # 添加ReLU激活层
        block.append(nn.ReLU())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            # 如果使用批量归一化，添加批量归一化层
            block.append(nn.BatchNorm2d(out_size))

        # 添加第二个卷积层
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        # 添加ReLU激活层
        block.append(nn.ReLU())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            # 如果使用批量归一化，添加批量归一化层
            block.append(nn.BatchNorm2d(out_size))

        # 将模块列表组合成顺序模块
        self.block = nn.Sequential(*block)

    def forward(self, x):
        # 通过卷积块进行前向传播
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        # 调用父类的构造函数
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            # 如果使用转置卷积上采样，定义转置卷积层
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            # 如果使用双线性上采样，定义上采样层和卷积层
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        # 定义卷积块
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        # 获取输入层的高度和宽度
        _, _, layer_height, layer_width = layer.size()
        # 计算高度的差值
        diff_y = (layer_height - target_size[0]) // 2
        # 计算宽度的差值
        diff_x = (layer_width - target_size[1]) // 2
        # 进行中心裁剪
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        # 进行上采样
        up = self.up(x)
        # 对跳跃连接的特征图进行中心裁剪
        crop1 = self.center_crop(bridge, up.shape[2:])
        # 将上采样结果和裁剪后的特征图进行拼接
        out = torch.cat([up, crop1], 1)
        # 通过卷积块进行前向传播
        out = self.conv_block(out)

        return out