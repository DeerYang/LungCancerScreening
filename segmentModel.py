import math
import random

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging
from util.unet import UNet

# 初始化日志记录器
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
# 设置日志级别为DEBUG，用于调试信息
log.setLevel(logging.DEBUG)


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        # 调用父类的构造函数
        super().__init__()
        # 输入数据的批量归一化层
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        # 初始化UNet模型
        self.unet = UNet(**kwargs)
        # 最终的Sigmoid激活层
        self.final = nn.Sigmoid()
        # 初始化模型参数
        self._init_weights()

    def _init_weights(self):
        # 定义需要初始化参数的层类型集合
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                # 使用Kaiming初始化方法初始化权重
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
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
        # 对输入数据进行批量归一化处理
        bn_output = self.input_batchnorm(input_batch)
        # 通过UNet模型进行前向传播
        un_output = self.unet(bn_output)
        # 通过Sigmoid激活层得到最终输出
        fn_output = self.final(un_output)
        return fn_output


class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        # 调用父类的构造函数
        super().__init__()

        # 翻转增强标志
        self.flip = flip
        # 偏移增强参数
        self.offset = offset
        # 缩放增强参数
        self.scale = scale
        # 旋转增强标志
        self.rotate = rotate
        # 噪声增强参数
        self.noise = noise

    def forward(self, input_g, label_g):
        # 构建2D变换矩阵
        transform_t = self._build2dTransformMatrix()
        # 扩展变换矩阵以适应批量大小
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        # 将变换矩阵移动到指定设备上
        transform_t = transform_t.to(input_g.device, torch.float32)
        # 生成仿射网格
        affine_t = F.affine_grid(transform_t[:, :2],
                                 input_g.size(), align_corners=False)

        # 对输入数据进行仿射变换
        augmented_input_g = F.grid_sample(input_g,
                                          affine_t, padding_mode='border',
                                          align_corners=False)
        # 对标签数据进行仿射变换
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                                          affine_t, padding_mode='border',
                                          align_corners=False)

        if self.noise:
            # 生成随机噪声
            noise_t = torch.randn_like(augmented_input_g)
            # 调整噪声强度
            noise_t *= self.noise

            # 将噪声添加到输入数据中
            augmented_input_g += noise_t

        # 返回增强后的输入数据和标签数据
        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        # 初始化2D变换矩阵为单位矩阵
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    # 随机进行翻转操作
                    transform_t[i, i] *= -1

            if self.offset:
                # 获取偏移参数
                offset_float = self.offset
                # 生成随机偏移值
                random_float = (random.random() * 2 - 1)
                # 更新变换矩阵的偏移量
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                # 获取缩放参数
                scale_float = self.scale
                # 生成随机缩放值
                random_float = (random.random() * 2 - 1)
                # 更新变换矩阵的缩放比例
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            # 生成随机旋转角度
            angle_rad = random.random() * math.pi * 2
            # 计算正弦值
            s = math.sin(angle_rad)
            # 计算余弦值
            c = math.cos(angle_rad)

            # 构建旋转矩阵
            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            # 将旋转矩阵与变换矩阵相乘
            transform_t @= rotation_t

        return transform_t