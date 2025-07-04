import math
from torch import nn as nn
from util.logconf import logging

# 创建日志记录器，用于记录模型相关信息
log = logging.getLogger(__name__)
# 设置日志级别为 WARN，可记录警告信息
log.setLevel(logging.WARN)
# 再次设置日志级别为 INFO，以记录更多信息
log.setLevel(logging.INFO)

class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        # 调用父类的构造函数
        super().__init__()

        # 定义批量归一化层，输入通道数为 1
        self.tail_batchnorm = nn.BatchNorm3d(1)

        # 定义四个 LunaBlock 块，通道数依次递增
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        # 定义全连接层，输入特征数为 1152，输出特征数为 2
        self.head_linear = nn.Linear(1152, 2)
        # 定义 Softmax 层，用于将输出转换为概率分布
        self.head_softmax = nn.Softmax(dim=1)

        # 初始化模型参数
        self._init_weights()

    def _init_weights(self):
        # 遍历模型的所有模块
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                # 使用 Kaiming 正态分布初始化权重
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    # 计算扇入和扇出
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    # 计算边界值
                    bound = 1 / math.sqrt(fan_out)
                    # 使用正态分布初始化偏置
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        # 对输入数据进行批量归一化
        bn_output = self.tail_batchnorm(input_batch)

        # 依次通过四个 LunaBlock 块
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        # 将卷积输出展平为一维向量
        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        # 通过全连接层得到线性输出
        linear_output = self.head_linear(conv_flat)

        # 返回线性输出和 Softmax 输出
        return linear_output, self.head_softmax(linear_output)

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        # 调用父类的构造函数
        super().__init__()

        # 定义第一个卷积层，输入通道数为 in_channels，输出通道数为 conv_channels
        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        # 定义第一个 ReLU 激活函数
        self.relu1 = nn.ReLU(inplace=True)
        # 定义第二个卷积层，输入通道数为 conv_channels，输出通道数为 conv_channels
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        # 定义第二个 ReLU 激活函数
        self.relu2 = nn.ReLU(inplace=True)

        # 定义最大池化层，池化核大小为 2，步长为 2
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        # 通过第一个卷积层
        block_out = self.conv1(input_batch)
        # 通过第一个 ReLU 激活函数
        block_out = self.relu1(block_out)
        # 通过第二个卷积层
        block_out = self.conv2(block_out)
        # 通过第二个 ReLU 激活函数
        block_out = self.relu2(block_out)

        # 通过最大池化层
        return self.maxpool(block_out)

# 模块中追加一个测试类，方便模块功能测试
class modelCheck:
    def __init__(self, arg):
        # 保存传入的参数
        self.arg = arg
        # 记录初始化信息
        log.info("init {}".format(type(self).__name__))

    def main(self):
        # 实例化 LunaModel 模型
        model = LunaModel()
        # 统计模型的总参数数量
        total_params = sum(p.numel() for p in model.parameters())
        # 打印模型总参数数量
        print(f"模型总参数: {total_params:,}")
        # 遍历模型的所有参数，打印参数名、形状和是否可训练
        for name, param in model.named_parameters():
            print(f"{name}: 形状={param.shape}, 可训练={param.requires_grad}")

if __name__ == "__main__":
    # 创建模型检查实例并调用 main 方法
    checkmodel = modelCheck('参数检查').main()