import argparse
import sys

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from segmentDsets import Luna2dSegmentationDataset
from util.logconf import logging

# 初始化日志记录器
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# 设置日志级别为INFO，用于记录信息
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        # 如果没有传入命令行参数，则使用sys.argv中的参数
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加批量大小参数，默认值为1024
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        # 添加工作进程数量参数，默认值为1
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=0,
            type=int,
        )
        # 以下参数被注释掉，暂不使用
        # parser.add_argument('--scaled',
        #     help="Scale the CT chunks to square voxels.",
        #     default=False,
        #     action='store_true',
        # )

        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        # 记录启动信息，包括类名和命令行参数
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 创建数据加载器，用于加载Luna2dSegmentationDataset数据集
        self.prep_dl = DataLoader(
            Luna2dSegmentationDataset(
                # sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        # 枚举数据集，并显示进度估计
        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        # 遍历每个批次的数据，但不做具体处理，主要用于填充缓存
        for batch_ndx, batch_tup in batch_iter:
            pass


if __name__ == '__main__':
    # 实例化LunaPrepCacheApp类并调用main方法
    LunaPrepCacheApp().main()