import argparse
import sys
import shutil
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from TumorDatasets import LunaDataset
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

# 定义Luna预缓存应用类
class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加批次大小参数
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        # 添加工作进程数参数
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )
        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 创建数据加载器
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        # 枚举数据加载器中的批次
        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            pass

if __name__ == '__main__':
    # 实例化Luna预缓存应用类并启动缓存填充
    LunaPrepCacheApp().main()