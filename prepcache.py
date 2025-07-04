import argparse
import sys
from torch.utils.data import DataLoader
from util.util import enumerateWithEstimate
from dsets import LunaDataset
from util.logconf import logging
#from cachedsets import LunaDataset

log = logging.getLogger(__name__)
# 设置日志级别为INFO
log.setLevel(logging.INFO)

class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        """
        初始化LunaPrepCacheApp类
        :param sys_argv: 命令行参数
        """
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加批量大小参数
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        # 添加工作进程数量参数
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )

        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        """
        主函数，填充缓存
        """
        # 打印开始信息
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 创建数据加载器
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        # 带有估计剩余时间的枚举迭代器
        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass

if __name__ == '__main__':
    # 调用主函数
    LunaPrepCacheApp().main()