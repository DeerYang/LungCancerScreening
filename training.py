import hashlib
import shutil
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import datetime
import multiprocessing
import os
from torch.utils.tensorboard import SummaryWriter

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from util.logconf import logging
from dsets import LunaDataset
from model import LunaModel
from util.util import enumerateWithEstimate

# 创建日志记录器，用于记录训练过程中的信息
log = logging.getLogger('Training Model')
# 设置日志级别为 INFO，可记录更多信息
log.setLevel(logging.INFO)
# 再次设置日志级别为 DEBUG，以记录更详细的调试信息
log.setLevel(logging.DEBUG)

# 用于 computeBatchLoss 和 logMetrics 函数中，对 metrics_t/metrics_a 进行索引
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
# 定义指标数组的大小
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        # 如果没有传入命令行参数，则使用系统的命令行参数（去除脚本名）
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加 --num-workers 参数，用于设置后台加载数据的进程数，默认为 4 个进程
        parser.add_argument('--num-workers',
            help='设置后台加载数据的进程数，默认为 8 个进程，这个需要配合训练环境中的 CPU 和 GPU 的数量',
            default=4,
            type=int,
        )
        # 添加 --batch-size 参数，用于设置每个训练轮次需要加载的数据数量
        parser.add_argument('--batch-size',
            help='设置每个训练轮次需要加载的数据数量',
            default=8,
            type=int,
        )
        # 添加 --epochs 参数，用于设置训练的轮次
        parser.add_argument('--epochs',
            help='设置训练的轮次',
            default=5,
            type=int,
        )
        # 添加 --balanced 参数，用于平衡训练数据，使得正负样本各占一半
        parser.add_argument('--balanced',
            help="平衡训练数据，使得正负样本各占一半",
            action='store_true',
            default=True,
        )
        # 添加 --augmented 参数，用于增强训练数据
        parser.add_argument('--augmented',
            help="增强训练数据.",
            action='store_true',
            default=True,
        )
        # 添加 --augment-flip 参数，用于通过图像翻转来增强和平衡数据
        parser.add_argument('--augment-flip',
            help="通过图像翻转来增强和平衡数据",
            action='store_true',
            default=True,
        )
        # 添加 --augment-offset 参数，用于通过图像平移来增强和平衡数据
        parser.add_argument('--augment-offset',
            help="通过图像平移来增强和平衡数据",
            action='store_true',
            default=True,
        )
        # 添加 --augment-scale 参数，用于通过图像缩放来增强和平衡数据
        parser.add_argument('--augment-scale',
            help="通过图像缩放来增强和平衡数据.",
            action='store_true',
            default=False,
        )
        # 添加 --augment-rotate 参数，用于通过旋转图像来增强和平衡数据
        parser.add_argument('--augment-rotate',
            help="通过旋转图像来增强和平衡数据.",
            action='store_true',
            default=False,
        )
        # 添加 --augment-noise 参数，用于通过给图像添加噪声的方法来增强和平衡数据
        parser.add_argument('--augment-noise',
            help="通过给图像添加噪声的方法来增强和平衡数据.",
            action='store_true',
            default=False,
        )
        # 添加 --tb-prefix 参数，用于设置 tensorboard 使用的前缀
        parser.add_argument('--tb-prefix',
            default='nodule-model',
            help="tensorboard使用的前缀.",
        )
        # 添加 comment 参数，用于设置 tensorboard 使用的后缀，可选参数，默认为 'nodule-comment'
        parser.add_argument('comment',
            help="tensorboard使用的后缀",
            nargs='?',
            default='nodule-comment',
        )

        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)
        # 检查是否有可用的 CUDA 设备
        self.use_cuda = torch.cuda.is_available()
        # 根据是否有 CUDA 设备，选择使用 GPU 或 CPU 作为计算设备
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化模型
        self.model = self.initModel()
        # 初始化优化器
        self.optimizer = self.initOptimizer()

        # 获取当前时间并格式化为字符串，用于日志文件名
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # 初始化训练和验证的 TensorBoard 写入器
        self.trn_writer = None
        self.val_writer = None
        # 记录总的训练样本数量
        self.totalTrainingSamples_count = 0
        # 初始化数据增强字典
        self.augmentation_dict = {}
        # 如果启用了数据增强或图像翻转增强，则添加 'flip' 到增强字典
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        # 如果启用了数据增强或图像平移增强，则添加 'offset' 到增强字典
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        # 如果启用了数据增强或图像缩放增强，则添加 'scale' 到增强字典
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        # 如果启用了数据增强或图像旋转增强，则添加 'rotate' 到增强字典
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        # 如果启用了数据增强或图像噪声增强，则添加 'noise' 到增强字典
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        # 初始化 TensorBoard 写入器
        self.initTensorboardWriters()

    def initModel(self):
        # 实例化 LunaModel 模型
        model = LunaModel()
        if self.use_cuda:
            # 如果使用 CUDA，记录使用的 CUDA 设备数量
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            # 如果有多个 CUDA 设备，使用 DataParallel 进行并行计算
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            # 将模型移动到指定的计算设备上
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        # 使用随机梯度下降优化器，设置学习率为 0.0001，动量为 0.85
        return SGD(self.model.parameters(), lr=0.0001, momentum=0.85)
        # 可选择使用 Adam 优化器
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        # 初始化训练数据集，设置验证步长为 10，非验证集，根据命令行参数平衡数据，并使用数据增强
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
            augmentation_dict=self.augmentation_dict,
        )

        # 获取命令行指定的批量大小
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            # 如果使用 CUDA，根据 CUDA 设备数量调整批量大小
            batch_size *= torch.cuda.device_count()

        # 创建训练数据加载器，设置批量大小、工作进程数和是否使用内存锁定
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        # 初始化验证数据集，设置验证步长为 10，为验证集
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        # 获取命令行指定的批量大小
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            # 如果使用 CUDA，根据 CUDA 设备数量调整批量大小
            batch_size *= torch.cuda.device_count()

        # 创建验证数据加载器，设置批量大小、工作进程数和是否使用内存锁定
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        # 设置训练和验证日志的根目录，包含前缀和时间戳
        log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
        # 如果训练写入器未初始化，则创建训练日志写入器
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
        # 如果验证写入器未初始化，则创建验证日志写入器
        if self.val_writer is None:
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        # 记录当前训练轮次和训练类名
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        # 创建负样本标签和预测的掩码
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        # 创建正样本标签和预测的掩码
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        # 统计负样本和正样本的数量
        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        # 统计真负样本和真正样本的数量
        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        # 统计假正样本和假负样本的数量
        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        # 初始化指标字典
        metrics_dict = {}
        # 计算所有样本的平均损失
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        # 计算负样本的平均损失
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        # 计算正样本的平均损失
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        # 计算所有样本的准确率
        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        # 计算负样本的准确率
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        # 计算正样本的准确率
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        # 计算精确率
        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        # 计算召回率
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)

        # 计算 F1 分数
        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)
        f1_score = metrics_dict['pr/f1_score']

        # 记录所有样本的损失、准确率、精确率、召回率和 F1 分数
        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        # 记录负样本的损失和准确率
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        # 记录正样本的损失和准确率
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        # 获取对应的 TensorBoard 写入器
        writer = getattr(self, mode_str + '_writer')

        # 将指标写入 TensorBoard
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        # 写入精确率 - 召回率曲线
        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        # 定义直方图的区间
        bins = [x / 50.0 for x in range(51)]

        # 创建负样本预测概率大于 0.01 的掩码
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        # 创建正样本预测概率小于 0.99 的掩码
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        # 如果有符合条件的负样本，写入负样本预测概率的直方图
        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        # 如果有符合条件的正样本，写入正样本预测概率的直方图
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        return f1_score

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        # 解包批量数据，包含输入数据、标签、系列列表和中心列表
        input_t, label_t, _series_list, _center_list = batch_tup

        # 将输入数据和标签移动到指定的计算设备上
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # 前向传播，得到模型的输出和概率
        logits_g, probability_g = self.model(input_g)

        # 定义交叉熵损失函数，不进行求和
        loss_func = nn.CrossEntropyLoss(reduction='none')
        # 计算损失
        loss_g = loss_func(
            logits_g,
            label_g[:,1],
        )
        # 计算当前批量数据在指标数组中的起始和结束索引
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        # 将标签、预测概率和损失存储到指标数组中
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        # 返回当前批量数据的平均损失
        return loss_g.mean()

    def doTraining(self, epoch_ndx, train_dl):
        # 将模型设置为训练模式
        self.model.train()
        # 打乱训练数据集的样本顺序
        train_dl.dataset.shuffleSamples()

        # 初始化训练指标张量
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )
        # 枚举训练数据加载器，显示训练进度
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            # 清空优化器的梯度
            self.optimizer.zero_grad()

            # 计算当前批量数据的损失
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )
            # 反向传播，计算梯度
            loss_var.backward()
            # 更新模型参数
            self.optimizer.step()

        # 更新总的训练样本数量
        self.totalTrainingSamples_count += len(train_dl.dataset)

        # 将训练指标张量移动到 CPU 上
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        # 禁用梯度计算
        with torch.no_grad():
            # 将模型设置为评估模式
            self.model.eval()
            # 初始化验证指标张量
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            # 枚举验证数据加载器，显示验证进度
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                # 计算当前批量数据的损失
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        # 将验证指标张量移动到 CPU 上
        return valMetrics_g.to('cpu')

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        # 构建模型保存的文件路径
        # 将模型保存到 'data-unversioned/checkPoint/models' 目录下，同时结合命令行参数 tb_prefix 来进一步区分不同的模型
        file_path = os.path.join(
            'data-unversioned',
            'checkPoint',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,  # 模型类型的字符串标识
                self.time_str,  # 时间字符串，用于区分不同时间的模型保存
                self.cli_args.comment,  # 命令行传入的注释信息
                self.totalTrainingSamples_count,  # 总的训练样本数量
            )
        )

        # 创建保存模型文件所需的目录
        # mode=0o755 表示设置目录权限为 755（即所有者有读、写、执行权限，组用户和其他用户有读、执行权限）
        # exist_ok=True 表示如果目录已经存在，不会抛出异常
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        # 获取要保存的模型
        # 如果模型是通过 torch.nn.DataParallel 封装的多 GPU 模型，提取其内部的单 GPU 模型进行保存
        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # 定义要保存的模型状态字典
        state = {
            'sys_argv': sys.argv,  # 保存当前脚本执行时的命令行参数
            'time': str(datetime.datetime.now()),  # 保存当前时间
            'model_state': model.state_dict(),  # 保存模型的参数状态
            'model_name': type(model).__name__,  # 保存模型的类名
            'optimizer_state': self.optimizer.state_dict(),  # 保存优化器的状态
            'optimizer_name': type(self.optimizer).__name__,  # 保存优化器的类名
            'epoch': epoch_ndx,  # 保存当前训练的轮次
            'totalTrainingSamples_count': self.totalTrainingSamples_count,  # 保存总的训练样本数量
        }
        # 将模型状态字典保存到指定的文件路径
        torch.save(state, file_path)

        # 记录保存模型的信息，方便后续查看和调试
        log.info("Saved model params to {}".format(file_path))

        # 如果该模型是当前最佳模型
        if isBest:
            # 构建最佳模型保存的文件路径
            best_path = os.path.join(
                'data-unversioned', 'nodule', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            # 获取最佳模型保存路径的目录
            des_dir = os.path.dirname(best_path)
            # 如果目录不存在，则创建该目录
            if not os.path.exists(des_dir):
                os.makedirs(des_dir, exist_ok=True)
            # 将当前保存的模型文件复制到最佳模型保存路径
            shutil.copy2(file_path, best_path)

            # 记录保存最佳模型的信息
            log.info("Saved model params to {}".format(best_path))

        # 计算保存文件的 SHA1 哈希值
        # 用于验证文件的完整性和唯一性
        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def main(self):
        # 记录训练开始信息，包含训练类名和命令行参数
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 初始化训练数据加载器
        train_dl = self.initTrainDl()
        # 初始化验证数据加载器
        val_dl = self.initValDl()

        best_val_f1 = 0  # 新增：记录最佳F1分数

        # 开始训练循环
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            # 记录当前训练轮次的信息，包含总轮次、训练和验证批次数量、批量大小等
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            # 进行训练，并获取训练指标
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            # 进行验证，并获取验证指标
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            current_val_f1 = self.logMetrics(epoch_ndx, 'val', valMetrics_t)

            # 1. 保存当前epoch的模型（用于断点续训）
            self.saveModel('checkpoint', epoch_ndx, isBest=False)

            # 2. 如果当前验证集性能优于之前的最佳性能，则保存最佳模型
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                self.saveModel('best', epoch_ndx, isBest=True)
                log.info(f"Saved new best model with F1: {best_val_f1:.4f}")
        '''
        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()
        '''

if __name__ == '__main__':
    # 创建训练应用实例并开始训练
    LunaTrainingApp().main()