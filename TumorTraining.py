import argparse
import datetime
import hashlib
import os
import shutil
import sys

import numpy as np
from matplotlib import pyplot

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import TumorDatasets
import TumorModel

from util.util import enumerateWithEstimate
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# 定义指标索引
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_PRED_P_NDX=2
METRICS_LOSS_NDX=3
METRICS_SIZE = 4

# 定义分类训练应用类
class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加批次大小参数
        parser.add_argument('--batch-size',
            help='设定每个训练批次的数据加载量',
            default=8,
            type=int,
        )
        # 添加工作进程数参数
        parser.add_argument('--num-workers',
            help='设定用于后台加载数据的工作进程数',
            default=4,
            type=int,
        )
        # 添加训练轮数参数
        parser.add_argument('--epochs',
            help='设定训练批次',
            default=10,
            type=int,
        )
        # 添加数据集参数
        parser.add_argument('--dataset',
            help="指定输入模型的数据集",
            action='store',
            default='MalignantLunaDataset',
        )
        # 添加模型参数
        parser.add_argument('--model',
            help="指定用于肿瘤分类的预训练模型",
            action='store',
            default='LunaModel',
        )
        # 添加是否训练恶性分类模型参数
        parser.add_argument('--malignant',
            help="是否将把结节分类模型训练为识别良性或恶性的模型",
            action='store_true',
            default=True,
        )
        # 添加微调模型路径参数
        parser.add_argument('--finetune',
            help="启动微调模型",
            default='data-unversioned/nodule/models/nodule-model/cls_2025-06-29_15.30.21_nodule-comment.best.state',
        )
        # 添加微调模型层次参数
        parser.add_argument('--finetune-depth',
            help="指定微调的的模型层次开，从模型头开始计算",
            type=int,
            default=2,
        )
        # 添加TensorBoard前缀参数
        parser.add_argument('--tb-prefix',
            default='tumor_cls',
            help="设定供tensorboard使用的数据文件的前缀",
        )
        # 添加TensorBoard后缀参数
        parser.add_argument('comment',
            help="设定供tensorboard使用的数据文件的后缀",
            nargs='?',
            default='finetune-depth2',
        )

        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)
        # 获取当前时间字符串
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        # 定义数据增强参数
        self.augmentation_dict = {}
        if True:
        # if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        # if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        # if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        # if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        # if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        # 检查是否有可用的CUDA设备
        self.use_cuda = torch.cuda.is_available()
        # 设置设备为CUDA或CPU
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化模型
        self.model = self.initModel()
        # 初始化优化器
        self.optimizer = self.initOptimizer()

    def initModel(self):
        # 获取指定的模型类
        model_cls = getattr(TumorModel, self.cli_args.model)
        # 实例化模型
        model = model_cls()

        if self.cli_args.finetune:
            # 加载预训练模型参数
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            # 获取模型的子模块名称
            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            # 获取需要微调的模块名称
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")
            # 加载部分模型参数
            model.load_state_dict(
                {
                    k: v for k,v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    # 冻结不需要微调的参数
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                # 使用多GPU并行训练
                model = nn.DataParallel(model)
            # 将模型移动到指定设备
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        # 根据是否微调设置学习率
        lr = 0.003 if self.cli_args.finetune else 0.001
        # 使用随机梯度下降优化器
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        #return Adam(self.model.parameters(), lr=3e-4)

    def initTrainDl(self):
        # 获取指定的数据集类
        ds_cls = getattr(TumorDatasets, self.cli_args.dataset)

        # 实例化训练数据集
        train_ds = ds_cls(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=1,
        )

        # 获取批次大小
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            # 如果使用CUDA，根据GPU数量调整批次大小
            batch_size *= torch.cuda.device_count()

        # 创建训练数据加载器
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        # 获取指定的数据集类
        ds_cls = getattr(TumorDatasets, self.cli_args.dataset)

        # 实例化验证数据集
        val_ds = ds_cls(
            val_stride=10,
            isValSet_bool=True,
        )

        # 获取批次大小
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            # 如果使用CUDA，根据GPU数量调整批次大小
            batch_size *= torch.cuda.device_count()

        # 创建验证数据加载器
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            # 定义TensorBoard日志目录
            log_dir = os.path.join('runs', self.cli_args.tb_prefix,
                                   self.time_str)

            # 创建训练和验证的TensorBoard写入器
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 初始化训练数据加载器
        train_dl = self.initTrainDl()
        # 初始化验证数据加载器
        val_dl = self.initValDl()

        best_score = 0.0
        # 根据是否微调设置验证频率
        validation_cadence = 5 if not self.cli_args.finetune else 1
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            # 进行训练
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            # 记录训练指标
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
            # 进行验证
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)

            # 记录验证指标并获取分数
            score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            if score > best_score:  # 仅当当前分数等于最高分（即新的最佳模型）时保存
                # 保存最佳模型
                self.saveModel('seg', epoch_ndx, True)
                best_score = score
            else:
                # 保存当前模型
                self.saveModel('seg', epoch_ndx, False)

        if hasattr(self, 'trn_writer'):
            # 关闭TensorBoard写入器
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        # 设置模型为训练模式
        self.model.train()
        # 打乱训练数据集样本
        train_dl.dataset.shuffleSamples()
        # 初始化训练指标张量
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        # 枚举训练数据加载器中的批次
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            # 清空优化器梯度
            self.optimizer.zero_grad()

            # 计算批次损失
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                augment=True
            )

            # 反向传播
            loss_var.backward()
            # 更新模型参数
            self.optimizer.step()

        # 更新总训练样本数
        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            # 设置模型为评估模式
            self.model.eval()
            # 初始化验证指标张量
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            # 枚举验证数据加载器中的批次
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                # 计算批次损失
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    augment=False
                )

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         augment=True):
        # 解包批次数据
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        # 将输入数据移动到指定设备
        input_g = input_t.to(self.device, non_blocking=True)
        # 将标签数据移动到指定设备
        label_g = label_t.to(self.device, non_blocking=True)
        # 将索引数据移动到指定设备
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            # 对输入数据进行增强
            input_g = TumorModel.augment3d(input_g)

        # 前向传播
        logits_g, probability_g = self.model(input_g)

        # 计算交叉熵损失
        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1],
                                             reduction="none")
        # 计算当前批次的起始和结束索引
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        # 获取预测标签
        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False,
                                   out=None)

        # 记录标签
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        # 记录预测标签
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        # 记录预测概率
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:,1]
        # 记录损失
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()

    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        # 初始化TensorBoard写入器
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        if self.cli_args.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'

        # 获取负样本标签掩码
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        # 获取负样本预测掩码
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0

        # 获取正样本标签掩码
        posLabel_mask = ~negLabel_mask
        # 获取正样本预测掩码
        posPred_mask = ~negPred_mask

        # 计算负样本数量
        neg_count = int(negLabel_mask.sum())
        # 计算正样本数量
        pos_count = int(posLabel_mask.sum())

        # 计算真负样本数量
        neg_correct = int((negLabel_mask & negPred_mask).sum())
        # 计算真正样本数量
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct

        # 计算假正样本数量
        falsePos_count = neg_count - neg_correct
        # 计算假负样本数量
        falseNeg_count = pos_count - pos_correct

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
            truePos_count / np.float64(truePos_count + falsePos_count)
        # 计算召回率
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float64(truePos_count + falseNeg_count)

        # 计算F1分数
        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        # 生成阈值数组
        threshold = torch.linspace(1, 0, steps=100)
        # 计算真正率
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        # 计算假正率
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        # 计算假正率的差值
        fp_diff = fpr[1:]-fpr[:-1]
        # 计算真正率的平均值
        tp_avg  = (tpr[1:]+tpr[:-1])/2
        # 计算AUC值
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        # 记录综合指标信息
        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score, "
                 + "{auc:.4f} auc"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        # 记录负样本指标信息
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + neg,
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        # 记录正样本指标信息
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + pos,
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        # 获取对应的TensorBoard写入器
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            # 替换键中的pos和neg
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            # 向TensorBoard写入标量
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        # 创建ROC曲线图像
        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        # 向TensorBoard写入ROC曲线图像
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)

        # 向TensorBoard写入AUC值
        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        # 生成直方图的区间
        bins = np.linspace(0, 1)

        # 向TensorBoard写入负样本预测概率直方图
        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        # 向TensorBoard写入正样本预测概率直方图
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        if not self.cli_args.malignant:
            # 如果不是训练恶性分类模型，使用F1分数作为评估指标
            score = metrics_dict['pr/f1_score']
        else:
            # 否则使用AUC值作为评估指标
            score = metrics_dict['auc']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        # 定义模型保存路径
        file_path = os.path.join(
            'data-unversioned',
            'tumor_checkPoint',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        # 创建保存路径的目录
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            # 如果使用多GPU并行训练，获取模型的原始模块
            model = model.module

        # 定义模型保存的状态字典
        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        # 保存模型状态
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            # 定义最佳模型保存路径
            best_path = os.path.join(
                'data-unversioned',
                'tumor',
                'models',
                self.cli_args.tb_prefix,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            # 获取最佳模型保存路径的目录
            des_dir = os.path.dirname(best_path)
            if not os.path.exists(des_dir):
                # 创建最佳模型保存路径的目录
                os.makedirs(des_dir, exist_ok=True)
            # 复制当前模型到最佳模型路径
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            # 记录模型文件的SHA1哈希值
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

if __name__ == '__main__':
    # 实例化分类训练应用类并启动训练
    ClassificationTrainingApp().main()