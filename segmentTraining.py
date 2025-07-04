import argparse
import datetime
import hashlib
import os
import shutil
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from util.util import enumerateWithEstimate
from segmentDsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from segmentModel import UNetWrapper, SegmentationAugmentation

# 创建日志记录器，用于记录程序运行过程中的信息
log = logging.getLogger(__name__)
# 设置日志级别为DEBUG，这样可以记录更多详细信息
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# 用于computeClassificationLoss和logMetrics函数中，对metrics_t/metrics_a进行索引
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1  # 损失值的索引
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7  # 真阳性（True Positive）的索引
METRICS_FN_NDX = 8  # 假阴性（False Negative）的索引
METRICS_FP_NDX = 9  # 假阳性（False Positive）的索引
METRICS_TN_NDX = 10 # 真阴性（True Negative）的索引
METRICS_SIZE = 11   # 指标数组的大小

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        # 如果没有传入命令行参数，则使用sys.argv[1:]
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 添加批次大小参数，用于设置每个训练批次的数据加载量
        parser.add_argument('--batch-size',
            help='设定每个训练批次的数据加载量',
            default=16,
            type=int,
        )
        # 添加工作进程数参数，用于设置后台加载数据的工作进程数
        parser.add_argument('--num-workers',
            help='设定用于后台加载数据的工作进程数',
            default=4,
            type=int,
        )
        # 添加训练轮数参数，用于设置训练的总轮数
        parser.add_argument('--epochs',
            help='设定训练批次',
            default=10,
            type=int,
        )

        # 添加数据增强标志参数，用于设置是否进行数据增强
        parser.add_argument('--augmented',
            help="设定是否增强数据",
            action='store_true',
            default=False,
        )
        # 添加图片翻转增强标志参数，用于设置是否通过图片翻转进行数据增强
        parser.add_argument('--augment-flip',
            help="设定是否通过图片翻转来增强数据",
            action='store_true',
            default=False,
        )
        # 添加图片水平偏移增强标志参数，用于设置是否通过图片水平偏移进行数据增强
        parser.add_argument('--augment-offset',
            help="设定是否通过图片的水平偏移来增强数据",
            action='store_true',
            default=False,
        )
        # 添加图片缩放增强标志参数，用于设置是否通过图片缩放进行数据增强
        parser.add_argument('--augment-scale',
            help="设定是否通过图片的缩放来增强数据",
            action='store_true',
            default=False,
        )
        # 添加图片旋转增强标志参数，用于设置是否通过图片旋转进行数据增强
        parser.add_argument('--augment-rotate',
            help="设定是否通过图片的旋转来增强数据",
            action='store_true',
            default=True,
        )
        # 添加图片噪声增强标志参数，用于设置是否在图片中加入随机噪声进行数据增强
        parser.add_argument('--augment-noise',
            help="设定是否在图片中加入随机噪声来增强数据",
            action='store_true',
            default=True,
        )

        # 添加TensorBoard数据文件前缀参数，用于设置TensorBoard使用的数据文件的前缀
        parser.add_argument('--tb-prefix',
            default='seg',
            help="设定供tensorboard使用的数据文件的前缀",
        )

        # 添加TensorBoard数据文件后缀参数，用于设置TensorBoard使用的数据文件的后缀
        parser.add_argument('comment',
            help="设定供tensorboard使用的数据文件的后缀",
            nargs='?',
            default='none',
        )

        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)
        # 获取当前时间并格式化为字符串，用于记录训练开始时间
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        # 初始化训练样本总数
        self.totalTrainingSamples_count = 0
        # 初始化训练和验证的TensorBoard写入器
        self.trn_writer = None
        self.val_writer = None

        # 初始化数据增强字典
        self.augmentation_dict = {}
        # 如果设置了数据增强或图片翻转增强，则添加翻转增强到字典中
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        # 如果设置了数据增强或图片水平偏移增强，则添加水平偏移增强到字典中
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        # 如果设置了数据增强或图片缩放增强，则添加缩放增强到字典中
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        # 如果设置了数据增强或图片旋转增强，则添加旋转增强到字典中
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        # 如果设置了数据增强或图片噪声增强，则添加噪声增强到字典中
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        # 检查是否有可用的CUDA设备
        self.use_cuda = torch.cuda.is_available()
        # 根据是否有CUDA设备选择使用的设备
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化分割模型和数据增强模型
        self.segmentation_model, self.augmentation_model = self.initModel()
        # 初始化优化器
        self.optimizer = self.initOptimizer()


    def initModel(self):
        # 初始化分割模型，使用UNetWrapper
        segmentation_model = UNetWrapper(
            in_channels=7,  # 输入通道数
            n_classes=1,    # 输出类别数
            depth=3,        # 网络深度
            wf=4,           # 特征图的宽度因子
            padding=True,   # 是否使用填充
            batch_norm=True,# 是否使用批量归一化
            up_mode='upconv',# 上采样模式
        )

        # 初始化数据增强模型，传入数据增强字典
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        # 如果有可用的CUDA设备
        if self.use_cuda:
            # 记录使用CUDA设备的信息
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            # 如果有多个CUDA设备，使用数据并行
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            # 将模型移动到CUDA设备上
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initOptimizer(self):
        # 使用随机梯度下降（SGD）优化器，设置学习率和动量
        return SGD(self.segmentation_model.parameters(), lr=0.0001, momentum=0.85)
        # 可以选择使用Adam优化器
        #return Adam(self.segmentation_model.parameters())
        # 也可以选择不同参数的SGD优化器
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)


    def initTrainDl(self):
        # 初始化训练数据集，使用TrainingLuna2dSegmentationDataset
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,      # 验证集的步长
            isValSet_bool=False,# 是否为验证集
            contextSlices_count=3,# 上下文切片的数量
        )

        # 获取批次大小
        batch_size = self.cli_args.batch_size
        # 如果使用CUDA设备，根据设备数量调整批次大小
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # 初始化训练数据加载器
        train_dl = DataLoader(
            train_ds,           # 训练数据集
            batch_size=batch_size, # 批次大小
            num_workers=self.cli_args.num_workers, # 工作进程数
            pin_memory=self.use_cuda, # 是否将数据固定在内存中
        )

        return train_dl

    def initValDl(self):
        # 初始化验证数据集，使用Luna2dSegmentationDataset
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,      # 验证集的步长
            isValSet_bool=True, # 是否为验证集
            contextSlices_count=3,# 上下文切片的数量
        )

        # 获取批次大小
        batch_size = self.cli_args.batch_size
        # 如果使用CUDA设备，根据设备数量调整批次大小
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # 初始化验证数据加载器
        val_dl = DataLoader(
            val_ds,             # 验证数据集
            batch_size=batch_size, # 批次大小
            num_workers=self.cli_args.num_workers, # 工作进程数
            pin_memory=self.use_cuda, # 是否将数据固定在内存中
        )

        return val_dl

    def initTensorboardWriters(self):
        # 如果训练写入器未初始化
        if self.trn_writer is None:
            # 构建TensorBoard日志目录
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            # 初始化训练和验证的TensorBoard写入器
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_trn_seg_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_seg_' + self.cli_args.comment)

    def main(self):
        # 记录程序开始信息
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 初始化训练和验证数据加载器
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        # 初始化最佳分数
        best_score = 0.0
        # 可以设置验证的间隔轮数
        #self.validation_cadence = 5
        # 开始训练循环
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            # 记录当前训练轮数的信息
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            # 进行训练并获取训练指标
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            # 记录训练指标
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            # 可以设置验证的条件
            #if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # 如果需要进行验证
            # 进行验证并获取验证指标
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            # 记录验证指标并获取分数
            score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            # 如果当前分数大于最佳分数，保存最佳模型
            if score > best_score:  # 仅当当前分数等于最高分（即新的最佳模型）时保存
                self.saveModel('seg', epoch_ndx, True)
                best_score = score
            else:
                # 否则保存当前模型
                self.saveModel('seg', epoch_ndx, False)

            # 记录训练和验证的图像
            self.logImages(epoch_ndx, 'trn', train_dl)
            self.logImages(epoch_ndx, 'val', val_dl)

        # 关闭训练和验证的TensorBoard写入器
        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        # 初始化训练指标张量
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        # 将分割模型设置为训练模式
        self.segmentation_model.train()
        # 打乱训练数据集的样本顺序
        train_dl.dataset.shuffleSamples()

        # 枚举训练数据加载器，显示训练进度
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        # 遍历训练批次
        for batch_ndx, batch_tup in batch_iter:
            # 清空优化器的梯度
            self.optimizer.zero_grad()

            # 计算当前批次的损失
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            # 反向传播计算梯度
            loss_var.backward()

            # 更新模型参数
            self.optimizer.step()

        # 更新训练样本总数
        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        # 将训练指标张量移动到CPU上
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        # 不计算梯度
        with torch.no_grad():
            # 初始化验证指标张量
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            # 将分割模型设置为评估模式
            self.segmentation_model.eval()

            # 枚举验证数据加载器，显示验证进度
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            # 遍历验证批次
            for batch_ndx, batch_tup in batch_iter:
                # 计算当前批次的损失
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        # 将验证指标张量移动到CPU上
        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        # 从批次元组中获取输入张量、标签张量、系列列表和切片索引列表
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        # 将输入张量和标签张量移动到指定设备上
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # 如果分割模型处于训练模式且有数据增强设置
        if self.segmentation_model.training and self.augmentation_dict:
            # 对输入和标签进行数据增强
            input_g, label_g = self.augmentation_model(input_g, label_g)

        # 使用分割模型进行预测
        prediction_g = self.segmentation_model(input_g)

        # 计算Dice损失
        diceLoss_g = self.diceLoss(prediction_g, label_g)
        # 计算假阴性损失
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        # 计算当前批次在指标张量中的起始和结束索引
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        # 不计算梯度
        with torch.no_grad():
            # 将预测结果转换为布尔张量
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            # 计算真阳性、假阴性、假阳性和真阴性的数量
            tp = (     predictionBool_g *  label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) *  label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])
            tn = ((1 - predictionBool_g) * (~label_g)).sum(dim=[1,2,3])

            # 将损失和指标值存储到指标张量中
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp
            metrics_g[METRICS_TN_NDX, start_ndx:end_ndx] = tn

        # 返回总损失
        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        # 计算标签的Dice系数分子
        diceLabel_g = label_g.sum(dim=[1,2,3])
        # 计算预测结果的Dice系数分子
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        # 计算预测结果和标签的交集的Dice系数分子
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        # 计算Dice系数
        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        # 返回Dice损失
        return 1 - diceRatio_g


    def logImages(self, epoch_ndx, mode_str, dl):
        # 将分割模型设置为评估模式
        self.segmentation_model.eval()
        # 获取数据集的系列列表并排序，取前12个
        images = sorted(dl.dataset.series_list)[:12]
        # 遍历系列列表
        for series_ndx, series_uid in enumerate(images):
            # 获取CT数据
            ct = getCt(series_uid)

            # 遍历切片索引
            for slice_ndx in range(6):
                # 计算切片索引
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
                # 获取完整切片的样本元组
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                # 从样本元组中获取CT张量、标签张量、系列ID和切片索引
                ct_t, label_t, series_uid, ct_ndx = sample_tup

                # 将CT张量和标签张量移动到指定设备上，并添加一个维度
                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = label_t.to(self.device).unsqueeze(0)

                # 使用分割模型进行预测
                prediction_g = self.segmentation_model(input_g)[0]
                # 将预测结果转换为布尔数组
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                # 将标签转换为布尔数组
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                # 对CT张量进行归一化处理
                ct_t[:-1,:,:] /= 2000
                ct_t[:-1,:,:] += 0.5

                # 获取CT切片数组
                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()
                # 获取对应的TensorBoard写入器
                writer = getattr(self, mode_str + '_writer')

                # 如果是第一轮训练
                if epoch_ndx == 1:
                    # 初始化图像数组
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    # 将CT切片数组复制到图像数组的所有通道
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    # 将标签添加到图像数组的绿色通道
                    image_a[:,:,1] += label_a  # Green

                    # 对图像数组进行归一化处理
                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    # 将标签图像添加到TensorBoard中
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',
                    )

                # 初始化图像数组
                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                # 将CT切片数组复制到图像数组的所有通道
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                # 将预测错误的部分添加到图像数组的红色通道
                image_a[:, :, 0] += prediction_a & (1 - label_a)
                image_a[:, :, 0] += (1 - prediction_a) & label_a
                # 将部分预测错误的部分添加到图像数组的绿色通道
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5

                # 将预测正确的部分添加到图像数组的绿色通道
                image_a[:, :, 1] += prediction_a & label_a
                # 对图像数组进行归一化处理
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                # 将预测图像添加到TensorBoard中
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )
                # 刷新写入器
                writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        # 记录当前训练轮数的信息
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        # 将指标张量转换为numpy数组
        metrics_a = metrics_t.detach().numpy()
        # 计算指标数组的总和
        sum_a = metrics_a.sum(axis=1)
        # 确保指标数组中的所有值都是有限的
        assert np.isfinite(metrics_a).all()

        # 计算所有正样本的数量
        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]
        # 计算所有负样本的数量
        allLabel_count2 = sum_a[METRICS_TN_NDX] + sum_a[METRICS_FP_NDX]

        # 初始化指标字典
        metrics_dict = {}
        # 计算平均损失
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        # 计算真阳性、假阴性、假阳性和真阴性的百分比
        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count2 or 1) * 100
        metrics_dict['percent_all/tn'] = \
            sum_a[METRICS_TN_NDX] / (allLabel_count2 or 1) * 100


        # 计算精确率
        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        # 计算召回率
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        # 计算F1分数
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        # 记录指标信息
        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp,{percent_all/tn:-9.1f}% tn"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        # 初始化TensorBoard写入器
        self.initTensorboardWriters()
        # 获取对应的TensorBoard写入器
        writer = getattr(self, mode_str + '_writer')

        # 设置指标前缀
        prefix_str = 'seg_'

        # 将指标写入TensorBoard
        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        # 刷新写入器
        writer.flush()

        # 返回召回率作为分数
        score = metrics_dict['pr/recall']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        # 构建模型保存路径
        file_path = os.path.join(
            'data-unversioned',
            'seg-checkPoint',
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

        # 获取分割模型
        model = self.segmentation_model
        # 如果使用了数据并行，获取原始模型
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # 保存模型的状态字典、优化器状态、训练轮数等信息
        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        # 保存模型状态到文件
        torch.save(state, file_path)

        # 记录模型保存信息
        log.info("Saved model params to {}".format(file_path))

        # 如果是最佳模型
        if isBest:
            # 构建最佳模型保存路径
            best_path = os.path.join(
                'data-unversioned', 'seg', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')

            # 获取最佳模型保存路径的目录
            des_dir = os.path.dirname(best_path)
            # 如果目录不存在，创建目录
            if not os.path.exists(des_dir):
                os.makedirs(des_dir, exist_ok=True)
            # 复制当前模型到最佳模型路径
            shutil.copyfile(file_path, best_path)

            # 记录最佳模型保存信息
            log.info("Saved model params to {}".format(best_path))

        # 计算模型文件的SHA1哈希值并记录
        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

if __name__ == '__main__':
    # 创建SegmentationTrainingApp实例并调用main方法开始训练
    SegmentationTrainingApp().main()