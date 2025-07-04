# 导入命令行参数解析模块
import argparse
# 导入系统相关模块
import sys
# 导入日期时间处理模块
import datetime
# 导入文件路径匹配模块
import glob
# 导入操作系统相关模块
import os
# 再次导入系统模块（可能用于确保导入成功或覆盖）
import sys

# 导入数值计算库
import numpy as np
# 导入scipy的形态学和测量工具（用于图像处理）
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

# 导入PyTorch深度学习框架
import torch
import torch.nn as nn
import torch.optim

# 导入PyTorch的数据加载工具
from torch.utils.data import DataLoader

# 从自定义工具模块导入带进度估计的枚举工具
from util.util import enumerateWithEstimate
# 导入2D分割数据集类
from segmentDsets import Luna2dSegmentationDataset
# 导入肿瘤相关数据集及工具函数（获取CT数据、候选信息等）
from TumorDatasets import LunaDataset, getCt, getCandidateInfoDict, getCandidateInfoList, CandidateInfoTuple
# 导入UNet分割模型包装类
from segmentModel import UNetWrapper

# 导入肿瘤模型模块
import TumorModel

# 从自定义日志配置模块导入日志工具
from util.logconf import logging
# 从自定义工具模块导入体素坐标到患者坐标的转换函数
from util.util import voxelCoord2patientCoord

# 获取日志记录器实例
log = logging.getLogger(__name__)
# 设置日志级别为DEBUG（调试级别，输出最详细信息）
log.setLevel(logging.DEBUG)
# 设置'tumor.detect'日志器的级别为WARNING（仅输出警告及以上信息）
logging.getLogger("tumor.detect").setLevel(logging.WARNING)


def print_confusion(label, confusions, do_mal):
    """
    打印混淆矩阵结果

    参数:
        label: 混淆矩阵的标签（如系列ID或"Total"）
        confusions: 混淆矩阵数据，形状为(3,4)
        do_mal: 是否包含恶性肿瘤分类（决定列标签）
    """
    # 行标签：非结节、良性、恶性
    row_labels = ['非结节', '良性', '恶性']
    # 根据是否包含恶性分类设置列标签
    if do_mal:
        col_labels = ['', '漏诊', '排除', '预测为良性', '预测为恶性']
    else:
        col_labels = ['', '漏诊', '排除', '预测为结节']
        # 合并预测为良性和恶性的列（都视为预测为结节）
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    # 单元格宽度
    cell_width = 16
    # 格式化字符串（右对齐）
    f = '{:>' + str(cell_width) + '}'
    # 打印标签
    print(label)
    # 打印列标题
    print(' | '.join([f.format(s) for s in col_labels]))
    # 打印每行数据
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        # 非结节行的漏诊列留空
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))


def match_and_score(detections, truth, threshold=0.5):
    """
    匹配检测结果与真实标签并计算混淆矩阵

    参数:
        detections: 检测结果列表
        truth: 真实标签列表（CandidateInfoTuple对象）
        threshold: 分类阈值（默认0.5）
    返回:
        3x4的混淆矩阵（行：非结节/良性/恶性；列：漏诊/排除/预测良性/预测恶性）
    """
    # 提取真实结节信息（仅包含是结节的）
    true_nodules = [c for c in truth if c.isNodule_bool]
    # 结节直径
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    # 结节中心坐标（xyz）
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    # 提取检测结果的坐标（xyz）
    detected_xyz = np.array([n[2] for n in detections])
    # 确定检测结果的预测类别（1=良性, 2=恶性, 3=其他）
    detected_classes = np.array([1 if d[0] < threshold
                                 else (2 if d[1] < threshold
                                       else 3) for d in detections])
    # 初始化3x4混淆矩阵
    # 行: 0=非结节, 1=良性结节, 2=恶性结节
    # 列: 0=完全漏检, 1=过滤排除, 2=预测良性, 3=预测恶性
    confusion = np.zeros((3, 4), dtype=int)
    # 处理特殊情况：无检测结果
    if len(detected_xyz) == 0:
        for tn in true_nodules:
            # 真实结节漏检（根据是否恶性更新对应行）
            confusion[2 if tn.isMal_bool else 1, 0] += 1
    # 处理特殊情况：无真实结节
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            # 非结节的检测结果分类
            confusion[0, dc] += 1
    # 主要匹配逻辑：计算真实结节与检测结果的距离
    else:
        # 计算每个真实结节与每个检测结果的归一化距离（距离/结节直径）
        normalized_dists = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / truth_diams[:,
                                                                                                     None]
        # 距离阈值0.7（小于此值视为匹配）
        matches = (normalized_dists < 0.7)
        # 标记未匹配的检测结果
        unmatched_detections = np.ones(len(detections), dtype=bool)
        # 记录每个真实结节匹配到的最高置信度类别
        matched_true_nodules = np.zeros(len(true_nodules), dtype=int)
        # 处理匹配成功的情况
        for i_tn, i_detection in zip(*matches.nonzero()):
            # 更新真实结节匹配到的最高类别
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            # 标记检测结果为已匹配
            unmatched_detections[i_detection] = False
        # 处理未匹配的检测结果（假阳性）
        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1
        # 处理真实结节的分类结果
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.isMal_bool else 1, dc] += 1
    return confusion


class NoduleAnalysisApp:
    """结节分析应用类，用于CT图像中结节的检测、分割和分类"""

    def __init__(self, sys_argv=None):
        """初始化应用，解析命令行参数，设置设备和模型路径"""
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser()
        # 批处理大小
        parser.add_argument('--batch-size',
                            help='设定加载量',
                            default=16,
                            type=int,
                            )
        # 数据加载工作进程数
        parser.add_argument('--num-workers',
                            help='设定用于后台加载数据的工作进程数',
                            default=4,
                            type=int,
                            )
        # 是否对所有CT影像进行验证
        parser.add_argument('--run-validation',
                            help='是否对所有的CT影像进行验证',
                            action='store_true',
                            default=True,
                            )
        # 是否包含训练集（默认只验证）
        parser.add_argument('--include-train',
                            help="是否包含训练集(默认只对验证集进行验证)",
                            action='store_true',
                            default=False,
                            )
        # 语义分割模型路径
        parser.add_argument('--segmentation-path',
                            help="指定语义分割模型的路径",
                            nargs='?',
                            default='data-unversioned/seg/models/seg/seg_2025-07-02_13.09.34_none.best.state',
                            )
        # 分类器模型类名
        parser.add_argument('--cls-model',
                            help="指定分类器模型的类名.",
                            action='store',
                            default='LunaModel',
                            )
        # 分类器模型文件地址
        parser.add_argument('--classification-path',
                            help="指定保存分类器模型文件的地址",
                            nargs='?',
                            default='data-unversioned/nodule/models/nodule-model/best_2025-07-02_10.36.28_nodule-comment.best.state',
                            )
        # 恶性肿瘤分类器模型类名
        parser.add_argument('--malignancy-model',
                            help="指定恶性肿瘤分类器模型的类名",
                            action='store',
                            default='LunaModel',
                            )
        # 恶性肿瘤分类器模型文件地址
        parser.add_argument('--malignancy-path',
                            help="指定保存恶性肿瘤分类器模型文件的地址",
                            nargs='?',
                            default='data-unversioned/tumor/models/tumor_cls/seg_2025-07-02_14.29.48_finetune-depth2.best.state',
                            )
        # CT影像文件标识
        parser.add_argument('series_uid',
                            nargs='?',
                            default=None,
                            help="指定要使用的CT影像文件的标识",
                            )

        # 解析命令行参数
        self.cli_args = parser.parse_args(sys_argv)

        # 校验参数：series_uid和--run-validation必须且只能提供一个
        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception("One and only one of series_uid and --run-validation should be given")

        # 检查是否使用GPU
        self.use_cuda = torch.cuda.is_available()
        # 设置设备（GPU或CPU）
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化模型
        self.seg_model, self.cls_model, self.malignancy_model = self.initModels()

    def initModels(self):
        """初始化分割模型、分类模型和恶性肿瘤分类模型"""
        log.debug(self.cli_args.segmentation_path)
        # 加载分割模型状态字典
        seg_dict = torch.load(self.cli_args.segmentation_path)

        # 创建UNet分割模型
        seg_model = UNetWrapper(
            in_channels=7,  # 输入通道数
            n_classes=1,  # 输出类别数
            depth=3,  # 网络深度
            wf=4,  # 初始特征图数量的对数（2^4=16）
            padding=True,  # 是否使用填充
            batch_norm=True,  # 是否使用批归一化
            up_mode='upconv',  # 上采样方式
        )

        # 加载模型权重
        seg_model.load_state_dict(seg_dict['model_state'])
        # 设置为评估模式
        seg_model.eval()

        log.debug(self.cli_args.classification_path)
        # 加载分类模型状态字典
        cls_dict = torch.load(self.cli_args.classification_path)

        # 动态获取分类模型类并实例化
        model_cls = getattr(TumorModel, self.cli_args.cls_model)
        cls_model = model_cls()
        # 加载分类模型权重
        cls_model.load_state_dict(cls_dict['model_state'])
        # 设置为评估模式
        cls_model.eval()

        # 如果使用GPU
        if self.use_cuda:
            # 多GPU处理
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            # 将模型移动到GPU
            seg_model.to(self.device)
            cls_model.to(self.device)

        # 初始化恶性肿瘤分类模型（如果提供路径）
        if self.cli_args.malignancy_path:
            model_cls = getattr(TumorModel, self.cli_args.malignancy_model)
            malignancy_model = model_cls()
            malignancy_dict = torch.load(self.cli_args.malignancy_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None
        return seg_model, cls_model, malignancy_model

    def initSegmentationDl(self, series_uid):
        """初始化分割任务的数据加载器"""
        # 创建分割数据集
        seg_ds = Luna2dSegmentationDataset(
            contextSlices_count=3,  # 上下文切片数量
            series_uid=series_uid,  # CT系列ID
            fullCt_bool=True,  # 是否使用完整CT
        )
        # 创建数据加载器
        seg_dl = DataLoader(
            seg_ds,
            # 批次大小（根据GPU数量调整）
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,  # 工作进程数
            pin_memory=self.use_cuda,  # 是否固定内存（加速GPU传输）
        )

        return seg_dl

    def initClassificationDl(self, candidateInfo_list):
        """初始化分类任务的数据加载器"""
        # 创建分类数据集
        cls_ds = LunaDataset(
            sortby_str='series_uid',  # 排序方式
            candidateInfo_list=candidateInfo_list,  # 候选结节信息列表
        )
        # 创建数据加载器
        cls_dl = DataLoader(
            cls_ds,
            # 批次大小（根据GPU数量调整）
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,  # 工作进程数
            pin_memory=self.use_cuda,  # 是否固定内存
        )

        return cls_dl

    def main(self):
        """主函数，协调整个结节分析流程"""
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # 创建验证数据集
        val_ds = LunaDataset(
            val_stride=10,  # 验证集步长
            isValSet_bool=True,  # 标记为验证集
        )
        # 提取验证集中的系列ID集合
        val_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in val_ds.candidateInfo_list
        )

        # 处理输入的系列ID
        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            # 获取所有系列ID
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in getCandidateInfoList()
            )

        # 区分训练集和验证集系列
        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)  # 训练集系列（排除验证集）
        else:
            train_list = []
        val_list = sorted(series_set & val_set)  # 验证集系列

        # 获取候选结节信息字典
        candidateInfo_dict = getCandidateInfoDict()
        # 枚举系列ID（带进度估计）
        series_iter = enumerateWithEstimate(
            val_list + train_list,
            "Series",
        )
        # 初始化总混淆矩阵
        all_confusion = np.zeros((3, 4), dtype=int)
        # 遍历每个系列
        for _, series_uid in series_iter:
            # 获取CT数据
            ct = getCt(series_uid)
            # 分割CT图像，获取掩码
            mask_a = self.segmentCt(ct, series_uid)

            # 根据分割结果分组，获取候选结节信息
            candidateInfo_list = self.groupSegmentationOutput(
                series_uid, ct, mask_a)
            # 对候选结节进行分类
            classifications_list = self.classifyCandidates(
                ct, candidateInfo_list)

            # 如果不运行验证，打印检测结果
            if not self.cli_args.run_validation:
                print(f"found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5:  # 结节概率阈值
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            # 如果该系列有真实标签，计算混淆矩阵
            if series_uid in candidateInfo_dict:
                one_confusion = match_and_score(
                    classifications_list, candidateInfo_dict[series_uid]
                )
                all_confusion += one_confusion
                # 打印该系列的混淆矩阵
                print_confusion(
                    series_uid, one_confusion, self.malignancy_model is not None
                )

        # 打印总混淆矩阵
        print_confusion(
            "Total", all_confusion, self.malignancy_model is not None
        )

    def classifyCandidates(self, ct, candidateInfo_list):
        """对候选结节进行分类，获取良性/恶性概率"""
        # 初始化分类数据加载器
        cls_dl = self.initClassificationDl(candidateInfo_list)
        # 存储分类结果
        classifications_list = []
        # 遍历分类数据加载器
        for batch_ndx, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            # 将输入数据移到设备（GPU/CPU）
            input_g = input_t.to(self.device)
            # 不计算梯度（推理模式）
            with torch.no_grad():
                # 结节分类推理（获取结节概率）
                _, probability_nodule_g = self.cls_model(input_g)
                # 恶性肿瘤分类推理（如果模型存在）
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            # 处理每个候选结节的结果
            zip_iter = zip(center_list,
                           probability_nodule_g[:, 1].tolist(),  # 结节概率（第二列是结节的概率）
                           probability_mal_g[:, 1].tolist())  # 恶性概率（第二列是恶性的概率）
            for center_irc, prob_nodule, prob_mal in zip_iter:
                # 将IRC坐标转换为XYZ坐标（患者坐标系）
                center_xyz = voxelCoord2patientCoord(center_irc,
                                                     direction_a=ct.direction_a,
                                                     origin_xyz=ct.origin_xyz,
                                                     vxSize_xyz=ct.vxSize_xyz,
                                                     )
                # 存储分类结果（概率、坐标）
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup)
        return classifications_list

    def segmentCt(self, ct, series_uid):
        """对CT图像进行分割，生成结节掩码"""
        # 不计算梯度（推理模式）
        with torch.no_grad():
            # 初始化输出数组（与CT图像同形状）
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            # 获取分割数据加载器
            seg_dl = self.initSegmentationDl(series_uid)
            # 遍历分割数据加载器
            for input_t, _, _, slice_ndx_list in seg_dl:

                # 将输入数据移到设备
                input_g = input_t.to(self.device)
                # 分割模型推理
                prediction_g = self.seg_model(input_g)

                # 处理每个切片的预测结果
                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            # 生成二值掩码（阈值0.5）
            mask_a = output_a > 0.5
            # 形态学腐蚀（去除小噪声）
            mask_a = morphology.binary_erosion(mask_a, iterations=1)

        return mask_a

    def groupSegmentationOutput(self, series_uid, ct, clean_a):
        """根据分割掩码分组，获取候选结节的中心坐标"""
        # 对分割掩码进行标记（连通区域分析）
        candidateLabel_a, candidate_count = measurements.label(clean_a)
        # 计算每个候选结节的质心（基于CT值，更接近真实结节中心）
        centerIrc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,  # 调整CT值（避免负值影响质心计算）
            labels=candidateLabel_a,  # 标记后的掩码
            index=np.arange(1, candidate_count + 1),  # 每个候选结节的索引
        )

        # 存储候选结节信息
        candidateInfo_list = []
        for i, center_irc in enumerate(centerIrc_list):
            # 将IRC坐标转换为XYZ坐标（患者坐标系）
            center_xyz = voxelCoord2patientCoord(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            # 检查坐标是否有效
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            # 创建候选结节信息对象（暂设为非结节，后续分类会更新）
            candidateInfo_tup = \
                CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz)
            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list


# 程序入口
if __name__ == '__main__':
    # 创建应用实例并运行主函数
    NoduleAnalysisApp().main()