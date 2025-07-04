import csv
import functools
import glob
import os
import copy
import util.util as myUtil
import SimpleITK as sitk
import numpy as np
import torch
import torch.cuda
import random
import math

import torch.nn.functional as F
from collections import namedtuple
from util.disk import getCache
from torch.utils.data import Dataset
from util.logconf import logging

log = logging.getLogger('Datasets')
# 设置日志级别为INFO
log.setLevel(logging.INFO)
# 设置日志级别为DEBUG
log.setLevel(logging.DEBUG)

# 获取缓存对象，并设置缓存目录
raw_cache = getCache('raw_data')

# 创建一个命名元组，用于存储候选结节信息
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

@functools.lru_cache(1) # 缓存处理后的数据
def getCandidateInfoList(requireOnDisk_bool=True):
    """
    获取候选结节信息列表
    :param requireOnDisk_bool: 是否只获取存在于磁盘上的样本
    :return: 候选结节信息列表
    """
    # 把所有的*.mhd文件名读到一个列表中
    mhd_list = glob.glob('data-unversioned/data/subset*/*.mhd')
    # 把CT扫描标识保存到一个集合中
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    # 读取标注结节信息
    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # 候选结节的中心坐标
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            # 候选结节的直径
            annotationDiameter_mm = float(row[4])
            # 把标注信息保存到一个字典中，其中键值为CT扫描文件标识
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
    # 读取候选结节信息
    candidateInfo_list = []
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # 过滤掉没有CT扫描文件的候选结节。
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # 把候选结节的良、恶标识转换为布尔值
            isNodule_bool = bool(int(row[4]))
            # 把候选结节的中心坐标转换为一个元组（X,Y,Z）
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            # 匹配确诊恶性肿瘤，并补齐肿瘤直径
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            # 把候选结节信息添加到命名元组的列表中
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))
    # 对列表进行降序排序
    candidateInfo_list.sort(reverse=True)
    # 返回候选元组列表
    return candidateInfo_list

class CT:
    def __init__(self, series_uid):
        """
        初始化CT类
        :param series_uid: CT扫描的唯一标识符
        """
        # 获取CT扫描的mhd文件路径
        mhd_path = glob.glob(
            'data-unversioned/data/subset*/{}.mhd'.format(series_uid)
        )[0]

        # 读取mhd文件
        ct_mhd = sitk.ReadImage(mhd_path)
        # 将CT图像转换为numpy数组
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # 对CT图像进行裁剪，将像素值限制在-1000到1000之间
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        # 获取CT图像的原点坐标
        self.origin_xyz = myUtil.PatientCoordTuple(*ct_mhd.GetOrigin())
        # 获取CT图像的体素尺寸
        self.vxSize_xyz = myUtil.PatientCoordTuple(*ct_mhd.GetSpacing())
        # 获取CT图像的方向矩阵
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        """
        获取候选结节的原始数据
        :param center_xyz: 候选结节的中心坐标
        :param width_irc: 候选结节的尺寸
        :return: 候选结节的原始数据和中心索引
        """
        # 把候选结节的患者坐标转换为图像存储的体素位置索引坐标
        center_irc = myUtil.patientCoord2voxelCoord(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )
        slice_list = []
        # 从CT扫描的体素数组中选出候选结节区域
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
            assert center_val >= 0 and center_val < self.hu_a.shape[axis],repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        # 返回结节对应的三维体素数组和其对应的中心索引元组
        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    """
    获取CT对象
    :param series_uid: CT扫描的唯一标识符
    :return: CT对象
    """
    return CT(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    """
    获取候选结节的原始数据，并进行缓存
    :param series_uid: CT扫描的唯一标识符
    :param center_xyz: 候选结节的中心坐标
    :param width_irc: 候选结节的尺寸
    :return: 候选结节的原始数据和中心索引
    """
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    """
    获取经过数据增强的候选结节数据
    :param augmentation_dict: 数据增强的参数字典
    :param series_uid: CT扫描的唯一标识符
    :param center_xyz: 候选结节的中心坐标
    :param width_irc: 候选结节的尺寸
    :param use_cache: 是否使用缓存
    :return: 经过数据增强的候选结节数据和中心索引
    """
    if use_cache:
        # 从缓存中获取候选结节的原始数据
        ct_chunk, center_irc = \
            getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        # 直接获取候选结节的原始数据
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    # 将候选结节的原始数据转换为torch张量
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    # 创建一个4x4的单位矩阵
    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                # 随机翻转
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            # 随机偏移
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            # 随机缩放
            transform_t[i,i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        # 随机旋转
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    # 生成仿射变换网格
    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    # 进行网格采样，得到经过数据增强的候选结节数据
    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        # 添加随机噪声
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc

class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidateInfo_list=None,
            ):
        """
        初始化LunaDataset类
        :param val_stride: 验证集的采样步长
        :param isValSet_bool: 是否为验证集
        :param series_uid: CT扫描的唯一标识符
        :param sortby_str: 排序方式
        :param ratio_int: 正负样本比例
        :param augmentation_dict: 数据增强的参数字典
        :param candidateInfo_list: 候选结节信息列表
        """
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            # 如果提供了候选结节信息列表，则直接使用
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            # 否则，获取候选结节信息列表
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        # 只保留与series_uid匹配的候选结节
        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        # # 取验证集
        if isValSet_bool:
            assert val_stride > 0, val_stride
            # 每隔val_stride个样本取一个作为验证集
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            #  # 从训练集中剔除验证样本
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            # 随机打乱候选结节信息列表
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            # 按series_uid和中心坐标排序
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            # 未知的排序方式，抛出异常
            raise Exception("Unknown sort: " + repr(sortby_str))

        # 筛选出阴性样本
        self.negative_list = [
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool
        ]
        # 筛选出阳性样本
        self.pos_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ]

        # 打印数据集信息
        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        """
        打乱正负样本列表
        """
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        """
        获取数据集的长度
        :return: 数据集的长度
        """
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        """
        获取指定索引的样本
        :param ndx: 样本索引
        :return: 候选结节数据、标签、CT扫描的唯一标识符和中心索引
        """
        if self.ratio_int:
            # 按正负样本比例获取样本
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            # 直接按索引获取样本
            candidateInfo_tup = self.candidateInfo_list[ndx]

        # 候选结节的尺寸
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            # 如果提供了数据增强的参数字典，则获取经过数据增强的候选结节数据
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            # 如果使用缓存，则从缓存中获取候选结节的原始数据
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            # 将候选结节的原始数据转换为torch张量
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            # 否则，直接获取候选结节的原始数据
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            # 将候选结节的原始数据转换为torch张量
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        # 生成标签张量
        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)

# 模块中追加一个测试类，方便模块功能测试
class testDataset:
    def __init__(self,arg):
        """
        初始化测试类
        :param arg: 参数
        """
        self.arg = arg
        # 打印初始化信息
        log.info("init {}".format(type(self).__name__))

    def main(self):
        """
        测试数据集
        """
        # 打印开始信息
        log.info("Starting {}".format(type(self).__name__))
        # 获取候选结节信息列表
        candidateList = getCandidateInfoList(requireOnDisk_bool=True)
        # 打印数据集中的数据总量
        log.info("数据集中的数据总量: {}".format(len(candidateList)))
        # 创建训练集
        trainDataset = LunaDataset(val_stride=10, isValSet_bool=True, )
        # 打印训练集中的数据总量
        log.info("训练集中的数据总量: {}".format(len(trainDataset)))
        # 创建验证集
        valDataset = LunaDataset(isValSet_bool=True, val_stride=10)
        # 打印验证集中的数据总量
        log.info("验证集中的数据总量: {}".format(len(valDataset)))