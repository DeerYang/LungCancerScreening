import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import PatientCoordTuple, patientCoord2voxelCoord
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# 获取肿瘤数据的缓存
raw_cache = getCache('tumor_data')

# 定义候选信息元组，包含是否为结节、是否有注释、是否为恶性、直径、系列UID和中心点坐标
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz',
)
# 定义掩码元组，包含多种掩码信息
MaskTuple = namedtuple(
    'MaskTuple',
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask',
)

# 使用functools.lru_cache进行缓存，只缓存一次结果
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # 获取所有.mhd文件的路径
    mhd_list = glob.glob('data-unversioned/data/subset*/*.mhd')
    # 获取存在于磁盘上的系列UID集合
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    # 读取注释文件
    with open('data/annotations_with_malignancy.csv', "r") as f:
        # 跳过标题行
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # 如果需要文件在磁盘上且当前系列UID不在磁盘上，则跳过
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # 获取注释中心点坐标
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            # 获取注释直径
            annotationDiameter_mm = float(row[4])
            # 获取是否为恶性的布尔值
            isMal_bool = {'False': False, 'True': True}[row[5]]

            # 将信息添加到候选信息列表中
            candidateInfo_list.append(CandidateInfoTuple(True, True, isMal_bool, annotationDiameter_mm, series_uid, annotationCenter_xyz))

    # 读取候选文件
    with open('data/candidates.csv', "r") as f:
        # 跳过标题行
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # 如果需要文件在磁盘上且当前系列UID不在磁盘上，则跳过
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            # 获取是否为结节的布尔值
            isNodule_bool = bool(int(row[4]))
            # 获取候选中心点坐标
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                # 如果不是结节，添加相应信息到候选信息列表
                candidateInfo_list.append(CandidateInfoTuple(
                    False,
                    False,
                    False,
                    0.0,
                    series_uid,
                    candidateCenter_xyz,
                ))

    # 对候选信息列表进行降序排序
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

# 使用functools.lru_cache进行缓存，只缓存一次结果
@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    # 获取候选信息列表
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        # 将候选信息按系列UID分组存储在字典中
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(candidateInfo_tup)

    return candidateInfo_dict

# 定义CT类，用于处理CT数据
class Ct:
    def __init__(self, series_uid):
        # 获取指定系列UID的.mhd文件路径
        mhd_path = glob.glob(
            'data-unversioned/data/subset*/{}.mhd'.format(series_uid)
        )[0]

        # 使用SimpleITK读取.mhd文件
        ct_mhd = sitk.ReadImage(mhd_path)
        # 将图像数据转换为numpy数组
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # 对CT值进行裁剪，限制在-1000到1000之间
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        # 获取图像的原点坐标
        self.origin_xyz = PatientCoordTuple(*ct_mhd.GetOrigin())
        # 获取图像的体素大小
        self.vxSize_xyz = PatientCoordTuple(*ct_mhd.GetSpacing())
        # 获取图像的方向矩阵
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        # 将患者坐标转换为体素坐标
        center_irc = patientCoord2voxelCoord(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            # 计算起始索引
            start_ndx = int(round(center_val - width_irc[axis]/2))
            # 计算结束索引
            end_ndx = int(start_ndx + width_irc[axis])

            # 确保中心点坐标在CT数据范围内
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # 如果起始索引小于0，调整为0
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # 如果结束索引大于CT数据大小，调整为CT数据大小
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            # 将切片信息添加到列表中
            slice_list.append(slice(start_ndx, end_ndx))

        # 从CT数据中提取候选区域
        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

# 使用functools.lru_cache进行缓存，只缓存一次结果
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

# 使用缓存装饰器，对结果进行缓存
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    # 获取指定系列UID的CT数据
    ct = getCt(series_uid)
    # 获取原始候选区域
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

# 使用缓存装饰器，对结果进行缓存
@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    # 获取指定系列UID的CT数据，不构建掩码
    ct = Ct(series_uid, buildMasks_bool=False)
    return len(ct.negative_indexes)

# 获取增强后的候选区域
def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    if use_cache:
        # 从缓存中获取原始候选区域
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        # 直接获取原始候选区域
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    # 将候选区域转换为torch张量，并添加维度
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    # 初始化变换矩阵
    transform_t = torch.eye(4)
    for i in range(3):
        if 'flip' in augmentation_dict:
            # 以0.5的概率进行翻转
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            # 获取偏移量
            offset_float = augmentation_dict['offset']
            # 生成随机偏移值
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            # 获取缩放比例
            scale_float = augmentation_dict['scale']
            # 生成随机缩放值
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        # 生成随机旋转角度
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        # 构建旋转矩阵
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # 合并变换矩阵
        transform_t @= rotation_t

    # 生成仿射网格
    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    # 进行网格采样，得到增强后的候选区域
    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        # 生成随机噪声
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        # 将噪声添加到增强后的候选区域
        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc

# 定义Luna数据集类，继承自torch.utils.data.Dataset
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
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            # 如果提供了候选信息列表，复制该列表
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            # 否则获取候选信息列表
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            # 如果指定了系列UID，只包含该系列的数据
            self.series_list = [series_uid]
        else:
            # 否则获取所有系列UID
            self.series_list = sorted(set(candidateInfo_tup.series_uid for candidateInfo_tup in self.candidateInfo_list))

        if isValSet_bool:
            # 如果是验证集，按步长选择系列UID
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            # 如果是训练集，移除按步长选择的系列UID
            del self.series_list[::val_stride]
            assert self.series_list

        # 获取当前数据集包含的系列UID集合
        series_set = set(self.series_list)
        # 筛选出属于当前数据集系列UID的候选信息
        self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid in series_set]

        if sortby_str == 'random':
            # 随机打乱候选信息列表
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            # 按系列UID和中心点坐标排序
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            # 抛出异常，提示未知排序方式
            raise Exception("Unknown sort: " + repr(sortby_str))

        # 筛选出非结节的候选信息
        self.neg_list = \
            [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        # 筛选出结节的候选信息
        self.pos_list = \
            [nt for nt in self.candidateInfo_list if nt.isNodule_bool]
        # 筛选出良性结节的候选信息
        self.ben_list = \
            [nt for nt in self.pos_list if not nt.isMal_bool]
        # 筛选出恶性结节的候选信息
        self.mal_list = \
            [nt for nt in self.pos_list if nt.isMal_bool]

        # 记录数据集信息
        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            # 如果设置了正负样本比例，打乱所有列表
            random.shuffle(self.candidateInfo_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            # 如果设置了正负样本比例，返回固定长度
            return 50000
        else:
            # 否则返回候选信息列表长度
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            # 计算正样本索引
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                # 如果不是正样本，获取负样本
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidateInfo_tup = self.neg_list[neg_ndx]
            else:
                # 如果是正样本，获取正样本
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            # 如果未设置正负样本比例，直接获取候选信息
            candidateInfo_tup = self.candidateInfo_list[ndx]

        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isNodule_bool
        )

    def sampleFromCandidateInfo_tup(self, candidateInfo_tup, label_bool):
        # 定义候选区域的大小
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            # 如果设置了增强参数，获取增强后的候选区域
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            # 如果使用缓存，从缓存中获取原始候选区域
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            # 将候选区域转换为torch张量，并添加维度
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            # 直接获取原始候选区域
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            # 将候选区域转换为torch张量，并添加维度
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        # 初始化标签张量
        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            # 如果不是结节，设置第一个标签为True
            label_t[0] = True
            index_t = 0
        else:
            # 如果是结节，设置第二个标签为True
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)

# 定义恶性Luna数据集类，继承自LunaDataset
class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            # 如果设置了正负样本比例，返回固定长度
            return 100000
        else:
            # 否则返回良性和恶性结节列表长度之和
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            if ndx % 2 != 0:
                # 如果索引为奇数，获取恶性结节
                candidateInfo_tup = self.mal_list[(ndx // 2) % len(self.mal_list)]
            elif ndx % 4 == 0:
                # 如果索引能被4整除，获取良性结节
                candidateInfo_tup = self.ben_list[(ndx // 4) % len(self.ben_list)]
            else:
                # 否则获取负样本
                candidateInfo_tup = self.neg_list[(ndx // 4) % len(self.neg_list)]
        else:
            if ndx >= len(self.ben_list):
                # 如果索引大于等于良性结节列表长度，获取恶性结节
                candidateInfo_tup = self.mal_list[ndx - len(self.ben_list)]
            else:
                # 否则获取良性结节
                candidateInfo_tup = self.ben_list[ndx]
        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isMal_bool
        )