import csv
import functools
import glob

import os
import random
from collections import namedtuple
import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import PatientCoordTuple, patientCoord2voxelCoord
from util.logconf import logging

# 初始化日志记录器
log = logging.getLogger(__name__)
# 设置日志级别为DEBUG，用于调试信息
log.setLevel(logging.DEBUG)

# 获取缓存对象
raw_cache = getCache('seg_data')

# 定义候选信息元组，包含结节相关信息
CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # 获取所有.mhd文件的路径
    mhd_list = glob.glob('data-unversioned/data/subset*/*.mhd')
    # 获取存在于磁盘上的系列UID集合
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    # 读取标注文件
    with open('data/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # 如果要求数据必须在磁盘上，且当前系列UID不在磁盘上，则跳过
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # 获取标注中心的坐标
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            # 获取标注的直径
            annotationDiameter_mm = float(row[4])
            # 获取是否为恶性的布尔值
            isMal_bool = {'False': False, 'True': True}[row[5]]

            # 将标注信息添加到候选信息列表中
            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,
                    True,
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    # 读取候选文件
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # 如果要求数据必须在磁盘上，且当前系列UID不在磁盘上，则跳过
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # 判断是否为结节
            isNodule_bool = bool(int(row[4]))
            # 获取候选中心的坐标
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            if not isNodule_bool:
                # 将非结节信息添加到候选信息列表中
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    # 对候选信息列表进行降序排序
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    # 获取候选信息列表
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    # 将候选信息列表转换为字典，键为系列UID，值为对应的候选信息元组列表
    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict

class Ct:
    def __init__(self, series_uid):
        # 获取指定系列UID的.mhd文件路径
        mhd_paths = glob.glob(
            'data-unversioned/data/subset*/{}.mhd'.format(series_uid)
        )
        
        # 检查是否找到了对应的.mhd文件
        if not mhd_paths:
            raise FileNotFoundError(f"找不到系列UID {series_uid} 对应的.mhd文件")
        
        mhd_path = mhd_paths[0]

        # 读取.mhd文件
        ct_mhd = sitk.ReadImage(mhd_path)
        # 将图像数据转换为numpy数组
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        self.series_uid = series_uid

        # 获取图像的原点坐标
        self.origin_xyz = PatientCoordTuple(*ct_mhd.GetOrigin())
        # 获取图像的体素大小
        self.vxSize_xyz = PatientCoordTuple(*ct_mhd.GetSpacing())
        # 获取图像的方向矩阵
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        # 获取当前系列UID的候选信息列表
        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        # 筛选出阳性（结节）信息列表
        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ]
        # 构建阳性掩码
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        # 获取阳性掩码中存在结节的切片索引列表
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2))
                                 .nonzero()[0].tolist())

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        # 初始化边界框数组
        boundingBox_a = np.zeros_like(self.hu_a, dtype=bool)

        for candidateInfo_tup in positiveInfo_list:
            # 将患者坐标转换为体素坐标
            center_irc = patientCoord2voxelCoord(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                # 扩展索引半径，直到遇到低于阈值的HU值或超出边界
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                # 扩展行半径，直到遇到低于阈值的HU值或超出边界
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                # 扩展列半径，直到遇到低于阈值的HU值或超出边界
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # 更新边界框数组
            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True

        # 构建掩码数组，只保留HU值大于阈值的区域
        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        # 将患者坐标转换为体素坐标
        center_irc = patientCoord2voxelCoord(center_xyz, self.origin_xyz, self.vxSize_xyz,
                             self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            # 计算起始索引
            start_ndx = int(round(center_val - width_irc[axis]/2))
            # 计算结束索引
            end_ndx = int(start_ndx + width_irc[axis])

            # 确保中心值在合法范围内
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            # 将切片索引添加到列表中
            slice_list.append(slice(start_ndx, end_ndx))

        # 获取CT图像块
        ct_chunk = self.hu_a[tuple(slice_list)]
        # 获取阳性掩码块
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    # 获取指定系列UID的CT对象
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    # 获取指定系列UID的CT对象
    ct = getCt(series_uid)
    # 获取原始候选数据
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz,
                                                         width_irc)
    # 对CT图像块进行裁剪，将值限制在-1000到1000之间
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    # 获取指定系列UID的CT对象
    ct = Ct(series_uid)
    # 返回CT图像的切片数量和阳性切片索引列表
    return int(ct.hu_a.shape[0]), ct.positive_indexes

class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
            ):
        # 上下文切片数量
        self.contextSlices_count = contextSlices_count
        # 是否使用完整CT数据
        self.fullCt_bool = fullCt_bool

        if series_uid:
            # 如果指定了系列UID，则使用该系列UID
            self.series_list = [series_uid]
        else:
            # 否则，获取所有系列UID并排序
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            # 如果是验证集，按照指定步长选取系列UID
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            # 如果是训练集，去除按照指定步长选取的系列UID
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            # 获取CT图像的切片数量和阳性切片索引列表
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                # 如果使用完整CT数据，将所有切片索引添加到样本列表中
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                # 否则，只将阳性切片索引添加到样本列表中
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]

        # 获取候选信息列表
        self.candidateInfo_list = getCandidateInfoList()
        # 获取样本列表中系列UID的集合
        series_set = set(self.series_list)
        # 筛选出样本列表中系列UID对应的候选信息
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]
        # 筛选出阳性（结节）候选信息
        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]

        # 记录数据集信息
        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        # 返回样本列表的长度
        return len(self.sample_list)

    def __getitem__(self, ndx):
        # 获取指定索引的样本信息
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        # 获取指定系列UID的CT对象
        ct = getCt(series_uid)
        # 初始化CT张量
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        # 计算起始和结束切片索引
        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            # 确保切片索引在合法范围内
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            # 将CT图像切片转换为张量并赋值给ct_t
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # 对CT张量进行裁剪，将值限制在-1000到1000之间
        ct_t.clamp_(-1000, 1000)

        # 将阳性掩码切片转换为张量并添加一个维度
        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

        # 正负样本比例
        self.ratio_int = 2

    def __len__(self):
        # 返回训练集的长度
        return 300000

    def shuffleSamples(self):
        # 打乱候选信息列表和阳性信息列表的顺序
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        # 获取指定索引的阳性候选信息
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        # 获取原始候选数据
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        # 截取中间切片的阳性掩码
        pos_a = pos_a[3:4]

        # 随机生成行和列的偏移量
        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        # 裁剪CT图像并转换为张量
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64,
                                     col_offset:col_offset+64]).to(torch.float32)
        # 裁剪阳性掩码并转换为张量
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64,
                                       col_offset:col_offset+64]).to(torch.long)

        # 获取中心切片的索引
        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx