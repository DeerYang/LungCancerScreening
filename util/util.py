import collections
import datetime
import time
import numpy as np

from util.logconf import logging

log = logging.getLogger(__name__)
# 设置日志级别为WARN
log.setLevel(logging.WARN)
# 设置日志级别为INFO
log.setLevel(logging.INFO)
# 设置日志级别为DEBUG
log.setLevel(logging.DEBUG)

# 定义一个命名元组，用于表示体素坐标，包含索引、行和列
VoxelCoordTuple = collections.namedtuple('VoxelCoordTuple', ['index', 'row', 'col'])
# 定义一个命名元组，用于表示患者坐标，包含x、y和z
PatientCoordTuple = collections.namedtuple('PatientCoordTuple', ['x', 'y', 'z'])

def voxelCoord2patientCoord(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    """
    将体素坐标转换为患者坐标
    :param coord_irc: 体素坐标（索引、行、列）
    :param origin_xyz: 患者坐标系的原点
    :param vxSize_xyz: 体素的尺寸
    :param direction_a: 方向矩阵
    :return: 患者坐标
    """
    # 将体素坐标反转并转换为numpy数组
    cri_a = np.array(coord_irc)[::-1]
    # 将原点坐标转换为numpy数组
    origin_a = np.array(origin_xyz)
    # 将体素尺寸转换为numpy数组
    vxSize_a = np.array(vxSize_xyz)
    # 根据公式计算患者坐标
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # 返回患者坐标的命名元组
    return PatientCoordTuple(*coords_xyz)

def patientCoord2voxelCoord(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    """
    将患者坐标转换为体素坐标
    :param coord_xyz: 患者坐标（x、y、z）
    :param origin_xyz: 患者坐标系的原点
    :param vxSize_xyz: 体素的尺寸
    :param direction_a: 方向矩阵
    :return: 体素坐标
    """
    # 将原点坐标转换为numpy数组
    origin_a = np.array(origin_xyz)
    # 将体素尺寸转换为numpy数组
    vxSize_a = np.array(vxSize_xyz)
    # 将患者坐标转换为numpy数组
    coord_a = np.array(coord_xyz)
    # 根据公式计算体素坐标
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    # 对体素坐标进行四舍五入
    cri_a = np.round(cri_a)
    # 返回体素坐标的命名元组
    return VoxelCoordTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def importstr(module_str, from_=None):
    """
    动态导入模块或模块中的对象
    :param module_str: 模块名
    :param from_: 要导入的对象名
    :return: 导入的模块或对象
    """
    if from_ is None and ':' in module_str:
        # 如果module_str中包含冒号，则分割出模块名和对象名
        module_str, from_ = module_str.rsplit(':')

    # 导入模块
    module = __import__(module_str)
    # 如果模块名包含点号，则获取子模块
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)
    if from_:
        try:
            # 如果指定了对象名，则从模块中获取该对象
            return getattr(module, from_)
        except:
            # 如果获取失败，则抛出导入错误
            raise ImportError('{}.{}'.format(module_str, from_))
    # 返回导入的模块
    return module

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    带有估计剩余时间的枚举函数
    :param iter: 可迭代对象
    :param desc_str: 描述信息
    :param start_ndx: 开始计时的迭代索引
    :param print_ndx: 开始打印估计信息的迭代索引
    :param backoff: 打印间隔的倍增系数
    :param iter_len: 可迭代对象的长度
    :return: 迭代索引和迭代项的元组
    """
    if iter_len is None:
        # 如果未指定可迭代对象的长度，则获取其长度
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        # 根据可迭代对象的长度调整backoff的值
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    # 确保print_ndx不小于start_ndx * backoff
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    # 打印开始信息
    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    # 记录开始时间
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        # 生成迭代索引和迭代项的元组
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # 计算剩余时间
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len - start_ndx)
                            )

            # 计算完成时间
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            # 计算剩余时间的时间差
            done_td = datetime.timedelta(seconds=duration_sec)

            # 打印估计信息
            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            # 调整下一次打印的迭代索引
            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            # 记录开始计时的时间
            start_ts = time.time()

    # 打印完成信息
    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))