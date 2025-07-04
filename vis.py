import matplotlib
import matplotlib.pyplot as plt
from dsets import CT, LunaDataset

# 设置绘制图表时要使用的后端，这里使用TkAgg
matplotlib.use('TkAgg')
# 设置绘制图表时要使用的字体，要在图表中显示中文时，一定要设置图表中要使用中文字体，否则中文会乱码
matplotlib.rc("font", family="SimSun")
# 解决坐标轴刻度“负号”乱码问题
matplotlib.rcParams['axes.unicode_minus'] = False
# 设置CT图像的显示范围
clim = (-1000.0, 300)

def findPositiveSamples(limit=100):
    """
    查找阳性样本（结节样本）
    :param limit: 最大查找数量
    :return: 阳性样本列表
    """
    # 创建LunaDataset实例
    ds = LunaDataset()
    positiveSample_list = []
    # 遍历数据集中的所有样本
    for sample_tup in ds.candidateInfo_list:
        if sample_tup.isNodule_bool:
            # 如果样本是结节样本，则添加到阳性样本列表中
            positiveSample_list.append(sample_tup)
        if len(positiveSample_list) >= limit:
            # 如果达到最大查找数量，则停止查找
            break
    return positiveSample_list

def showCandidate(series_uid, batch_ndx=None, **kwargs):
    """
    显示候选结节的CT图像
    :param series_uid: CT扫描的唯一标识符
    :param batch_ndx: 样本索引
    :param kwargs: 其他参数
    """
    # 创建LunaDataset实例，只包含指定series_uid的样本
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    # 找出所有阳性样本的索引
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]
    if batch_ndx is None:
        if pos_list:
            # 如果有阳性样本，则使用第一个阳性样本的索引
            batch_ndx = pos_list[0]
        else:
            # 如果没有阳性样本，则使用第一个阴性样本的索引，并打印警告信息
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0
    # 创建CT实例
    ct = CT(series_uid)
    # 获取指定索引的样本
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    # 将样本转换为numpy数组
    ct_a = ct_t[0].numpy()
    # 创建一个图形窗口
    fig = plt.figure(figsize=(30, 60))
    # 定义子图分组
    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    # 显示原始CT扫描的索引切片
    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('原始CT扫描索引 {}'.format(int(center_irc[0])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap='gray')

    # 显示原始CT扫描的行索引切片
    subplot = fig.add_subplot(len(group_list) + 4, 3, 2)
    subplot.set_title('原始CT扫描行索引 {}'.format(int(center_irc[1])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct.hu_a[:, int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    # 显示原始CT扫描的列索引切片
    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('原始CT扫描列索引 {}'.format(int(center_irc[2])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct.hu_a[:, :, int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    # 显示候选结节的中心切片索引
    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('候选结节的中心切片索引 {}'.format(int(center_irc[0])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct_a[ct_a.shape[0] // 2], clim=clim, cmap='gray')

    # 显示候选结节的中心切片的行索引
    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('候选结节的中心切片的行索引 {}'.format(int(center_irc[1])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct_a[:, ct_a.shape[1] // 2, :], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    # 显示候选结节的中心切片的列索引
    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('候选结节的中心切片的列索引 {}'.format(int(center_irc[2])), fontsize=10)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(8)
    plt.imshow(ct_a[:, :, ct_a.shape[2] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    # 显示候选结节的其他切片
    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('候选结节的切片索引 {}'.format(index), fontsize=10)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(8)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')
    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.6)  # 调整行间距（值为高度的比例）
    # 显示图形
    plt.show()
    # 打印相关信息
    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)