from dsets import getCandidateInfoList, LunaDataset

# 1. 读取所有候选结节信息，requireOnDisk_bool=True表示只获取存在于磁盘上的样本
candidateList = getCandidateInfoList(requireOnDisk_bool=True)
# 打印数据集中的数据总量
print(f"数据集中的数据总量: {len(candidateList)}")

# 2. 划分训练集和验证集（val_stride=10表示每10个样本取1个为验证集）
# 创建训练集，isValSet_bool=False表示是训练集
trainDataset = LunaDataset(val_stride=10, isValSet_bool=False)
# 打印训练集中的数据总量
print(f"训练集中的数据总量: {len(trainDataset)}")

# 创建验证集，isValSet_bool=True表示是验证集
valDataset = LunaDataset(isValSet_bool=True, val_stride=10)
# 打印验证集中的数据总量
print(f"验证集中的数据总量: {len(valDataset)}")