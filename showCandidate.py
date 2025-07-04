from vis import showCandidate,findPositiveSamples

# 查找阳性样本，最多查找1个
positiveSamples = findPositiveSamples(limit=1)
# 获取第一个阳性样本的CT扫描唯一标识符
uid = positiveSamples[0].series_uid
# 打印阳性样本的数量
print(len(positiveSamples))

# 显示候选结节的CT图像
showCandidate(series_uid = uid)