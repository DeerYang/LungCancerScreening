import gzip
# from diskcache import FanoutCache, Disk
# from diskcache.core import BytesType, MODE_BINARY, BytesIO
from cassandra.cqltypes import BytesType
from diskcache import FanoutCache, Disk,core
from diskcache.core import io
from io import BytesIO
from diskcache.core import MODE_BINARY

class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        重写diskcache.Disk的store方法，对存储的值进行gzip压缩
        :param value: 要存储的值
        :param read: 是否以读取模式存储
        :param key: 键
        :return: 存储结果
        """
        if type(value) is BytesType:
            if read:
                # 如果以读取模式存储，则读取值
                value = value.read()
                read = False

            # 创建一个BytesIO对象
            str_io = BytesIO()
            # 创建一个gzip文件对象
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            # 分块写入数据
            for offset in range(0, len(value), 2 ** 30):
                gz_file.write(value[offset:offset + 2 ** 30])
            # 关闭gzip文件对象
            gz_file.close()

            # 获取压缩后的数据
            value = str_io.getvalue()

        # 调用父类的store方法进行存储
        return super(GzipDisk, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        """
        重写diskcache.Disk的fetch方法，对读取的值进行gzip解压缩
        :param mode: 读取模式
        :param filename: 文件名
        :param value: 要读取的值
        :param read: 是否以读取模式读取
        :return: 读取结果
        """
        # 调用父类的fetch方法进行读取
        value = super(GzipDisk, self).fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            # 创建一个BytesIO对象
            str_io = BytesIO(value)
            # 创建一个gzip文件对象
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            # 创建一个BytesIO对象用于存储解压缩后的数据
            read_csio = BytesIO()

            # 分块读取并解压缩数据
            while True:
                uncompressed_data = gz_file.read(2 ** 30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break
            # 获取解压缩后的数据
            value = read_csio.getvalue()
        return value

def getCache(scope_str):
    """
    获取缓存对象
    :param scope_str: 缓存范围
    :return: 缓存对象
    """
    return FanoutCache('data-unversioned/cache/' + scope_str,
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11,
                       # disk_min_file_size=2**20,
                       )