import logging.handlers

# 获取根日志记录器
root_logger = logging.getLogger()
# 设置根日志记录的级别
# 日志记录器的层级：DEBUG < INFO < WARNING < ERROR < CRITICAL
root_logger.setLevel(logging.INFO)

# 删除其他库的日志跟记录器
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)
# 配置日志信息的格式
logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

# 创建一个流处理程序，用于将日志输出到控制台
streamHandler = logging.StreamHandler()
# 设置流处理程序的日志格式
streamHandler.setFormatter(formatter)
# 设置控制台日志记录器的记录级别
streamHandler.setLevel(logging.INFO)
# 把控制台日志记录器添加到根日日志记录器中
root_logger.addHandler(streamHandler)