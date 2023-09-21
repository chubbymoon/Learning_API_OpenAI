"""程序日志记录器"""

import logging
import os

# 设置日志文件路径, 为空将日志输出到控制台
# log_file_path = os.path.join('.', 'log.log')
log_file_path = ''

# 配置日志输出等级
# log_level = logging.DEBUG
# log_level = logging.INFO
log_level = logging.WARNING


try:
    # 配置日志输出格式
    handler = logging.FileHandler(log_file_path, encoding='utf-8')
# 处理路径不存在
except FileNotFoundError:
    log_file_location = os.path.dirname(log_file_path)
    # 创建路径
    if not os.path.exists(log_file_location):
        os.makedirs(log_file_location)
    # 创建文件
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', encoding='utf-8') as f:
            pass
    # 重新配置日志输出格式
    handler = logging.FileHandler(log_file_path, encoding='utf-8')
except Exception as e:
    if not log_file_path:
        # 当日志路径 `log_file_path` 不存在时将日志输出到控制台
        handler = logging.StreamHandler()
    else:
        print(e)
        raise

logging.basicConfig(
    level=log_level,
    handlers=[handler],
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s] %(filename)s - %(lineno)d >>> %(message)s"
)

# 创建记录日志的对象 logger
logger = logging.getLogger(__name__)

# 使用 logger 使用示例
logger.debug("已开启调试模式! ")

