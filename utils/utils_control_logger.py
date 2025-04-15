import os
import datetime
from loguru import logger
import sys

def setup_control_logger():
    log_dir = "Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
    # 创建一个新的logger实例
    control_logger = logger.bind(name="control_logger")
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    # 添加文件处理器
    logger.add(
        sys.stderr,
        format=log_format,
        colorize=True,
        enqueue=True
    )
    logger.add(log_file, rotation="1 day", retention="7 days", format=log_format)
    control_logger.level("TRACE")
    return control_logger

# 初始化全局control logger
control_logger = setup_control_logger()

