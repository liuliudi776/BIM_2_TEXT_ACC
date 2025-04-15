import os
import datetime
from utils.utils_control_logger import control_logger as logger

def setup_gpt_logger():
    log_dir = "GPT_Logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gpt_log_file = os.path.join(log_dir, f"{timestamp}.log")
    
    # 创建一个新的logger实例
    gpt_logger = logger.bind(name="gpt_logger")
    
    # 添加文件处理器
    logger.add(
        gpt_log_file,
        filter=lambda record: record["extra"].get("name") == "gpt_logger",
        level="TRACE",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="1 day",
        retention="10 days",
        enqueue=True
    )
    
    gpt_logger.level("TRACE")
    return gpt_logger

# 初始化全局GPT logger
gpt_logger = setup_gpt_logger()