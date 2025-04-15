"""
数据库初始化及辅助函数模块
"""

from pathlib import Path
import hashlib
import sqlite3
import pandas as pd
from utils.utils_control_logger import control_logger as logger

def compute_file_hash(file_path):
    """计算文件的SHA-256哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 每次读取4MB数据
        for byte_block in iter(lambda: f.read(4096 * 1024), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def init_database(config):
    """初始化数据库和表"""
    logger.info("开始初始化数据库...")
    
    # 确保数据库目录存在
    db_path = Path(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表并导入数据
    for table in config["database"]["tables"]:
        logger.info(f"正在初始化表 {table['name']} ...")
        
        # 创建表
        create_sql = f"CREATE TABLE IF NOT EXISTS {table['name']} ({table['schema']})"
        cursor.execute(create_sql)
        
        # 只有存在csv_path的表才需要导入CSV数据
        if "csv_path" in table:
            try:
                df = pd.read_csv(table["csv_path"])
                
                # 确保列名匹配
                df.columns = table["columns"]
                
                # 将数据导入表
                df.to_sql(table['name'], conn, if_exists="replace", index=False)
                logger.success(f"表 {table['name']} 数据导入成功")
                
            except Exception as e:
                logger.error(f"导入表 {table['name']} 数据失败: {str(e)}")
    
    # 添加IFC文件哈希表
    create_ifc_hash_table = """
    CREATE TABLE IF NOT EXISTS IFC文件哈希 (
        file_path TEXT PRIMARY KEY,
        hash_value TEXT,
        last_processed TIMESTAMP,
        properties_extracted BOOLEAN DEFAULT 0
    )
    """
    cursor.execute(create_ifc_hash_table)
    
    conn.commit()
    conn.close() 