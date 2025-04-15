"""
IFC文件相关操作：提取IFC实体和IFC属性集
"""

import datetime
import sqlite3
from pathlib import Path
import ifcopenshell
from utils.utils_control_logger import control_logger as logger
from modules.database import compute_file_hash

def extract_ifc_entities(config):
    """提取IFC模型中的实体信息并保存到数据库"""
    logger.info("开始提取IFC模型实体信息...")
    
    ifc_file_path = Path(config["ifc"]["model_path"])
    if not ifc_file_path.exists():
        logger.error(f"IFC文件不存在: {ifc_file_path}")
        return
    
    try:
        # 连接数据库
        conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
        cursor = conn.cursor()
        
        # 计算当前IFC文件的哈希值
        current_hash = compute_file_hash(ifc_file_path)
        
        # 检查是否已处理过该文件
        cursor.execute("""
            SELECT hash_value, last_processed 
            FROM IFC文件哈希 
            WHERE file_path = ?
        """, (str(ifc_file_path),))
        result = cursor.fetchone()
        
        if result and result[0] == current_hash:
            logger.info("IFC文件未发生变化，跳过实体提取步骤")
            conn.close()
            return
            
        # 更新或插入文件哈希记录
        cursor.execute("""
            INSERT OR REPLACE INTO IFC文件哈希 
            (file_path, hash_value, last_processed, properties_extracted)
            VALUES (?, ?, ?, ?)
        """, (str(ifc_file_path), current_hash, datetime.datetime.now(), 0))
        conn.commit()
        
        # 删除表中所有记录
        delete_sql = "DELETE FROM IFC实体"
        cursor.execute(delete_sql)
        logger.info("已清除IFC实体表中的所有记录")
        
        # 加载IFC文件
        ifc_model = ifcopenshell.open(str(ifc_file_path))
        
        # 创建IFC实体表
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS IFC实体 (
            {config['database']['tables'][3]['schema']}
        )
        """
        cursor.execute(create_table_sql)
        
        # 准备插入语句
        insert_sql = """
        INSERT OR REPLACE INTO IFC实体 (guid, ifc_type, name, description)
        VALUES (?, ?, ?, ?)
        """
        
        # 遍历所有IFC实体，只保留 'IfcElement' 和 'IfcSpatialStructureElement' 的子类，并排除 IfcPropertySet
        for entity in ifc_model:
            try:
                # 排除 IfcPropertySet 实体以及非目标子类
                if entity.is_a() == "IfcPropertySet" or not (entity.is_a("IfcElement") or entity.is_a("IfcSpatialStructureElement")):
                    continue
                guid = entity.GlobalId if hasattr(entity, 'GlobalId') else None
                ifc_type = entity.is_a()
                name = entity.Name if hasattr(entity, 'Name') else None
                description = entity.Description if hasattr(entity, 'Description') else None

                if guid:  # 只保存有GUID的实体
                    cursor.execute(insert_sql, (guid, ifc_type, name, description))

            except Exception as e:
                logger.warning(f"处理实体时出错: {str(e)}")
                continue
        
        conn.commit()
        logger.success("IFC实体信息已成功保存到数据库")
        
    except Exception as e:
        logger.error(f"处理IFC文件时发生错误: {str(e)}")
    finally:
        conn.close()

def extract_ifc_properties(config):
    """提取IFC实体的所有属性集并保存到数据库"""
    logger.info("开始提取IFC实体的所有属性集...")
    
    ifc_file_path = Path(config["ifc"]["model_path"])
    if not ifc_file_path.exists():
        logger.error(f"IFC文件不存在: {ifc_file_path}")
        return
    
    try:
        # 连接数据库
        conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
        cursor = conn.cursor()
        
        # 计算当前IFC文件的哈希值
        current_hash = compute_file_hash(ifc_file_path)
        
        # 检查是否已处理过该文件
        cursor.execute("""
            SELECT hash_value, properties_extracted 
            FROM IFC文件哈希 
            WHERE file_path = ?
        """, (str(ifc_file_path),))
        result = cursor.fetchone()
        
        if result and result[0] == current_hash and result[1]:
            logger.info("IFC文件未发生变化且属性已提取，跳过属性提取步骤")
            conn.close()
            return
        
        # 更新或插入文件哈希记录
        cursor.execute("""
            INSERT OR REPLACE INTO IFC文件哈希 
            (file_path, hash_value, last_processed, properties_extracted)
            VALUES (?, ?, ?, 0)
        """, (str(ifc_file_path), current_hash, datetime.datetime.now()))
        conn.commit()
        
        # 创建IFC属性集表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS IFC属性集 (
            guid TEXT,
            property_set_name TEXT,
            property_name TEXT,
            property_value TEXT,
            PRIMARY KEY (guid, property_set_name, property_name),
            FOREIGN KEY (guid) REFERENCES IFC实体(guid)
        )
        """
        cursor.execute(create_table_sql)
        
        # 清空表中所有记录
        logger.info("清空IFC属性集表中的所有记录...")
        cursor.execute("DELETE FROM IFC属性集")
        conn.commit()
        
        # 加载IFC文件
        ifc_model = ifcopenshell.open(str(ifc_file_path))
        
        # 准备插入语句
        insert_sql = """
        INSERT OR REPLACE INTO IFC属性集 (guid, property_set_name, property_name, property_value)
        VALUES (?, ?, ?, ?)
        """
        
        # 遍历所有IFC实体，只处理 'IfcElement' 和 'IfcSpatialStructureElement' 的子类
        for entity in ifc_model:
            try:
                # 只处理符合条件的实体
                if not (entity.is_a("IfcElement") or entity.is_a("IfcSpatialStructureElement")):
                    continue
                guid = entity.GlobalId if hasattr(entity, 'GlobalId') else None
                if not guid:
                    continue

                # 遍历实体的所有属性集
                if hasattr(entity, 'IsDefinedBy'):
                    for relationship in entity.IsDefinedBy:
                        if relationship.is_a('IfcRelDefinesByProperties'):
                            property_set = relationship.RelatingPropertyDefinition
                            if property_set.is_a('IfcPropertySet'):
                                property_set_name = property_set.Name if hasattr(property_set, 'Name') else None
                                if not property_set_name:
                                    continue

                                # 遍历属性集中的所有属性
                                for property_item in property_set.HasProperties:
                                    if hasattr(property_item, 'Name') and hasattr(property_item, 'NominalValue'):
                                        property_name = property_item.Name
                                        property_value = str(property_item.NominalValue)
                                        cursor.execute(insert_sql, (guid, property_set_name, property_name, property_value))

            except Exception as e:
                logger.warning(f"处理实体 {guid} 的属性集时出错: {str(e)}")
                continue
        
        conn.commit()
        logger.success("IFC实体属性集信息已成功保存到数据库")
        
        # 提取完成后更新properties_extracted标志
        cursor.execute("""
            UPDATE IFC文件哈希 
            SET properties_extracted = 1 
            WHERE file_path = ?
        """, (str(ifc_file_path),))
        conn.commit()
        
    except Exception as e:
        logger.error(f"处理IFC文件时发生错误: {str(e)}")
    finally:
        conn.close() 