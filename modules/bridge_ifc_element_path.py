from utils.utils_control_logger import control_logger as logger
import ifcopenshell
import ifcopenshell.util.element
import json
import os
import sqlite3
import pandas as pd
import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from itertools import combinations

def process_bridge_part(part_obj):
    """递归处理桥梁构件及其子构件"""
    logger.trace(f"处理桥梁构件: {part_obj.GlobalId}")
    part_data = {
        'type': 'IfcBridgePart',
        'name': getattr(part_obj, 'Name', '未命名构件'),
        'guid': part_obj.GlobalId,
        'sub_parts': {},
        'elements': {}
    }
    
    # 处理子构件
    for rel in part_obj.IsDecomposedBy:
        for sub_obj in rel.RelatedObjects:
            if sub_obj.is_a('IfcBridgePart'):
                part_data['sub_parts'][sub_obj.GlobalId] = process_bridge_part(sub_obj)
    
    return part_data

def process_bridge(bridge_obj):
    """递归处理桥梁及其子桥梁结构"""
    logger.trace(f"处理桥梁: {bridge_obj.GlobalId}")
    bridge_data = {
        'type': 'IfcBridge',
        'name': getattr(bridge_obj, 'Name', '未命名桥梁'),
        'guid': bridge_obj.GlobalId,
        'sub_bridges': {},
        'parts': {}
    }
    
    # 处理子桥梁和构件
    for rel in bridge_obj.IsDecomposedBy:
        for sub_obj in rel.RelatedObjects:
            if sub_obj.is_a('IfcBridge'):
                bridge_data['sub_bridges'][sub_obj.GlobalId] = process_bridge(sub_obj)
            elif sub_obj.is_a('IfcBridgePart'):
                bridge_data['parts'][sub_obj.GlobalId] = process_bridge_part(sub_obj)
    
    return bridge_data

def get_project_hierarchy(file):
    logger.trace("获取项目层次结构...")
    project_hierarchy = {}
    
    # 获取项目
    projects = file.by_type("IfcProject")
    for project in projects:
        logger.trace(f"处理项目: {project.GlobalId}")
        project_hierarchy[project.GlobalId] = {
            'type': 'IfcProject',
            'name': getattr(project, 'Name', '未命名项目'),
            'guid': project.GlobalId,
            'roads': {}
        }
        
        # 获取道路
        for rel in project.IsDecomposedBy:
            for road in rel.RelatedObjects:
                if road.is_a('IfcRoad'):
                    logger.trace(f"处理道路: {road.GlobalId}")
                    project_hierarchy[project.GlobalId]['roads'][road.GlobalId] = {
                        'type': 'IfcRoad',
                        'name': getattr(road, 'Name', '未命名道路'),
                        'guid': road.GlobalId,
                        'bridges': {}
                    }
                    
                    # 获取桥梁并递归处理
                    for road_rel in road.IsDecomposedBy:
                        for bridge in road_rel.RelatedObjects:
                            if bridge.is_a('IfcBridge'):
                                project_hierarchy[project.GlobalId]['roads'][road.GlobalId]['bridges'][bridge.GlobalId] = process_bridge(bridge)
    
    return project_hierarchy

def get_civil_elements(file, project_hierarchy):
    logger.trace("获取土木工程元素...")
    
    def process_part_elements(part_data):
        """递归处理构件的元素和子构件"""
        part_obj = file.by_guid(part_data['guid'])
        if part_obj:
            # 处理当前构件的元素
            for rel in part_obj.ContainsElements:
                for element in rel.RelatedElements:
                    if element.is_a('IfcCivilElement'):
                        logger.trace(f"处理土木工程元素: {element.GlobalId}")
                        part_data['elements'][element.GlobalId] = {
                            'type': 'IfcCivilElement',
                            'name': getattr(element, 'Name', '未命名元素'),
                            'guid': element.GlobalId,
                            'properties': {}
                        }
                        
                        # 获取元素属性
                        for pset_rel in element.IsDefinedBy:
                            if hasattr(pset_rel, 'RelatingPropertyDefinition'):
                                pset = pset_rel.RelatingPropertyDefinition
                                if hasattr(pset, 'HasProperties'):
                                    for prop in pset.HasProperties:
                                        if hasattr(prop, 'Name') and hasattr(prop, 'NominalValue'):
                                            part_data['elements'][element.GlobalId]['properties'][prop.Name] = str(prop.NominalValue.wrappedValue)
        
        # 递归处理子构件
        for sub_part_id, sub_part_data in part_data['sub_parts'].items():
            process_part_elements(sub_part_data)
    
    def process_bridge_structure(bridge_data):
        """递归处理桥梁结构"""
        # 处理当前桥梁的构件
        for part_id, part_data in bridge_data['parts'].items():
            process_part_elements(part_data)
        
        # 递归处理子桥梁
        for sub_bridge_id, sub_bridge_data in bridge_data['sub_bridges'].items():
            process_bridge_structure(sub_bridge_data)
    
    for project_id, project in project_hierarchy.items():
        for road_id, road in project['roads'].items():
            for bridge_id, bridge in road['bridges'].items():
                process_bridge_structure(bridge)
    
    return project_hierarchy

def traverse_json(obj, path=""):
    """遍历JSON对象并生成路径"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                yield from traverse_json(v, new_path)
            else:
                yield f"{new_path}:{v}"
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            new_path = f"{path}[{index}]"
            if isinstance(item, (dict, list)):
                yield from traverse_json(item, new_path)
            else:
                yield f"{new_path}:{item}"

def process_nested_structure(config, nested_structure):
    """处理嵌套结构并保存到数据库"""
    logger.trace("开始处理嵌套结构")
    
    try:
        # 构建数据库路径
        db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
        
        # 确保数据库目录存在
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.trace(f"创建数据库目录: {db_dir}")

        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 创建表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS 结果_元素路径 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path_value TEXT NOT NULL
            )
        """)
        
        # 创建嵌套结构表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS 结果_嵌套结构 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                structure_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 清空现有数据
        cursor.execute("DELETE FROM 结果_元素路径")
        cursor.execute("DELETE FROM 结果_嵌套结构")

        # 插入新的路径数据
        for path_value in traverse_json(nested_structure):
            cursor.execute("INSERT INTO 结果_元素路径 (path_value) VALUES (?)", (path_value,))
        
        # 将完整的嵌套结构保存为JSON
        structure_json = json.dumps(nested_structure, ensure_ascii=False)
        cursor.execute("INSERT INTO 结果_嵌套结构 (structure_json) VALUES (?)", (structure_json,))

        conn.commit()
        conn.close()
        logger.trace(f"成功将路径数据和嵌套结构保存到数据库: {db_path}")
    except Exception as e:
        logger.trace(f"保存结果到数据库失败: {e}")

def main(config):
    logger.trace("开始处理IFC文件...")
    try:
        file = ifcopenshell.open(config['ifc']['model_path'])
        
        # 构建完整的项目层次结构
        project_hierarchy = get_project_hierarchy(file)
        project_hierarchy = get_civil_elements(file, project_hierarchy)
        
        # 处理并保存到数据库
        process_nested_structure(config, project_hierarchy)

    except Exception as e:
        logger.trace(f"发生错误: {e}")