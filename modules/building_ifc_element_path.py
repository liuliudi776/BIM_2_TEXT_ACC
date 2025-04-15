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
from modules.calculate_rule_element_groups import calculate_rule_element_groups

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

def get_building_storeys(file):
    logger.trace("获取建筑楼层信息...")
    building_storeys_map = {}
    buildings = file.by_type("IfcBuilding")
    for building in buildings:
        logger.trace(f"处理建筑: {building.GlobalId}")
        unique_storeys = {}
        decomposition = building.IsDecomposedBy
        if decomposition:
            for rel in decomposition:
                related_objects = rel.RelatedObjects
                for obj in related_objects:
                    if obj.is_a("IfcBuildingStorey"):
                        unique_storeys[obj.GlobalId] = {
                            'name': getattr(obj, 'Name', '未命名楼层'),
                            'guid': obj.GlobalId
                        }
                        logger.trace(f"找到楼层: {obj.GlobalId}")
        building_storeys_map[building.GlobalId] = {
            'name': getattr(building, 'Name', '未命名建筑'),
            'guid': building.GlobalId,
            'storeys': unique_storeys
        }
    return building_storeys_map

def get_contained_elements(file, element_type):
    logger.trace(f"获取类型为 {element_type} 的构件...")
    contained_elements_map = {}
    elements = file.by_type(element_type)
    for element in elements:
        logger.trace(f"处理构件: {element.GlobalId}")
        unique_elements = {}
        contained = ifcopenshell.util.element.get_contained(element)
        if contained:
            for item in contained:
                unique_elements[item.GlobalId] = {
                    'name': getattr(item, 'Name', '未命名'),
                    'guid': item.GlobalId,
                    'type': item.is_a()
                }
                logger.trace(f"包含项: {item.GlobalId}")
        contained_elements_map[element.GlobalId] = {
            'name': getattr(element, 'Name', '未命名'),
            'guid': element.GlobalId,
            'elements': unique_elements
        }
    return contained_elements_map

def get_space_elements(file):
    logger.trace("获取空间元素信息...")
    space_elements_map = {}
    spaces = file.by_type("IfcSpace")
    for space in spaces:
        logger.trace(f"处理空间: {space.GlobalId}")
        unique_elements = {}
        containment_relations = space.BoundedBy
        if containment_relations:
            for relation in containment_relations:
                related_element = relation.RelatedBuildingElement
                if related_element:
                    unique_elements[related_element.GlobalId] = {
                        'name': getattr(related_element, 'Name', '未命名'),
                        'guid': related_element.GlobalId,
                        'type': related_element.is_a()
                    }
                    logger.trace(f"找到相关元素: {related_element.GlobalId}")
        space_elements_map[space.GlobalId] = {
            'name': space.Name if space.Name else "未命名空间",
            'guid': space.GlobalId,
            'elements': unique_elements
        }
    return space_elements_map

def get_storey_spaces(file):
    logger.trace("获取楼层与空间的对应关系...")
    storey_spaces_map = {}
    storeys = file.by_type("IfcBuildingStorey")
    for storey in storeys:
        logger.trace(f"处理楼层: {storey.GlobalId}")
        unique_spaces = {}
        decomposition = getattr(storey, 'IsDecomposedBy', [])
        if decomposition and len(decomposition) > 0:
            contained = getattr(decomposition[0], 'RelatedObjects', [])
            for item in contained:
                if item.is_a("IfcSpace"):
                    unique_spaces[item.GlobalId] = {
                        'name': getattr(item, 'Name', '未命名空间'),
                        'guid': item.GlobalId
                    }
                    logger.trace(f"找到空间: {item.GlobalId}")
        storey_spaces_map[storey.GlobalId] = {
            'name': getattr(storey, 'Name', '未命名楼层'),
            'guid': storey.GlobalId,
            'spaces': unique_spaces
        }
    return storey_spaces_map

def create_nested_structure(building_data, storey_data, space_data, storey_spaces_data):
    logger.trace("创建嵌套结构...")
    nested_structure = {}

    for building_id, building_info in building_data.items():
        logger.trace(f"添加建筑: {building_id}")
        nested_structure[building_id] = {
            'type': 'IfcBuilding',
            'name': building_info['name'],
            'storeys': {}
        }

        for storey_id in building_info['storeys']:
            if storey_id in storey_data:
                logger.trace(f"添加楼层: {storey_id}")
                storey_info = storey_data[storey_id]
                nested_structure[building_id]['storeys'][storey_id] = {
                    'type': 'IfcBuildingStorey',
                    'name': storey_info['name'],
                    'spaces': {},
                    'elements': {}
                }

                if storey_id in storey_spaces_data:
                    for space_id in storey_spaces_data[storey_id]['spaces']:
                        if space_id in space_data:
                            space_info = space_data[space_id]
                            nested_structure[building_id]['storeys'][storey_id]['spaces'][space_id] = {
                                'type': 'IfcSpace',
                                'name': space_info['name'],
                                'elements': space_info['elements']
                            }

                for element_id, element_info in storey_info['elements'].items():
                    if not any(element_id in space['elements'] for space in nested_structure[building_id]['storeys'][storey_id]['spaces'].values()):
                        nested_structure[building_id]['storeys'][storey_id]['elements'][element_id] = element_info

    return nested_structure

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

        # 连接数据库（如果不存在会自动创建）
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 创建表（如果不存在）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS 结果_元素路径 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path_value TEXT NOT NULL
            )
        """)
        
        # 创建嵌套结构表（如果不存在）
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
        # 处理IFC文件并生成嵌套结构
        file = ifcopenshell.open(config['ifc']['model_path'])
        building_data = get_building_storeys(file)
        storey_data = get_contained_elements(file, "IfcBuildingStorey")
        space_data = get_space_elements(file)
        storey_spaces_data = get_storey_spaces(file)
        nested_structure = create_nested_structure(building_data, storey_data, space_data, storey_spaces_data)
        
        # 直接处理嵌套结构并保存到数据库
        process_nested_structure(config, nested_structure)

    except Exception as e:
        logger.trace(f"发生错误: {e}")
