import sqlite3
import pandas as pd
import json
import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from utils.utils_control_logger import control_logger as logger
import re
from collections import defaultdict
import asyncio
import utils.utils_chat as utils_chat
import json_repair
def calculate_rule_element_groups(config):
    """
    计算规范元素组：按规范实体类型分组后进行组合计算
    """
    logger.info("开始计算规范元素组...")
    
    # 连接数据库
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    cursor = conn.cursor()
    
    # 创建规范元素组结果表
    create_groups_table = """
    CREATE TABLE IF NOT EXISTS 结果_5_规范元素组 (
        规则序号 INTEGER,
        组序号 INTEGER,
        规范实体组 TEXT,
        IFC实体组 TEXT,
        路径组 TEXT,
        计算时间 TIMESTAMP,
        PRIMARY KEY (规则序号, 组序号),
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
    )
    """
    cursor.execute(create_groups_table)
    cursor.execute("DELETE FROM 结果_5_规范元素组")
    
    # 获取所有规范的实体对齐结果
    alignment_query = """
    SELECT 规则序号, 规范实体文本, 规范实体类型, ifc_guid, ifc_entity_with_type 
    FROM 结果_4_实体对齐
    ORDER BY 规则序号
    """
    alignments = pd.read_sql(alignment_query, conn)

    # 获取所有规范类型
    rule_type_query = """
    SELECT 规则序号,识别类型
    FROM 结果_1_规范类型识别
    """
    rule_types = pd.read_sql(rule_type_query, conn)
    rule_types_dict = dict(zip(rule_types['规则序号'], rule_types['识别类型']))
    
    # 获取所有元素路径
    path_query = "SELECT path_value FROM 结果_元素路径"
    paths = pd.read_sql(path_query, conn)['path_value'].tolist()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        main_task = progress.add_task(
            "[cyan]处理规范规则...", 
            total=len(alignments.groupby('规则序号'))
        )
        
        for rule_id, rule_alignments in alignments.groupby('规则序号'):
            logger.trace(f"处理规则 {rule_id} 的元素组...")
            progress.update(main_task, advance=1, description=f"[cyan]处理规则 {rule_id}...")
            
            # 获取当前规则的类型
            rule_type = rule_types_dict.get(rule_id, "")
            is_containment_rule = rule_type == "包含关系类"
            
            if is_containment_rule:
                # 对数量统计类规范使用特殊分组逻辑
                calculate_entity_count_group(rule_id, conn, config)
            else:
                # 按规范实体类型分组
                type_groups = {}
                for _, row in rule_alignments.iterrows():
                    entity_type = row['规范实体类型']
                    if entity_type not in type_groups:
                        type_groups[entity_type] = []
                    type_groups[entity_type].append(row.to_dict())
            
                # 生成组合
                group_number = 0
                process_task = progress.add_task(
                    f"[green]处理规则 {rule_id} 的组合...",
                    total=len(next(iter(type_groups.values())))  # 使用第一组的长度作为进度条基准
                )
                
                # 递归函数来生成组合
                def generate_combinations(current_combo, remaining_types):
                    nonlocal group_number
                    
                    if not remaining_types:  # 已经处理完所有类型组
                        # 获取组合中的GUID列表
                        guids = [item['ifc_guid'] for item in current_combo]
                        
                        # 在路径中查找这些GUID的关系
                        matching_path = None
                        for path in paths:
                            if all(guid in path for guid in guids):
                                matching_path = path
                                break  # 找到第一条匹配的路径就停止
                        
                        # 如果找到匹配的路径，保存这个组合
                        if matching_path:
                            group_number += 1
                            
                            norm_entities = [{
                                "text": item['规范实体文本'],
                                "type": item['规范实体类型']
                            } for item in current_combo]
                            
                            ifc_entities = [{
                                "guid": item['ifc_guid'],
                                "type": item['ifc_entity_with_type']
                            } for item in current_combo]
                            
                            try:
                                cursor.execute("""
                                    INSERT INTO 结果_5_规范元素组
                                    (规则序号, 组序号, 规范实体组, IFC实体组, 路径组, 计算时间)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    rule_id,
                                    group_number,
                                    json.dumps(norm_entities, ensure_ascii=False),
                                    json.dumps(ifc_entities, ensure_ascii=False),
                                    json.dumps([matching_path], ensure_ascii=False),
                                    datetime.datetime.now()
                                ))
                                conn.commit()
                                logger.trace(f"规则 {rule_id} 的第 {group_number} 组已保存")
                            except Exception as e:
                                logger.error(f"保存规则 {rule_id} 的元素组时出错: {str(e)}")
                        return
                    
                    current_type = remaining_types[0]
                    next_types = remaining_types[1:]
                    
                    # 遍历当前类型组的所有实体
                    for entity in type_groups[current_type]:
                        progress.update(process_task, advance=1)
                        generate_combinations(current_combo + [entity], next_types)
                
                # 开始递归生成组合
                generate_combinations([], list(type_groups.keys()))
                
                progress.remove_task(process_task)
    
    conn.close()
    logger.success("规范元素组计算完成")

def calculate_entity_count_group(rule_id, conn, config):
    """
    计算指定规则的实体数量，确保所有需要统计的实体都加入到匹配的路径组，而不是单独生成组。

    逻辑：
    1. 先生成 **不含 count entity** 的组合。
    2. 计算这些组合的路径。
    3. 找到 **与路径一致的 count entity**，合并到组合里。
    4. 存入数据库。

    参数:
    - rule_id: 规则的唯一标识符
    - conn: 数据库连接对象
    - config: 配置文件内容
    """
    cursor = conn.cursor()

    # 获取规则内容
    cursor.execute("SELECT 内容 FROM 输入_规范列表 WHERE 规则序号 = ?", (rule_id,))
    rule = cursor.fetchone()
    rule_text = rule[0] if rule else ""

    # 获取该规则的所有规范实体
    cursor.execute("SELECT 实体文本 FROM 结果_2_规范实体识别 WHERE 规则序号 = ?", (rule_id,))
    norm_entities = [row[0] for row in cursor.fetchall()]

    # 生成 prompt 以识别需要统计数量的实体
    prompt = f"""请分析规范中需要统计数量的实体，并返回符合要求的结果，格式为JSON。
    
    规则描述：{rule_text}
    规范实体列表：{', '.join(norm_entities)}
    
    请注意，您需要识别出在规范中明确要求统计数量的实体，并将其以JSON格式返回，格式为：{{"entity": "实体名称"}}。"""

    # 异步调用 OpenAI API
    result = asyncio.run(utils_chat.process_single_prompt_special(prompt, config))
    
    # 解析 LLM 返回的 JSON 数据
    try:
        count_entity_name = json_repair.loads(result)
        # 确保返回的是字典格式，如果不是则进行转换
        if isinstance(count_entity_name, list) and len(count_entity_name) > 0:
            count_entity_name = {"entity": count_entity_name[0]}
        elif not isinstance(count_entity_name, dict):
            count_entity_name = {"entity": str(count_entity_name)}
        
        # 确保字典中有 "entity" 键
        if "entity" not in count_entity_name:
            count_entity_name = {"entity": ""}
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"解析 OpenAI 结果失败: {str(e)}")
        count_entity_name = {"entity": ""}

    logger.trace(f"规则 {rule_id} 需要统计数量的实体: {count_entity_name}")

    # 获取所有对齐的实体
    alignment_query = """
    SELECT 规范实体文本, 规范实体类型, ifc_guid, ifc_entity_with_type 
    FROM 结果_4_实体对齐
    WHERE 规则序号 = ?
    """
    alignments = pd.read_sql(alignment_query, conn, params=(rule_id,))

    # 获取所有元素路径
    path_query = "SELECT path_value FROM 结果_元素路径"
    paths = pd.read_sql(path_query, conn)['path_value'].tolist()

    # 按规范实体类型分组
    type_groups = defaultdict(list)
    count_entities_group = []  # 存储需要统计数量的实体

    for _, row in alignments.iterrows():
        entity_text = row['规范实体文本']
        entity_type = row['规范实体类型']

        if entity_text == count_entity_name.get("entity", ""):
            count_entities_group.append(row.to_dict())  # 需要统计数量的实体
        else:
            type_groups[entity_type].append(row.to_dict())

    # **步骤 1：生成不含 count entity 的组合**
    group_number = 0
    valid_groups = []

    def generate_combinations(current_combo, remaining_types):
        """
        递归生成组合（不包含 count entity）
        """
        nonlocal group_number

        if not remaining_types:  # 处理完所有类型组
            guids = [item['ifc_guid'] for item in current_combo]

            # 找到匹配的路径
            matching_paths = [path for path in paths if all(guid in path for guid in guids)]

            if matching_paths:  # 如果路径匹配
                valid_groups.append((current_combo, matching_paths))

            return

        current_type = remaining_types[0]
        next_types = remaining_types[1:]

        # 遍历当前类型组的所有实体
        for entity in type_groups[current_type]:
            generate_combinations(current_combo + [entity], next_types)

    # 递归生成组合
    generate_combinations([], list(type_groups.keys()))

    # **步骤 2：遍历组合，匹配路径，找到 count entity**
    final_groups = []

    for group, matching_paths in valid_groups:
        group_guids = {item['ifc_guid'] for item in group}

        # 找到与当前路径匹配的 count entity
        related_count_entities = [
            entity for entity in count_entities_group
            if any(entity['ifc_guid'] in path for path in matching_paths)
        ]

        # **步骤 3：合并 count entity**
        if related_count_entities:
            group.extend(related_count_entities)

        # 规范实体组
        norm_entities = [{
            "text": item['规范实体文本'],
            "type": item['规范实体类型']
        } for item in group]

        # IFC 实体组
        ifc_entities = [{
            "guid": item['ifc_guid'],
            "type": item['ifc_entity_with_type']
        } for item in group]

        final_groups.append((norm_entities, ifc_entities, matching_paths))

    # **步骤 4：存入数据库**
    for norm_entities, ifc_entities, matching_paths in final_groups:
        group_number += 1

        try:
            cursor.execute("""
                INSERT INTO 结果_5_规范元素组
                (规则序号, 组序号, 规范实体组, IFC实体组, 路径组, 计算时间)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rule_id,
                group_number,
                json.dumps(norm_entities, ensure_ascii=False),
                json.dumps(ifc_entities, ensure_ascii=False),
                json.dumps(matching_paths, ensure_ascii=False),
                datetime.datetime.now()
            ))
            conn.commit()
            logger.trace(f"规则 {rule_id} 的第 {group_number} 组已保存")
        except Exception as e:
            logger.error(f"保存规则 {rule_id} 的元素组时出错: {str(e)}")

    logger.trace(f"规则 {rule_id} 的实体数量计算完成")

