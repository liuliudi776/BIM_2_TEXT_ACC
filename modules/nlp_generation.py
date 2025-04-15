"""
自然语言描述生成与合规性审查
"""

import datetime
import sqlite3
import pandas as pd
import json
from pathlib import Path
import tomli
import json_repair
from tqdm import tqdm
import asyncio
from utils.utils_control_logger import control_logger as logger
import utils.utils_chat as utils_chat
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from diskcache import Cache
import hashlib
import re
from utils.utils_cache_manager import _clear_cache_for_prompts

def generate_natural_language_descriptions(config):
    """使用大模型将属性集信息转换为自然语言描述"""
    logger.info("开始使用大模型生成自然语言描述...")
    
    # 读取额外信息
    add_info_path = Path(config["add_info_file"]["path"])
    try:
        with open(add_info_path, "rb") as f:
            additional_info = tomli.load(f)
    except Exception as e:
        logger.warning(f"无法加载额外信息文件: {str(e)}")
        additional_info = {}
    
    # 获取全局信息
    global_info = additional_info.get("global", {})
    global_info_text = ""
    if global_info:
        global_info_text = "\n全局信息:\n"
        for key, value in global_info.items():
            global_info_text += f"- {key}: {value}\n"
    
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    cursor = conn.cursor()
    
    # 创建自然语言描述表（增加适用性字段）
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS 结果_8_属性自然语言描述 (
        规则序号 INTEGER,
        组序号 INTEGER,
        description TEXT,
        原始属性集 TEXT,
        PRIMARY KEY (规则序号, 组序号) ON CONFLICT REPLACE,
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
    )
    """
    cursor.execute(create_table_sql)
    cursor.execute("DELETE FROM 结果_8_属性自然语言描述")
    conn.commit()
    
    # 获取相关属性数据和规范内容
    query = """
    SELECT 
        reg.规则序号,
        reg.组序号,
        reg.规范实体组,
        reg.IFC实体组,
        reg.路径组,
        r.内容 as rule_content
    FROM 结果_5_规范元素组 reg
    JOIN 输入_规范列表 r ON reg.规则序号 = r.规则序号
    """
    relevant_groups = pd.read_sql(query, conn)

    # 新增：查询 `结果_6_规范属性选择策略` 数据
    strategy_query = """
    SELECT 规则序号, property_set_name, property_name, 优先级, 策略说明
    FROM 结果_6_规范属性选择策略
    """
    property_strategies = pd.read_sql(strategy_query, conn)
    
    processed_rules = set()
    nld_prompts = []
    groups_info = []  # 存储 (rule_id, group_id)
    
    for _, row in relevant_groups.iterrows():
        rule_id = row['规则序号']
        group_id = row['组序号']
        
        # 移除适用性判断
        # if rule_id in processed_rules:
        #     logger.info(f"跳过规则 {rule_id}，因为该规则已被标记为不适用或信息不足")
        #     continue
            
        norm_entities = json.loads(row['规范实体组'])
        ifc_entities = json.loads(row['IFC实体组'])
        paths = json.loads(row['路径组'])
        rule_content = row['rule_content']
        
        # 获取每个IFC实体的属性信息
        properties_text = ""
        
        
        for entity in ifc_entities:
            entity_props_query = """
            SELECT property_set_name, property_name, property_value
            FROM 结果_7_相关属性
            WHERE ifc_guid = ?
            """
            entity_props = pd.read_sql(entity_props_query, conn, params=(entity['guid'],))
            unique_properties = set()  # 用于存储唯一的属性信息
            if not entity_props.empty:
                properties_text += f"\n实体 {entity['guid']} ({entity['type']}) 的属性:\n"
                for _, prop in entity_props.iterrows():
                    # 创建属性信息字符串
                    prop_info = f"属性集: {prop['property_set_name']}, 属性: {prop['property_name']}, 值: {prop['property_value']}"
                    # 只添加未出现过的属性信息
                    if prop_info not in unique_properties:
                        unique_properties.add(prop_info)
                        properties_text += prop_info + "\n"
        
        rule_add_info = additional_info.get("rules", {}).get(str(rule_id), {})
        rule_info_text = ""
        if rule_add_info:
            rule_info_text = "\n规则特定信息:\n" + "\n".join([f"- {k}: {v}" for k, v in rule_add_info.items()])

        knowledge_path = config["knowledge"]["path"]
        with open(knowledge_path, "rb") as f:
            knowledge_base = tomli.load(f)
        knowledge = knowledge_base.get("global", {})

        # 获取规范类型
        conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 识别类型 FROM 结果_1_规范类型识别 WHERE 规则序号 = ?", (rule_id,))
        rule_type = cursor.fetchone()[0]
        
        prompt = f"""请分析以下构件组的属性信息，并生成自然语言描述。请考虑所有提供的信息。
        如果是包含关系类则着重描述相关构件的包含关系以及数量，并明确相关构件的id.
        如果是属性约束类则着重描述相关构件的相关属性.
        注意，不要给出任何合规性判断，仅仅是忠实的描述构件组的属性信息，不进行任何添加。

领域知识：
{knowledge}

规范类型：
{rule_type}

规范要求：
{rule_content}

构件组信息：
规范实体: {json.dumps(norm_entities, ensure_ascii=False)}
IFC实体: {json.dumps(ifc_entities, ensure_ascii=False)}
实体路径: {str(json.dumps(paths, ensure_ascii=False))[:100]}

属性信息：
{properties_text}

{global_info_text}{rule_info_text}

请返回以下格式的结果（使用JSON格式）：
{{
    "description": "基于实体组将实体组信息完整地拼接成自然语言描述",
}}"""


        # 检查是否包含策略中指定的所有属性
        missing_properties = []
        strategy_for_rule = property_strategies[property_strategies["规则序号"] == rule_id]
        
        # 从所有实体的属性中提取属性集和属性名
        all_entity_props = []
        for entity in ifc_entities:
            entity_props_query = """
            SELECT property_set_name, property_name
            FROM 结果_7_相关属性
            WHERE ifc_guid = ?
            """
            entity_props = pd.read_sql(entity_props_query, conn, params=(entity['guid'],))
            all_entity_props.append(entity_props)
        
        # 合并所有实体的属性
        if all_entity_props:
            combined_props = pd.concat(all_entity_props)
            
            for _, strategy in strategy_for_rule.iterrows():
                # 检查是否存在匹配属性名
                matching_props = combined_props[
                    (combined_props['property_name'].str.contains(strategy['property_name']))
                ]
                
                if matching_props.empty:
                    missing_properties.append(f"{strategy['property_name']}")

        # 如果存在缺失属性，则将缺失信息添加到prompt中，而不是直接标记为信息不足
        if missing_properties:
            logger.trace(f"规则序号 {rule_id}, 组序号 {group_id} 缺失属性：{missing_properties}")
            missing_properties_text = f"\n注意：内部缺失以下属性：{', '.join(missing_properties)}\n请基于现有信息进行分析，并在描述中说明缺失的属性可能对分析结果的影响。"
            prompt += missing_properties_text
            nld_prompts.append(prompt)
            groups_info.append((rule_id, group_id, properties_text))
        else:
            nld_prompts.append(prompt)
            groups_info.append((rule_id, group_id, properties_text))
    
    # 直接调用批处理，不使用额外的进度条
    nld_responses = asyncio.run(utils_chat.batch_process_prompts(nld_prompts, config, batch_size=10000))

    # 存储失败的项目
    failed_items = []
    
    for (rule_id, group_id, properties_text), nld_response in zip(groups_info, nld_responses):
        try:
            result = json.loads(json_repair.repair_json(nld_response))
            
            cursor.execute("""
                INSERT INTO 结果_8_属性自然语言描述 
                (规则序号, 组序号, description, 原始属性集)
                VALUES (?, ?, ?, ?)
            """, (
                int(rule_id),
                int(group_id),
                str(result['description']),
                properties_text
            ))
            conn.commit()
        except Exception as e:
            logger.warning(f"处理规则序号: {rule_id}, 组序号: {group_id} 时出错: {str(e)}")
            failed_items.append((rule_id, group_id, nld_prompts[groups_info.index((rule_id, group_id,properties_text))]))
            continue
    
    # 对失败的项目进行重试
    if failed_items:
        logger.info(f"开始重试 {len(failed_items)} 个失败的项目...")
        retry_prompts = [item[2] for item in failed_items]
        
        # 清除失败项目的缓存
        _clear_cache_for_prompts(retry_prompts, config)
        
        retry_responses = asyncio.run(utils_chat.batch_process_prompts(retry_prompts, config, batch_size=5))
        
        for (rule_id, group_id, properties_text), retry_response in zip(failed_items, retry_responses):
            try:
                result = json.loads(json_repair.repair_json(retry_response))
                cursor.execute("""
                    INSERT INTO 结果_8_属性自然语言描述 
                    (规则序号, 组序号, description, 原始属性集)
                    VALUES (?, ?, ?, ?)
                """, (
                    int(rule_id),
                    int(group_id),
                    str(result['description']),
                    properties_text
                ))
                conn.commit()
                logger.info(f"重试成功 - 规则序号: {rule_id}, 组序号: {group_id}")
            except Exception as e:
                logger.error(f"重试失败 - 规则序号: {rule_id}, 组序号: {group_id}: {str(e)}")
    
    conn.close()
    logger.success("属性集自然语言描述和规范适用性判断已完成")

def check_compliance(config):
    """将自然语言序列和规范输入给大模型，判断合规性"""
    logger.info("开始使用大模型判断合规性...")
    # 读取额外信息
    add_info_path = Path(config["add_info_file"]["path"])
    try:
        with open(add_info_path, "rb") as f:
            additional_info = tomli.load(f)
    except Exception as e:
        logger.warning(f"无法加载额外信息文件: {str(e)}")
        additional_info = {}
    
    # 获取全局信息
    global_info = additional_info.get("global", {})
    global_info_text = ""
    if global_info:
        global_info_text = "\n全局信息:\n"
        for key, value in global_info.items():
            global_info_text += f"- {key}: {value}\n"
    
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    cursor = conn.cursor()
    
    # 创建合规性审查结果表 (添加 IFC实体组 列)
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS 结果_9_合规性审查 (
        规则序号 INTEGER,
        组序号 INTEGER,
        analysis_process TEXT,
        judgment_result TEXT,
        check_time TIMESTAMP,
        IFC实体组 TEXT,
        PRIMARY KEY (规则序号, 组序号),
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
    )
    """
    cursor.execute(create_table_sql)
    
    logger.info("清空结果_9_合规性审查表中的所有记录...")
    cursor.execute("DELETE FROM 结果_9_合规性审查")
    conn.commit()
    
    # 修改查询，包含 IFC实体组 信息
    query = """
    SELECT 
        nld.规则序号,
        nld.组序号,
        nld.description,
        r.内容 as rule_content,
        reg.IFC实体组
    FROM 结果_8_属性自然语言描述 nld
    JOIN 输入_规范列表 r ON nld.规则序号 = r.规则序号
    JOIN 结果_5_规范元素组 reg ON nld.规则序号 = reg.规则序号 AND nld.组序号 = reg.组序号
    WHERE nld.description IS NOT NULL
    """
    
    results = pd.read_sql(query, conn)
    
    # ----------------- 批量处理合规性审查提示 -----------------
    compliance_prompts = []
    compliance_data = []  # 存储 (rule_id, group_id, ifc_entities)

    # 获取专业知识
    with open(config["knowledge"]["path"], "rb") as f:
        knowledge_base = tomli.load(f)
    knowledge = knowledge_base.get("global", {})
    
    for _, row in results.iterrows():
        rule_id = row['规则序号']
        group_id = row['组序号']
        description = row['description']
        rule_content = row['rule_content']
        ifc_entities = row['IFC实体组']  # 获取IFC实体组
    
        rule_add_info = additional_info.get("rules", {}).get(str(rule_id), {})
        rule_info_text = ""
        if rule_add_info:
            rule_info_text = "\n规则特定信息:\n" + "\n".join([f"- {k}: {v}" for k, v in rule_add_info.items()])
        
        prompt = f"""请对给定的规范要求和构件组描述进行严格、系统的分析，请遵循以下判断流程：

### 第一步：判断规范适用性
首先，判断规范要求是否严格适用于所描述的构件组，需关注以下要点：
- 规范明确指定的适用范围（如等级、类型、形式等）
- 构件组的实际类型、等级、用途是否完全符合规范所针对的对象
- 任何偏离规范适用范围的情况都应导致"不适用"判断

### 第二步：合规性分析（仅当规范适用时进行）
只有在确认规范完全适用于构件组的情况下，才进行合规性分析，判断构件组是否满足规范的具体要求。

请综合分析以下所有信息：

规范要求：
{rule_content}

构件组描述：
{description}

领域知识：
{knowledge.get("领域常识", [])}

{global_info_text}{rule_info_text}

注意事项：
1. 规范的适用性判断必须严格执行，不得通过类比或相似推理扩大规范的适用范围
2. 如果规范明确针对特定类型，而构件组属于不同类型，则应判定为"不适用"
3. 不要尝试通过寻找相似点或共同标准来强行适用不相关的规范
4. "不适用"不等同于"不合规"，而是表示该规范本身不应用于评判该构件组
5. 如果构件组描述中由于"缺失信息"导致无法判断是否适用，则应仔细分析构件组描述，并判定为"不适用"或"不合规"。
请严格按照以下JSON格式返回分析结果：
{{
    "analysis": "详细的分析过程，首先明确判断规范是否适用并给出充分理由，若适用则继续分析是否合规",
    "judgment": "合规" 或 "不合规" 或 "不适用"
}}"""

        compliance_prompts.append(prompt)
        compliance_data.append((rule_id, group_id, ifc_entities))
    
    # 直接调用批处理，不使用额外的进度条
    compliance_responses = asyncio.run(utils_chat.batch_process_prompts(compliance_prompts, config, batch_size=10000, enforce_model_type="default"))
    # ----------------- 合规性批处理结束 -----------------
    
    # 存储失败的项目
    failed_items = []
    
    for (rule_id, group_id, ifc_entities), compliance_response in zip(compliance_data, compliance_responses):
        try:
            result = json.loads(json_repair.repair_json(compliance_response))
            analysis = result['analysis']
            judgment = result['judgment']
            current_time = datetime.datetime.now()
            cursor.execute("""
                INSERT INTO 结果_9_合规性审查 
                (规则序号, 组序号, analysis_process, judgment_result, check_time, IFC实体组)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rule_id, 
                group_id,
                analysis,
                judgment,
                current_time,
                ifc_entities
            ))
            conn.commit()
            logger.trace(f"规则序号: {rule_id}, 组序号: {group_id} 的合规性审查结果已保存")
        except Exception as e:
            logger.warning(f"处理规则序号: {rule_id}, 组序号: {group_id} 时出错: {str(e)}")
            failed_items.append((rule_id, group_id, ifc_entities, compliance_prompts[compliance_data.index((rule_id, group_id, ifc_entities))]))
            continue
    
    # 对失败的项目进行重试
    if failed_items:
        logger.info(f"开始重试 {len(failed_items)} 个失败的合规性审查...")
        retry_prompts = [item[3] for item in failed_items]
        
        # 清除失败项目的缓存
        _clear_cache_for_prompts(retry_prompts, config)
        
        retry_responses = asyncio.run(utils_chat.batch_process_prompts(retry_prompts, config, batch_size=5))
        
        for (rule_id, group_id, ifc_entities, retry_prompt), retry_response in zip(failed_items, retry_responses):
            try:
                result = json.loads(json_repair.repair_json(retry_response))
                cursor.execute("""
                    INSERT INTO 结果_9_合规性审查 
                    (规则序号, 组序号, analysis_process, judgment_result, check_time, IFC实体组)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    rule_id,
                    group_id,
                    result['analysis'],
                    result['judgment'],
                    datetime.datetime.now(),
                    ifc_entities
                ))
                conn.commit()
                logger.info(f"重试成功 - 规则序号: {rule_id}, 组序号: {group_id}")
            except Exception as e:
                logger.error(f"重试失败 - 规则序号: {rule_id}, 组序号: {group_id}: {str(e)}")
    
    conn.close()
    logger.success("合规性审查完成")


