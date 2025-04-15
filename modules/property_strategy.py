"""
属性选择策略及相关属性选择：保存策略、选择相关属性
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
import re
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from utils.utils_cache_manager import _clear_cache_for_prompts

def save_property_selection_strategy(config):
    """为每条规范批量保存属性选择策略"""
    logger.info("开始创建和保存每条规范的属性选择策略...")
    
    try:
        # 连接数据库
        conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
        cursor = conn.cursor()
        
        # 创建结果_6_规范属性选择策略表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS 结果_6_规范属性选择策略 (
            规则序号 INTEGER,
            property_set_name TEXT,
            property_name TEXT,
            优先级 INTEGER,
            策略说明 TEXT,
            PRIMARY KEY (规则序号, property_set_name, property_name),
            FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
        )
        """
        cursor.execute(create_table_sql)
        
        # 清空表中所有记录
        cursor.execute("DELETE FROM 结果_6_规范属性选择策略")
        conn.commit()
        
        # 获取规范数据和对齐结果
        query = """
        SELECT DISTINCT r.规则序号, r.内容, a.ifc_guid, a.规范实体文本
        FROM 输入_规范列表 r
        JOIN 结果_4_实体对齐 a ON r.规则序号 = a.规则序号
        """
        rules_with_entities = pd.read_sql(query, conn)
        
        # 获取领域常识
        knowledge_path = config["knowledge"]["path"]
        with open(knowledge_path, "rb") as f:
            knowledge_data = tomli.load(f)
        knowledge = knowledge_data["global"]
        
        # 准备批量处理的提示词列表
        prompts = []
        rule_property_map = {}  # 用于存储规则ID和其属性的映射
        
        # 按规则序号分组处理
        for rule_id, group in rules_with_entities.groupby('规则序号'):
            try:
                rule_content = group['内容'].iloc[0]
                
                # 按规范实体文本分组
                entity_groups = []
                for entity_text, entity_group in group.groupby('规范实体文本'):
                    guids = tuple(entity_group['ifc_guid'].unique())
                    if not guids:
                        continue
                        
                    properties_query = """
                    SELECT DISTINCT property_set_name, property_name 
                    FROM IFC属性集
                    WHERE guid IN {}
                    """.format(guids if len(guids) > 1 else f"('{guids[0]}')")
                    
                    properties = pd.read_sql(properties_query, conn)
                    
                    # 为每个规范实体文本构建属性信息
                    if not properties.empty:
                        properties_text = f"规范实体: {entity_text}\n" + "\n".join([
                            f"属性集: {row['property_set_name']}, 属性: {row['property_name']}"
                            for _, row in properties.iterrows()
                        ])
                        entity_groups.append(properties_text)
                
                if not entity_groups:
                    logger.warning(f"规则序号 {rule_id} 没有找到相关的属性信息")
                    continue

                # 合并所有规范实体的属性信息
                all_properties_text = "\n\n".join(entity_groups)

                # 构建提示模板
                prompt = f"""请分析以下规范条款，并从给定的IFC属性列表中选择最相关的属性。

规范规则序号: {rule_id}
规范内容: {rule_content}

专业知识:
{knowledge}

该规范相关实体的可用IFC属性:
{all_properties_text}

请根据以下要求输出 JSON 格式的结果：

property_set：属性集名称。
property：完整的属性名称，需包含编号和中英文全称（如存在）。
priority：优先级，使用 1-5 的等级（1 为最高）。优先级依据属性与规范条款的相关性进行选择，尽量且最多 5 个属性，注意：不得编造。
explanation：选择原因，需明确说明该属性与规范条款的关联程度。
输出格式：
[
    {{
        "property_set": "属性集名称",
        "property": "完整的属性名称",
        "priority": 优先级,
        "explanation": "选择原因"
    }}
]"""
                prompts.append(prompt)
                rule_property_map[len(prompts) - 1] = rule_id
                
            except Exception as e:
                logger.error(f"准备规则序号 {rule_id} 的提示词时出错: {str(e)}")
                continue
        
        # 批量处理所有提示词
        responses = asyncio.run(utils_chat.batch_process_prompts(prompts, config, batch_size=10000))
        
        # 收集需要重试的提示词
        retry_prompts = []
        retry_rule_map = {}
        
        # 使用rich进度条处理响应结果
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("处理属性选择策略...", total=len(responses))
            
            for idx, response in enumerate(responses):
                rule_id = rule_property_map[idx]
                try:
                    # 验证和格式化响应
                    selected_properties = validate_and_format_response(response)
                    
                    if selected_properties:
                        # 保存有效的属性策略
                        for prop in selected_properties:
                            try:
                                    cursor.execute("""
                                        INSERT INTO 结果_6_规范属性选择策略
                                        (规则序号, property_set_name, property_name, 优先级, 策略说明)
                                        VALUES (?, ?, ?, ?, ?)
                                """, (
                                    rule_id,
                                    prop['property_set'],
                                    prop['property'],
                                    prop['priority'],
                                    prop['explanation']
                                ))
                            except sqlite3.IntegrityError:
                                # 重复记录,跳过
                                pass
                        
                        conn.commit()
                        logger.trace(f"规则序号 {rule_id} 的属性选择策略已保存")
                    else:
                        logger.warning(f"规则序号 {rule_id} 的响应验证失败，将加入重试队列")
                        retry_prompts.append(prompts[idx])
                        retry_rule_map[len(retry_prompts) - 1] = rule_id
                        
                except Exception as e:
                    logger.error(f"处理规则序号 {rule_id} 的响应时出错: {str(e)}")
                    retry_prompts.append(prompts[idx])
                    retry_rule_map[len(retry_prompts) - 1] = rule_id
                    continue
                
                progress.update(task, advance=1)

        # 如果有需要重试的提示词，清除缓存并重试
        if retry_prompts:
            logger.info(f"开始重试 {len(retry_prompts)} 条失败的请求...")
            
            # 清除缓存
            _clear_cache_for_prompts(retry_prompts, config)
            
            # 重新发送请求
            retry_responses = asyncio.run(utils_chat.batch_process_prompts(retry_prompts, config, batch_size=10000))
            
            # 处理重试响应
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("处理重试响应...", total=len(retry_responses))
                
                for idx, response in enumerate(retry_responses):
                    rule_id = retry_rule_map[idx]
                    try:
                        selected_properties = validate_and_format_response(response)
                        
                        if selected_properties:
                            for prop in selected_properties:
                                try:
                                    cursor.execute("""
                                        INSERT INTO 结果_6_规范属性选择策略
                                        (规则序号, property_set_name, property_name, 优先级, 策略说明)
                                        VALUES (?, ?, ?, ?, ?)
                                    """, (
                                        rule_id,
                                        prop['property_set'],
                                        prop['property'],
                                        prop['priority'],
                                        prop['explanation']
                                    ))
                                except sqlite3.IntegrityError:
                                    pass
                            
                            conn.commit()
                            logger.trace(f"重试成功：规则序号 {rule_id} 的属性选择策略已保存")
                        else:
                            logger.error(f"重试后规则序号 {rule_id} 的响应仍然验证失败")
                            
                    except Exception as e:
                        logger.error(f"重试处理规则序号 {rule_id} 的响应时出错: {str(e)}")
                        continue
                    
                    progress.update(task, advance=1)

    except Exception as e:
        logger.error(f"保存属性选择策略时发生错误: {str(e)}")
    finally:
        conn.close()

def save_relevant_properties_with_strategy(config):
    """使用预定义的规范策略选择相关属性"""
    logger.info("开始使用预定义规范策略选择相关属性...")
    
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    cursor = conn.cursor()
    
    # 创建结果_7_相关属性表（如果不存在）
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS 结果_7_相关属性 (
        规则序号 INTEGER,
        ifc_guid TEXT,
        property_set_name TEXT,
        property_name TEXT,
        property_value TEXT,
        PRIMARY KEY (规则序号, ifc_guid, property_set_name, property_name),
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号),
        FOREIGN KEY (ifc_guid) REFERENCES IFC实体(guid)
    )
    """
    cursor.execute(create_table_sql)
    
    # 清空表中所有记录
    cursor.execute("DELETE FROM 结果_7_相关属性")
    conn.commit()
    
    # 获取结果_4_实体对齐
    query = """
    SELECT a.规则序号, a.ifc_guid
    FROM 结果_4_实体对齐 a
    """
    alignments = pd.read_sql(query, conn)
    
    # 获取结果_6_规范属性选择策略
    strategies = pd.read_sql("SELECT * FROM 结果_6_规范属性选择策略", conn)
    
    # 获取所有IFC属性集
    properties = pd.read_sql("SELECT * FROM IFC属性集", conn)
    
    # 使用rich进度条处理属性选择
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("选择相关属性...", total=len(alignments))
        
        for _, row in alignments.iterrows():
            try:
                rule_id = row['规则序号']
                ifc_guid = row['ifc_guid']
                
                # 获取该规范的属性策略
                rule_strategy = strategies[strategies['规则序号'] == rule_id].sort_values('优先级')
                
                # 获取实体的属性
                entity_properties = properties[properties['guid'] == ifc_guid]
                
                # 根据策略选择属性，使用更灵活的匹配逻辑
                for _, strategy in rule_strategy.iterrows():
                    # 预处理策略中的属性名，移除多余空格并转为小写
                    strategy_property = strategy['property_name'].strip().lower()
                    # 将策略属性名拆分成单词
                    strategy_words = [word for word in strategy_property.split() if word]
                    if strategy_words:
                        # 构建正则表达式模式，允许单词之间有任意字符
                        pattern = '.*'.join(map(re.escape, strategy_words))
                        
                        matching_properties = entity_properties[
                            (entity_properties['property_set_name'] == strategy['property_set_name']) &
                            (entity_properties['property_name'].str.strip().str.lower().str.contains(
                                pattern,
                                case=False,
                                na=False,
                                regex=True
                            ))
                        ]
                    
                    if not matching_properties.empty:
                        for _, prop in matching_properties.iterrows():
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO 结果_7_相关属性
                                    (规则序号, ifc_guid, property_set_name, property_name, property_value)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    rule_id,
                                    ifc_guid,
                                    prop['property_set_name'],
                                    prop['property_name'],
                                    prop['property_value']
                                ))
                                conn.commit()  # 每次插入后提交
                            except sqlite3.Error as e:
                                logger.warning(f"插入属性时出现SQL错误: {str(e)}, 继续处理下一条...")
                                continue
                
                logger.trace(f"规则序号 {rule_id}, IFC GUID {ifc_guid} 的相关属性已选择")
                progress.update(task, advance=1)
                
            except Exception as e:
                logger.error(f"处理规则 {rule_id} 的属性时出错: {str(e)}")
                progress.update(task, advance=1)
                continue
        
    conn.close()
    logger.success("使用预定义规范策略选择相关属性完成") 

def validate_and_format_response(response_text: str) -> list:
    """验证并格式化OpenAI API的响应"""
    try:
        properties = json_repair.loads(response_text)

        if get_depth(properties) == 1:
            if not isinstance(properties, list):
                logger.warning("响应格式错误：不是列表格式")
                return None
                
            validated_properties = []
            for prop in properties:
                if not all(key in prop for key in ['property_set', 'property', 'priority', 'explanation']):
                    logger.warning(f"属性缺少必要字段: {prop}")
                    return None
                    
                if not (1 <= prop['priority'] <= 5):
                    logger.warning(f"优先级超出范围(1-5): {prop}")
                    return None
                    
                validated_properties.append(prop)
                
            return validated_properties if validated_properties else None
            
        elif get_depth(properties) == 2:
            validated_properties = []
            for props in properties:
                for prop in props:
                    if not all(key in prop for key in ['property_set', 'property', 'priority', 'explanation']):
                        logger.warning(f"属性缺少必要字段: {prop}")
                        return None
                        
                    if not (1 <= prop['priority'] <= 5):
                        logger.warning(f"优先级超出范围(1-5): {prop}")
                        return None
                    
                    validated_properties.append(prop)
                
            return validated_properties if validated_properties else None
            
        else:
            logger.warning("响应格式错误：深度不符合预期")
            return None
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"验证响应时发生错误: {str(e)}")
        return None
    
def get_depth(lst):
    if not isinstance(lst, list):
        return 0
    if not lst:  # 空列表
        return 1
    return 1 + max(get_depth(item) for item in lst)