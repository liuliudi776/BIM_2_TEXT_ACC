"""
规范处理相关功能模块：识别规范类型、抽取命名实体、实体对齐等
"""

import datetime
import asyncio
import sqlite3
import pandas as pd
import json
from pathlib import Path
import tomli
import json_repair
from tqdm import tqdm
from utils.utils_control_logger import control_logger as logger
import utils.utils_chat as utils_chat
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
import ifcopenshell
import ifcopenshell.util.element as element
from utils.utils_cache_manager import _clear_cache_for_prompts

def batch_process_with_retry(prompts, config, data_items=None, max_retries=3, batch_size=10000,enforce_model_type="default"):
    """
    带有重试机制的批处理函数
    
    Args:
        prompts: 需要处理的提示列表
        config: 配置信息
        data_items: 与prompts对应的数据项（可选）
        max_retries: 最大重试次数
        batch_size: 批处理大小
    
    Returns:
        results: 处理结果列表
        failed_items: 处理失败的项目列表（包含索引和数据）
    """
    results = [None] * len(prompts)
    failed_indices = []
    failed_items = []
    
    # 第一次处理
    first_responses = asyncio.run(utils_chat.batch_process_prompts(prompts, config, batch_size=batch_size,enforce_model_type=enforce_model_type))
    
    # 检查结果，记录失败的项目
    for i, response in enumerate(first_responses):
        try:
            if response is None or not response.strip():
                raise Exception("空响应")
            json_repair.loads(response)  # 验证JSON是否有效
            results[i] = response
        except Exception as e:
            failed_indices.append(i)
            failed_items.append({
                'index': i,
                'prompt': prompts[i],
                'data': data_items[i] if data_items else None,
                'error': str(e)
            })
    
    # 重试失败的项目
    retry_count = 0
    while failed_indices and retry_count < max_retries:
        retry_count += 1
        logger.info(f"第 {retry_count} 次重试，处理 {len(failed_indices)} 个失败项...")
        
        retry_prompts = [prompts[i] for i in failed_indices]
        retry_responses = asyncio.run(utils_chat.batch_process_prompts(retry_prompts, config, batch_size=batch_size,enforce_model_type=enforce_model_type))
        
        # 更新结果，记录仍然失败的项目
        still_failed_indices = []
        still_failed_items = []
        
        for idx, (original_idx, response) in enumerate(zip(failed_indices, retry_responses)):
            try:
                if response is None or not response.strip():
                    raise Exception("空响应")
                json_repair.loads(response)  # 验证JSON是否有效
                results[original_idx] = response
            except Exception as e:
                still_failed_indices.append(original_idx)
                still_failed_items.append(failed_items[idx])
        
        failed_indices = still_failed_indices
        failed_items = still_failed_items
        
        if failed_indices:
            logger.warning(f"重试后仍有 {len(failed_indices)} 个项目失败")
    
    return results, failed_items

def recognize_rule_types(config):
    """基于数据库中的预定义类型识别规范类型"""
    logger.info("开始识别规范类型...")
    
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    
    # 创建新表来存储识别结果
    create_results_table = """
    CREATE TABLE IF NOT EXISTS 结果_1_规范类型识别 (
        规则序号 INTEGER PRIMARY KEY,
        规范来源 TEXT,
        条款编号 TEXT,
        规范内容 TEXT,
        识别类型 TEXT,
        识别时间 TIMESTAMP,
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
    )
    """
    conn.execute(create_results_table)
    
    # 获取规范内容和预定义的规范类型
    query_rules = "SELECT * FROM 输入_规范列表"
    query_types = "SELECT * FROM 输入_预定义规范类型"
    df_rules = pd.read_sql(query_rules, conn)
    df_types = pd.read_sql(query_types, conn)
    
    # 获取已识别的规则序号
    existing_rules = pd.read_sql("SELECT 规则序号 FROM 结果_1_规范类型识别", conn)
    existing_rule_ids = set(existing_rules['规则序号'])
    
    # 筛选未识别的规范
    unprocessed_rules = df_rules[~df_rules['规则序号'].astype(str).isin(existing_rule_ids)]
    
    if unprocessed_rules.empty:
        logger.info("所有规范都已完成类型识别，无需重复处理")
        conn.close()
        return
        
    logger.info(f"发现 {len(unprocessed_rules)} 条未识别的规范，开始处理...")
    
    # 构建包含所有规范类型信息的提示
    type_info = df_types.apply(
        lambda x: f"类型：{x['条文类型']}\n描述：{x['描述']}\n示例：{x['示例']}", 
        axis=1
    ).str.cat(sep='\n\n')
    
    # 准备插入语句
    insert_sql = """
    INSERT OR REPLACE INTO 结果_1_规范类型识别 
    (规则序号, 规范来源, 条款编号, 规范内容, 识别类型, 识别时间)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    
    prompts = []
    rows_list = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("识别规范类型prompts收集", total=len(unprocessed_rules))
        for idx, row in unprocessed_rules.iterrows():
            rule_content = row['内容']
            prompt = f"""请根据以下预定义的规范类型，判断给定内容属于哪种类型.
            
预定义的规范类型：
{type_info}

待分类的规范内容：
"{rule_content}"

请仅返回对应的规范类型名称，必须且仅返回一个最符合的类型名称。
返回格式为json格式：
{{"规范类型": "规范类型名称"}}"""
            prompts.append(prompt)
            rows_list.append(row)
            progress.advance(task)

    responses, failed_items = batch_process_with_retry(prompts, config, rows_list)
    
    if failed_items:
        logger.warning(f"有 {len(failed_items)} 条规范类型识别失败:")
        for item in failed_items:
            logger.warning(f"规则序号: {item['data']['规则序号']}, 错误: {item['error']}")
    
    for row, response in zip(rows_list, responses):
        type_name = json_repair.loads(response)['规范类型']
        logger.trace(f"规范类型识别结果: {type_name}")
        current_time = datetime.datetime.now()
        try:
            conn.execute(insert_sql, (
                row['规则序号'],
                row['规范来源'],
                row['条款编号'],
                row['内容'],
                type_name,
                current_time
            ))
            conn.commit()
            logger.trace(f"条款序号: {row['规则序号']}, 条款编号: {row['条款编号']}, 识别的类型: {type_name}, 已保存到数据库")
        except Exception as e:
            logger.error(f"保存识别结果失败: {str(e)}")

    conn.close()

def recognize_ner_candidates(config, ner_r=1):
    """识别每条规范中的预定义候选实体，并使用BIO标注或直接使用标注表数据
        ner_r参数为控制是否使用LLM进行NER识别
        ner_r = 1 时，直接使用标注表的数据
        ner_r = 0 时，使用LLM进行NER识别
    """
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    cursor = conn.cursor()
    
    # 创建新表来存储NER识别结果
    create_results_table = """
    CREATE TABLE IF NOT EXISTS 结果_2_规范实体识别 (
        规则序号 INTEGER,
        实体文本 TEXT,
        实体类型 TEXT,
        开始位置 INTEGER,
        结束位置 INTEGER,
        识别时间 TIMESTAMP,
        PRIMARY KEY (规则序号, 实体文本),
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
    )
    """
    cursor.execute(create_results_table)
    
    # 清空现有结果
    cursor.execute("DELETE FROM 结果_2_规范实体识别")
    conn.commit()
    
    if ner_r == 1:
        logger.info("使用标注表 '标注_规范实体识别' 的数据进行实体识别")
        query_labeled = "SELECT * FROM 标注_规范实体识别"
        df_labeled = pd.read_sql(query_labeled, conn)
        
        if df_labeled.empty:
            logger.warning("标注表 '标注_规范实体识别' 没有数据。")
        else:
            # 重命名 '标注时间' 为 '识别时间'
            df_labeled = df_labeled.rename(columns={"标注时间": "识别时间"})
            
            # 选择需要的列
            columns_needed = ["规则序号", "实体文本", "实体类型", "开始位置", "结束位置", "识别时间"]
            df_labeled = df_labeled[columns_needed]
            
            # 插入到 '结果_2_规范实体识别' 表
            df_labeled.to_sql('结果_2_规范实体识别', conn, if_exists='append', index=False)
            logger.info(f"已从标注表中导入 {len(df_labeled)} 条实体识别结果。")
        
        conn.close()
        logger.success("使用标注表的数据进行实体识别完成")
        return
    else:
        # 继续原有的LLM NER识别逻辑
        # 获取规范内容和预定义的规范类型
        query_rules = "SELECT * FROM 输入_规范列表"
        query_tags = "SELECT * FROM 输入_预定义候选实体标签"
        df_rules = pd.read_sql(query_rules, conn)
        df_tags = pd.read_sql(query_tags, conn)
        
        logger.info(f"开始处理 {len(df_rules)} 条规范...")
        
        # 构建包含所有规范类型信息的提示
        type_info = df_tags.apply(
            lambda x: f"{x['标签']}: {x['描述']}", 
            axis=1
        ).str.cat(sep='\n')
        
        # 准备插入语句
        insert_sql = """
        INSERT OR REPLACE INTO 结果_2_规范实体识别
        (规则序号, 实体文本, 实体类型, 开始位置, 结束位置, 识别时间)
        VALUES (?, ?, ?, ?, ?, ?)
        """
    
        # 加载工程知识库
        try:
            with open(config["knowledge"]["path"], "rb") as f:
                knowledge_base = tomli.load(f)
        except Exception as e:
            logger.error(f"加载知识库失败: {str(e)}")
            return
    
        prompts = []
        rows_list = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("规范实体识别prompt收集", total=len(df_rules))
            for idx, row in df_rules.iterrows():
                rule_content = row['内容']
                prompt = f"""
    专业知识（对象类型识别）：
    {knowledge_base["ner_general"].get("对象类型识别注意事项", {})}
    请根据专业知识对以下文本进行实体识别，并返回JSON格式的结果。每个实体包含文本、类型、开始和结束位置.           
    待识别文本: "{rule_content}"
    可用的实体类型及其描述:
    {type_info}
    注意：不属于上面任何类型的实体不要识别（例如属性、参数等）。
    
    请按以下格式返回（仅返回JSON，不要其他内容）：
    [{{"text": "实体文本", "type": "实体类型", "start": 开始位置, "end": 结束位置}}]"""
                prompts.append(prompt)
                rows_list.append(row)
                progress.advance(task)
    
        responses, failed_items = batch_process_with_retry(prompts, config, rows_list)
        
        if failed_items:
            logger.warning(f"有 {len(failed_items)} 条实体识别失败:")
            for item in failed_items:
                logger.warning(f"规则序号: {item['data']['规则序号']}, 错误: {item['error']}")
        
        for row, response in zip(rows_list, responses):
            try:
                # 确保 response 不为空且是有效的 JSON 字符串
                if not response or not response.strip():
                    raise ValueError("空响应")
                    
                # 解析 JSON 并验证格式
                entities = json_repair.loads(response)
                if not isinstance(entities, list):
                    raise ValueError(f"返回格式错误，期望列表类型，实际返回: {type(entities)}，实际返回内容: {entities}")
                    
                current_time = datetime.datetime.now()
                
                for entity in entities:
                    if not isinstance(entity, dict):
                        raise ValueError(f"实体格式错误，期望字典类型，实际返回: {type(entity)}，实际返回内容: {entity}")
                        
                    # 验证必需的字段
                    required_fields = {'text', 'type', 'start', 'end'}
                    if not all(field in entity for field in required_fields):
                        missing_fields = required_fields - set(entity.keys())
                        raise ValueError(f"实体缺少必需字段: {missing_fields}")
                    
                    # 验证字段类型
                    if not isinstance(entity['text'], str):
                        raise ValueError(f"text 字段必须是字符串类型，实际类型: {type(entity['text'])}")
                    if not isinstance(entity['type'], str):
                        raise ValueError(f"type 字段必须是字符串类型，实际类型: {type(entity['type'])}")
                    if not isinstance(entity['start'], (int, float)):
                        raise ValueError(f"start 字段必须是数字类型，实际类型: {type(entity['start'])}")
                    if not isinstance(entity['end'], (int, float)):
                        raise ValueError(f"end 字段必须是数字类型，实际类型: {type(entity['end'])}")
                    
                    try:
                        conn.execute(insert_sql, (
                            row['规则序号'],
                            entity['text'],
                            entity['type'],
                            int(entity['start']),  # 确保转换为整数
                            int(entity['end']),    # 确保转换为整数
                            current_time
                        ))
                        conn.commit()
                    except sqlite3.Error as e:
                        logger.error(f"数据库插入失败 - 规则序号: {row['规则序号']}, 错误: {str(e)}")
                        continue
                        
                logger.trace(f"规则序号: {row['规则序号']} 的实体识别结果已保存")
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 - 规则序号: {row['规则序号']}, 响应内容: {response[:100]}..., 错误: {str(e)}")
                continue
            except ValueError as e:
                logger.warning(f"数据验证失败 - 规则序号: {row['规则序号']}, 错误: {str(e)}")
                # 清除失败项目的缓存并重试
                retry_prompts = [prompts[i] for i in range(len(prompts)) if rows_list[i]['规则序号'] == row['规则序号']]
                _clear_cache_for_prompts(retry_prompts, config)
                # 重新处理该规则的所有提示
                retry_responses = asyncio.run(utils_chat.batch_process_prompts(retry_prompts, config))
                # 更新结果
                for i, response in enumerate(retry_responses):
                    try:
                        if not response or not response.strip():
                            raise ValueError("重试后仍为空响应")
                        entities = json_repair.loads(response)
                        if not isinstance(entities, list):
                            raise ValueError(f"重试后返回格式仍错误，期望列表类型，实际返回: {type(entities)}")
                        # 验证和保存重试结果
                        current_time = datetime.datetime.now()
                        for entity in entities:
                            conn.execute(insert_sql, (
                                row['规则序号'],
                                entity['text'],
                                entity['type'], 
                                int(entity['start']),
                                int(entity['end']),
                                current_time
                            ))
                        conn.commit()
                        logger.info(f"规则 {row['规则序号']} 重试成功")
                    except Exception as e:
                        logger.error(f"规则 {row['规则序号']} 重试仍然失败: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"处理规则 {row['规则序号']} 时发生未知错误: {str(e)}")
                logger.error(f"响应内容: {response[:200]}")  # 添加更多日志信息
                continue
    
        conn.close()
        logger.success("识别每条规范中的预定义候选实体完成")

def align_entities(config, ifc_class_r=1):
    """基于规则和知识库进行实体对齐，并统计is_relevant状态
    Args:
        config: 配置信息
        ifc_class_r: IFC类的对齐方法实现，0为默认使用LLM进行对齐，1为使用映射表进行对齐
    """
    logger.info("开始进行实体对齐...")

    # 加载知识库
    try:
        with open(config["knowledge"]["path"], "rb") as f:
            knowledge_base = tomli.load(f)
        logger.info("成功加载工程知识库")
    except Exception as e:
        logger.error(f"加载知识库失败: {str(e)}")
        return

    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    cursor = conn.cursor()

    # 创建对齐结果表
    create_alignment_table = """
    CREATE TABLE IF NOT EXISTS 结果_4_实体对齐 (
        规则序号 INTEGER,
        规范实体文本 TEXT,
        规范实体类型 TEXT,
        ifc_guid TEXT,
        ifc_entity_with_type TEXT,
        匹配原因 TEXT,
        属性集 TEXT,
        对齐时间 TIMESTAMP,
        PRIMARY KEY (规则序号, 规范实体文本, ifc_guid),
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号),
        FOREIGN KEY (ifc_guid) REFERENCES IFC实体(guid)
    )
    """
    cursor.execute(create_alignment_table)

    # 创建统计表
    create_stats_table = """
    CREATE TABLE IF NOT EXISTS 结果_5_is_relevant统计 (
        规则序号 INTEGER,
        IFC_GUID TEXT,
        is_relevant BOOLEAN,
        识别时间 TIMESTAMP,
        规范内容 TEXT,
        IFC实体名称 TEXT,
        IFC实体类型 TEXT,
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号),
        FOREIGN KEY (IFC_GUID) REFERENCES IFC实体(guid)
    )
    """
    cursor.execute(create_stats_table)
    conn.commit()

    cursor.execute("DELETE FROM 结果_4_实体对齐")
    cursor.execute("DELETE FROM 结果_5_is_relevant统计")
    conn.commit()

    # 获取规范实体和IFC实体数据
    ner_query = "SELECT * FROM 结果_2_规范实体识别"
    ifc_query = "SELECT * FROM IFC实体"

    ner_entities = pd.read_sql(ner_query, conn)
    ifc_entities = pd.read_sql(ifc_query, conn)

    # 获取规范内容
    rules_query = "SELECT * FROM 输入_规范列表"
    rules_df = pd.read_sql(rules_query, conn)

    # 新增：加载包含关系识别结果，用于获取关系图信息
    try:
        inclusion_df = pd.read_sql("SELECT * FROM 结果_3_包含关系识别", conn)
        inclusion_map = {row['规则序号']: row['包含关系'] for _, row in inclusion_df.iterrows()}
    except Exception as e:
        inclusion_map = {}
        logger.warning(f"加载包含关系识别结果表失败: {str(e)}")

    # 加载IFC文件
    try:
        ifc_file = ifcopenshell.open(config["ifc"]["model_path"])
        logger.info("成功加载IFC文件")
    except Exception as e:
        logger.error(f"加载IFC文件失败: {str(e)}")
        return

    # 根据ifc_class_r进行不同处理
    if ifc_class_r == 0:
        # ----------------- 使用LLM进行IFC类识别 -----------------
        outer_prompts = []
        ner_data = []  # 存储 (ner_row, rule_content)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("实体对齐prompt收集", total=len(ner_entities))
            for _, ner_row in ner_entities.iterrows():
                rule_id = ner_row['规则序号']
                rule_content = rules_df[rules_df['规则序号'] == int(rule_id)]['内容'].iloc[0]
                # 读取额外信息
                add_info_path = Path(config["add_info_file"]["path"])
                try:
                    with open(add_info_path, "rb") as f:
                        additional_info = tomli.load(f)
                except Exception as e:
                    logger.warning(f"无法加载额外信息文件: {str(e)}")
                    additional_info = {}
                # 为每种IFC类型获取一个示例实例
                ifc_type_examples = []
                for ifc_type in ifc_entities['ifc_type'].unique():
                    example = ifc_entities[ifc_entities['ifc_type'] == ifc_type].iloc[0]
                    example_text = f"{ifc_type} - 示例: {example['name']} ({example['description'] or '无描述'})"
                    ifc_type_examples.append(example_text)
                outer_prompt = f"""请基于专业知识判断以下规范实体应该对应哪些类型的IFC实体，注意是规范实体对应的IFC实体，而不是规范全文对应的IFC实体，要严格区分。

规范内容：{rule_content}
规范实体：{ner_row['实体文本']} (类型: {ner_row['实体类型']})

可用的IFC实体类型:
{knowledge_base["IFC_CLASS_DESCRIPTION"]}

ner指引参考:
{knowledge_base["ner_general"].get('IFC实体类型识别注意事项', {})}

请返回JSON格式:
{{
    "target_ifc_types": "IFC类型",
    "matching_reason": "选择这些类型的原因说明"
}}"""
                outer_prompts.append(outer_prompt)
                ner_data.append((ner_row, rule_content))
                progress.advance(task)

        # 批量调用GPT获取外部返回结果，批量并发最大数为10
        outer_responses, failed_outer = batch_process_with_retry(outer_prompts, config, ner_data, enforce_model_type="default")
        # ----------------- 外部批处理结束 -----------------

    elif ifc_class_r == 1:
        # ----------------- 使用映射表进行IFC类识别 -----------------
        logger.info("使用映射表进行IFC类识别...")
        # 加载映射表数据
        mapping_query = "SELECT * FROM 标注_规范实体_IfcClass映射"
        try:
            mapping_df = pd.read_sql(mapping_query, conn)
            logger.info("成功加载标注_规范实体_IfcClass映射表")
        except Exception as e:
            logger.error(f"加载映射表失败: {str(e)}")
            return

        # 将映射表转换为字典，键为(规则序号, 实体文本, 实体类型)，值为IFC_Class
        mapping_dict = {}
        for _, row in mapping_df.iterrows():
            key = (row['规则序号'], row['实体文本'], row['实体类型'])
            mapping_dict[key] = row['IFC_Class']

        outer_responses = []
        failed_outer = []
        ner_data = []  # 存储 (ner_row, rule_content)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("实体对齐（映射表）", total=len(ner_entities))
            for _, ner_row in ner_entities.iterrows():
                rule_id = ner_row['规则序号']
                entity_text = ner_row['实体文本']
                entity_type = ner_row['实体类型']
                rule_content = rules_df[rules_df['规则序号'] == int(rule_id)]['内容'].iloc[0]
                key = (rule_id, entity_text, entity_type)
                if key in mapping_dict:
                    ifc_class_str = mapping_dict[key]
                    # 假设IFC_Class字段中多个类型以逗号分隔
                    target_ifc_types = [cls.strip() for cls in ifc_class_str.split(',')]
                    matching_reason = "使用映射表中的IFC_Class映射"
                    outer_response = json.dumps({
                        "target_ifc_types": target_ifc_types,
                        "matching_reason": matching_reason
                    }, ensure_ascii=False)
                    outer_responses.append(outer_response)
                else:
                    # 如果没有找到映射，则记录失败
                    failed_outer.append({
                        'data': {'ner_row': ner_row, 'rule_content': rule_content},
                        'error': "未在映射表中找到对应的IFC_Class"
                    })
                    outer_responses.append(None)  # 保持索引对齐
                ner_data.append((ner_row, rule_content))
                progress.advance(task)

    else:
        logger.error(f"无效的ifc_class_r值: {ifc_class_r}")
        conn.close()
        return

    if ifc_class_r == 0 and failed_outer:
        logger.warning(f"有 {len(failed_outer)} 条外部实体对齐失败:")
        for item in failed_outer:
            logger.warning(f"规则序号: {item['data'][0]['规则序号']}, 实体文本: {item['data'][0]['实体文本']}, 错误: {item['error']}")

    # 收集所有内部候选实体提示
    all_inner_prompts = []
    all_candidate_info = []  # 存储(ner_row, entity, rule_content, outer_result)用于后续匹配

    # 读取额外信息
    add_info_path = Path(config["add_info_file"]["path"])
    try:
        with open(add_info_path, "rb") as f:
            additional_info = tomli.load(f)
    except Exception as e:
        logger.warning(f"无法加载额外信息文件: {str(e)}")
        additional_info = {}

    for idx, (ner_row, rule_content) in enumerate(ner_data):
        if ifc_class_r == 0:
            outer_response = outer_responses[idx]
            if outer_response is None:
                continue  # 跳过失败的外部对齐
            try:
                result_outer = json.loads(json_repair.repair_json(outer_response))
            except Exception as e:
                logger.error(f"解析外部响应失败: {str(e)}")
                continue
        elif ifc_class_r == 1:
            outer_response_str = outer_responses[idx]
            if outer_response_str is None:
                continue  # 跳过没有映射的情况
            try:
                result_outer = json.loads(outer_response_str)
            except Exception as e:
                logger.error(f"解析映射表中的外部响应失败: {str(e)}")
                continue
        else:
            continue  # 已在前面处理无效的ifc_class_r

        target_types = result_outer.get('target_ifc_types')
        matching_reason = result_outer.get('matching_reason', "无匹配原因说明")
        if isinstance(target_types, list):
            matching_entities = ifc_entities[ifc_entities['ifc_type'].isin(target_types)]
        else:
            matching_entities = ifc_entities[ifc_entities['ifc_type'] == str(target_types)]

        for _, entity in matching_entities.iterrows():
            try:
                ifc_entity = ifc_file.by_guid(entity['guid'])
                if ifc_entity and hasattr(ifc_entity, 'is_a'):  # 检查是否为有效的IFC实体
                    psets = {}
                    # 获取直接属性
                    direct_attributes = {name: getattr(ifc_entity, name) 
                                       for name in ifc_entity.wrapped_data.get_attribute_names()}
                    psets['直接属性'] = direct_attributes

                    # 获取属性集
                    try:
                        entity_psets = element.get_psets(ifc_entity)
                        if entity_psets:
                            psets['属性集'] = entity_psets
                    except Exception as e:
                        logger.warning(f"获取属性集失败: {str(e)}")

                    attribute_set_str = json.dumps(psets, ensure_ascii=False, indent=2, default=str)
                else:
                    attribute_set_str = "{}"
                    logger.warning(f"无法找到GUID为 {entity['guid']} 的有效IFC实体")
            except Exception as e:
                logger.error(f"处理实体失败: {str(e)}")
                attribute_set_str = "{}"

            entity_info = f"""
实体类型: {entity['ifc_type']}
实体名称: {entity['name']}
实体描述: {entity['description'] or '无描述'}
属性集: 
{attribute_set_str}
            """
            # 新增：根据当前规范的规则序号获取对应的关系图信息
            relationship_graph_info = inclusion_map.get(ner_row['规则序号'], "无关系图信息")

            inner_prompt = f"""请分析以下IFC实体是否与规范要求相关。
判断步骤：
1. 关系图：
    若实体类型出现在关系图中（无论作为 container 还是 contained），大概率判定为相关（is_relevant = true）。
    然而，需要进一步确认实体名称是否与规范要求的对象类型匹配。

2. 仅当实体类型确实未在关系图中出现时，才继续判断：
   a. 对比规范中描述的对象特征与IFC实体的属性是否匹配
   b. 只有在关系图中未出现，且属性不匹配时，才能判定为不相关(is_relevant=false)
   c. 例如楼梯和楼板并非同一类型，不能因为出现在关系图中就直接判定为相关。(is_relevant=false)

特别强调：
1. 关系图中出现的实体类型，即使其属性中缺少信息，也必须判定为相关
2. "相关"包括直接关系和间接关系（如container-contained关系）
3. 关系图中的实体名称可能与IFC实体的名称不一致，需要根据IFC实体的名称进行判断，若IFC实体的名称与关系图中的实体名称不一致，则判定为不相关(is_relevant=false)。
4. 如果实体名称与规范内容中的审查内容不一致，则判定为不相关(is_relevant=false)。

规范内容：
{rule_content}

ner指引参考:
{knowledge_base["ner_general"].get('IFC实体类型识别注意事项', {})}

领域常识:
{knowledge_base.get('global', {})}

IFC实体信息:
{entity_info}

额外信息:
{additional_info.get('global', {})}

关系图:
{relationship_graph_info}

请仅返回JSON格式:
{{
    "reason": "判断原因的详细说明",
    "is_relevant": true/false
}}"""
            all_inner_prompts.append(inner_prompt)
            all_candidate_info.append({
                'ner_row': ner_row,
                'entity': entity,
                'rule_content': rule_content,
                'outer_result': result_outer
            })

    # 批量处理所有内部候选实体提示
    logger.info(f"开始批量处理 {len(all_inner_prompts)} 个内部候选实体...")
    all_inner_responses, failed_inner = batch_process_with_retry(
        all_inner_prompts, 
        config, 
        all_candidate_info,
        enforce_model_type="default",
    )

    if failed_inner:
        logger.warning(f"有 {len(failed_inner)} 条内部实体对齐失败:")
        for item in failed_inner:
            ner_row = item['data']['ner_row']
            entity = item['data']['entity']
            logger.warning(f"规则序号: {ner_row['规则序号']}, IFC GUID: {entity['guid']}, 错误: {item['error']}")

    # 处理结果并写入数据库
    current_time = datetime.datetime.now()
    for candidate_info, inner_response in zip(all_candidate_info, all_inner_responses):
        try:
            result_inner = json_repair.loads(json_repair.repair_json(inner_response))
            if type(result_inner) == list:
                result_inner = result_inner[-1]
            if result_inner.get('is_relevant'):
                # 使用 IfcOpenShell 获取属性集
                ifc_entity = ifc_file.by_guid(candidate_info['entity']['guid'])
                if ifc_entity:
                    # 获取所有属性集和属性
                    psets = {}

                    # 获取直接属性
                    direct_attributes = {name: getattr(ifc_entity, name) for name in ifc_entity.wrapped_data.get_attribute_names()}
                    psets['直接属性'] = direct_attributes

                    # 获取属性集
                    try:
                        entity_psets = element.get_psets(ifc_entity)
                        if entity_psets:
                            psets['属性集'] = entity_psets
                    except Exception as e:
                        logger.warning(f"获取属性集失败: {str(e)}")

                    # 转换为JSON字符串，确保处理特殊字符
                    psets_json = json.dumps(psets, ensure_ascii=False, default=str)
                else:
                    psets_json = "{}"

                cursor.execute("""
                    INSERT INTO 结果_4_实体对齐
                    (规则序号, 规范实体文本, 规范实体类型, ifc_guid, ifc_entity_with_type, 匹配原因, 属性集, 对齐时间)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate_info['ner_row']['规则序号'],
                    candidate_info['ner_row']['实体文本'],
                    candidate_info['ner_row']['实体类型'],
                    candidate_info['entity']['guid'],
                    f"{candidate_info['entity']['name']} ({candidate_info['entity']['ifc_type']})",
                    candidate_info['outer_result'].get('matching_reason', "无匹配原因"),
                    psets_json,  # 使用 IfcOpenShell 获取的属性集
                    current_time
                ))
                conn.commit()
                logger.trace(f"实体 {candidate_info['entity']['guid']} 与规范相关: {result_inner.get('reason')}")

                # 记录到统计表
                cursor.execute("""
                    INSERT INTO 结果_5_is_relevant统计
                    (规则序号, IFC_GUID, is_relevant, 识别时间, 规范内容, IFC实体名称, IFC实体类型)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate_info['ner_row']['规则序号'],
                    candidate_info['entity']['guid'],
                    True,
                    current_time,
                    candidate_info['rule_content'],
                    candidate_info['entity']['name'],
                    candidate_info['entity']['ifc_type']
                ))
                conn.commit()
            else:
                logger.trace(f"实体 {candidate_info['entity']['guid']} 与规范无关: {result_inner.get('reason')}")

                # 记录到统计表
                cursor.execute("""
                    INSERT INTO 结果_5_is_relevant统计
                    (规则序号, IFC_GUID, is_relevant, 识别时间, 规范内容, IFC实体名称, IFC实体类型)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate_info['ner_row']['规则序号'],
                    candidate_info['entity']['guid'],
                    False,
                    current_time,
                    candidate_info['rule_content'],
                    candidate_info['entity']['name'],
                    candidate_info['entity']['ifc_type']
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"处理实体对齐结果时出错: {str(e)}")
            continue

    conn.close()
    logger.success("实体对齐完成，并统计了is_relevant状态")

def recognize_inclusion_relationships(config):
    """基于GPT识别规范中各个实体之间的包含关系，并保存到数据库中"""
    logger.info("开始识别规范实体之间的包含关系...")

    # 连接数据库
    conn = sqlite3.connect(config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db")
    
    # 创建新的包含关系结果表
    create_inclusion_table = """
    CREATE TABLE IF NOT EXISTS 结果_3_包含关系识别 (
        规则序号 INTEGER PRIMARY KEY,
        包含关系 TEXT,
        识别时间 TIMESTAMP,
        FOREIGN KEY (规则序号) REFERENCES 输入_规范列表(规则序号)
    )
    """
    conn.execute(create_inclusion_table)

    # 查询所有规范和对应的识别实体结果
    df_rules = pd.read_sql("SELECT * FROM 输入_规范列表", conn)
    df_entities = pd.read_sql("SELECT * FROM 结果_2_规范实体识别", conn)

    # 按规则分组，收集每个规范对应的实体列表（只处理包含至少两个实体的规范）
    grouped = df_entities.groupby('规则序号')
    valid_rule_ids = [rule_id for rule_id, group in grouped if len(group) >= 2]
    
    if not valid_rule_ids:
        logger.info("没有符合条件的规范（至少包含两个实体），无需进行包含关系识别")
        conn.close()
        return

    prompts = []
    rules_info = []  # 存储每条规范的信息，用于后续更新数据库
    for rule_id in valid_rule_ids:
        # 获取对应规范内容
        rule_row = df_rules[df_rules['规则序号'] == rule_id]
        if rule_row.empty:
            continue
        rule_content = rule_row['内容'].iloc[0]
        # 获取所有实体信息的列表
        entity_rows = grouped.get_group(rule_id)
        entities_list = []
        for _, ent in entity_rows.iterrows():
            entities_list.append({
                "text": ent["实体文本"],
                "type": ent["实体类型"],
                "start": ent["开始位置"],
                "end": ent["结束位置"]
            })
        # 构建GPT提示：要求返回实体之间的包含关系，
        # 格式要求：返回JSON格式的列表，列表中每个元素为字典，字典格式如下：
        # {"container": "包含方实体文本", "contained": "被包含实体文本", "relation": "包含"}
        prompt = r"""
请判断给定实体间是否存在包含关系。包含关系定义如下：

包含关系类型：
1. 空间从属：整体与部分的关系（如：公路包含桥梁、建筑包含楼层）
2. 层次包含：上级空间对下级空间的包含（如：园区包含建筑、楼层包含房间）
3. 设施依附：空间与其中固定设施的关系（如：房间包含设备、车站包含站台）

判断规则：
1. 实体A包含实体B，实体B包含实体C，则实体A也包含实体C（传递性）
2. 同一实体可以被多个实体包含，也可以包含多个实体
3. 包含关系需基于文本语境和实体类型进行判断
4. 地理空间、建筑设施、交通基础设施等均可构成包含关系

返回格式：
[
    {
        "container": "包含方实体文本",
        "contained": "被包含实体文本",
        "relation": "包含"
    }
]

返回规则：
- 单个实体：返回空列表 []
- 两个实体：存在包含关系则返回该关系，否则返回空列表 []
- 多个实体：返回所有存在的直接和传递包含关系
- 不存在包含关系：返回空列表 []

示例：
输入文本："园区内的3号楼的实验室"
实体列表：[{"text": "园区"}, {"text": "3号楼"}, {"text": "实验室"}]
返回结果：
[
    {"container": "园区", "contained": "3号楼", "relation": "包含"},
    {"container": "3号楼", "contained": "实验室", "relation": "包含"},
    {"container": "园区", "contained": "实验室", "relation": "包含"}
]

请根据以上规则，判断给定文本和实体列表中的包含关系：
"""+f"""
规范内容：
"{rule_content}"

实体列表：
{json.dumps(entities_list, ensure_ascii=False, indent=2)}

请仅返回JSON，不要包含其他说明内容。请严格按照上述要求判断并返回结果。
"""
        prompts.append(prompt)
        rules_info.append({ "规则序号": rule_id })

    # 使用批处理函数调用GPT进行包含关系识别
    responses, failed_items = batch_process_with_retry(prompts, config, rules_info)
    
    if failed_items:
        logger.warning(f"有 {len(failed_items)} 条规范的包含关系识别失败:")
        for item in failed_items:
            logger.warning(f"规则序号: {item['data']['规则序号']}, 错误: {item['error']}")
    
    current_time = datetime.datetime.now()
    insert_sql = """
    INSERT OR REPLACE INTO 结果_3_包含关系识别
    (规则序号, 包含关系, 识别时间)
    VALUES (?, ?, ?)
    """
    
    # 处理每条包含关系识别结果并保存到数据库
    for rule_info, response in zip(rules_info, responses):
        rule_id = rule_info["规则序号"]
        try:
            # 解析返回的JSON结果
            inclusion_relations = json_repair.loads(response)
            if not isinstance(inclusion_relations, list):
                raise ValueError(f"返回格式错误，期望列表类型，实际返回: {type(inclusion_relations)}")
            # 将包含关系保存为格式化的JSON字符串
            inclusion_json = json.dumps(inclusion_relations, ensure_ascii=False, indent=2)
            conn.execute(insert_sql, (
                rule_id,
                inclusion_json,
                current_time
            ))
            conn.commit()
            logger.trace(f"规则 {rule_id} 的包含关系已保存")
        except Exception as e:
            logger.error(f"规则 {rule_id} 包含关系识别结果处理失败: {str(e)}")
    
    conn.close()
    logger.success("规范包含关系识别完成")

