import sqlite3

def create_acc_report_tables(config):
    """
    创建报告相关的视图
    
    Args:
        config: 包含数据库配置信息的字典
    """
    # 构建数据库路径
    db_path = config["database"]["path"].replace("data.db", config["current_model"]["type"].lower()) + "/data.db"
    # 连接数据库创建报告视图
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 创建报告视图
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS 报告_自然语言生成与合规性审查合并视图 AS
        SELECT 
            s.规则序号,
            s.内容,
            COALESCE(g.组序号, '0') as 组序号,
            g.规范实体组,
            g.IFC实体组,
            g.路径组,
            n.description as 自然语言描述,
            n.原始属性集,
            analysis_process as 分析过程,
            c.judgment_result as 判断结果
        FROM 输入_规范列表 s
        LEFT JOIN 结果_5_规范元素组 g 
            ON s.规则序号 = g.规则序号
        LEFT JOIN 结果_8_属性自然语言描述 n 
            ON s.规则序号 = n.规则序号 
            AND g.组序号 = n.组序号
        LEFT JOIN 结果_9_合规性审查 c
            ON s.规则序号 = c.规则序号
            AND g.组序号 = c.组序号
        """)
        
        conn.commit()

def parse_property_str(property_str):
    """
    将属性字符串解析为JSON格式
    
    Args:
        property_str: 原始属性字符串
        
    Returns:
        dict: 解析后的JSON对象
    """
    result = {}
    current_entity = None
    current_pset = None
    
    # 按行分割
    lines = property_str.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # 跳过空行
        if not line:
            continue
            
        # 解析实体行
        if line.startswith('实体'):
            entity_parts = line.split('(')
            entity_id = entity_parts[0].split()[1].strip()
            entity_type = entity_parts[-1].strip('()').strip()
            current_entity = {
                'id': entity_id,
                'type': entity_type,
                'property_sets': {}
            }
            result[entity_id] = current_entity
            continue
            
        # 解析属性行
        if line.startswith('属性集:'):
            parts = line.split(',')
            current_pset = parts[0].split(':')[1].strip()
            if current_pset not in current_entity['property_sets']:
                current_entity['property_sets'][current_pset] = {}
                
            # 解析属性和值
            prop_name = parts[1].split(':')[1].strip()
            prop_value = parts[2].split(':')[1].strip()
            current_entity['property_sets'][current_pset][prop_name] = prop_value
            
    return result

def main(config):
    """
    主函数
    
    Args:
        config: 配置参数字典
    """
    create_acc_report_tables(config)
